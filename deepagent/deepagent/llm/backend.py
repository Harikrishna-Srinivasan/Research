"""LLM Backend — load and run state-of-the-art open-source models via transformers."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from deepagent.config import LLMConfig
from deepagent.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------
class LLMBackend(ABC):
    """Interface every LLM backend must implement."""

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate a text completion from a chat message list."""

    @abstractmethod
    def generate_with_tools(
        self, messages: list[dict], tools: list[dict], **kwargs
    ) -> dict:
        """Generate a completion that may include a tool call.

        Returns a dict with keys:
            - "content": str | None   (text response)
            - "tool_calls": list[dict] | None
              each tool_call: {"name": str, "arguments": dict}
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return approximate token count for *text*."""


# ---------------------------------------------------------------------------
# Transformers-based backend  (primary)
# ---------------------------------------------------------------------------
class TransformersBackend(LLMBackend):
    """Uses HuggingFace transformers with optional 4-bit / 8-bit quantization.

    Supports any chat model that works with ``AutoModelForCausalLM`` and the
    HF chat template format (Qwen3.5, Gemma-3, Llama-3, Mistral, …).
    """

    def __init__(self, cfg: LLMConfig):
        import torch
        import os
        from dotenv import load_dotenv
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load environment variables (.env) - search up one level if not found
        if not load_dotenv():
            parent_env = os.path.join(os.path.dirname(os.getcwd()), ".env")
            load_dotenv(parent_env)
            
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            log.warning("HF_TOKEN not found in environment. Access to gated models (e.g. Gemma) may fail.")

        self.cfg = cfg
        log.info("Loading model %s (quant=%s) …", cfg.model_id, cfg.quantization)
        start = time.time()

        # ---------- quantization & CPU Optimization ----------
        # Note: BitsAndBytes 4-bit is not natively supported on Windows CPU.
        # We manually use bfloat16 or float32 for model weights to save RAM/Time.
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "cpu", # Force CPU-only
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() or hasattr(torch, 'bfloat16') else torch.float32,
            "low_cpu_mem_usage": True
        }

        # ---------- tokenizer and model with offline-first local cache checking ----------
        try:
            log.info("Checking local cache purely offline before hitting HF...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_id, trust_remote_code=True, local_files_only=True, token=hf_token
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.padding_side != "left":
                self.tokenizer.padding_side = "left"
                
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id, local_files_only=True, token=hf_token, **load_kwargs
            )
            log.info("Transformers model acquired offline.")
        except Exception as offline_e:
            log.info(f"Local models not found ({offline_e}). Dynamically downloading from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_id, trust_remote_code=True, local_files_only=False, token=hf_token
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.padding_side != "left":
                self.tokenizer.padding_side = "left"
                
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id, local_files_only=False, token=hf_token, **load_kwargs
            )

        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if self.model.generation_config.eos_token_id is None:
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
            
        elapsed = time.time() - start
        log.info("Model loaded in %.1f s", elapsed)


    # ---- public API ----
    def generate(self, messages: list[dict], **kwargs) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            try:
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", self.cfg.max_new_tokens),
                    temperature=kwargs.get("temperature", self.cfg.temperature),
                    do_sample=kwargs.get("temperature", self.cfg.temperature) > 0,
                    top_p=kwargs.get("top_p", 0.9),
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            except RuntimeError as e:
                if "probability tensor contains either `inf`, `nan` or element < 0" in str(e):
                    log.error("Invalid probabilities encountered during generation. Check model logits.")
                    raise ValueError("Model generated invalid probabilities. Adjust temperature or check logits.") from e
                else:
                    raise

        # Decode only the NEW tokens
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_with_tools(
        self, messages: list[dict], tools: list[dict], **kwargs
    ) -> dict:
        """Generate and detect tool calls in the output.

        We inject the tool schemas into the system prompt and parse the
        model's structured output.  Works with any model that can follow
        instructions — no special fine-tune needed.
        """
        tool_description = self._format_tools_for_prompt(tools)
        augmented = self._inject_tool_prompt(messages, tool_description)
        raw = self.generate(augmented, **kwargs)
        return self._parse_tool_response(raw)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # ---- internal helpers ----

    @staticmethod
    def _format_tools_for_prompt(tools: list[dict]) -> str:
        lines = ["You have access to the following tools:\n"]
        for t in tools:
            lines.append(f"### {t['name']}")
            lines.append(f"Description: {t.get('description', 'No description.')}")
            if "parameters" in t:
                lines.append(f"Parameters: {json.dumps(t['parameters'], indent=2)}")
            lines.append("")
        lines.append(
            "To call a tool, output EXACTLY this JSON block (and nothing else after it):\n"
            '```tool_call\n{"name": "<tool_name>", "arguments": {<args>}}\n```'
        )
        return "\n".join(lines)

    @staticmethod
    def _inject_tool_prompt(messages: list[dict], tool_desc: str) -> list[dict]:
        """Prepend tool descriptions to the system message."""
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {
                "role": "system",
                "content": msgs[0]["content"] + "\n\n" + tool_desc,
            }
        else:
            msgs.insert(0, {"role": "system", "content": tool_desc})
        return msgs

    @staticmethod
    def _parse_tool_response(raw: str) -> dict:
        """Extract tool_call JSON from the model output, if present."""
        # Look for ```tool_call ... ``` blocks
        pattern = r"```tool_call\s*\n?(.*?)\n?```"
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                call = json.loads(match.group(1).strip())
                return {
                    "content": raw[: match.start()].strip() or None,
                    "tool_calls": [
                        {
                            "name": call["name"],
                            "arguments": call.get("arguments", {}),
                        }
                    ],
                }
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: look for raw JSON object with "name" & "arguments"
        json_pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:'
        match2 = re.search(json_pattern, raw)
        if match2:
            # Try to extract full JSON object
            brace_count = 0
            start_idx = match2.start()
            for i in range(start_idx, len(raw)):
                if raw[i] == "{":
                    brace_count += 1
                elif raw[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            call = json.loads(raw[start_idx : i + 1])
                            return {
                                "content": raw[:start_idx].strip() or None,
                                "tool_calls": [
                                    {
                                        "name": call["name"],
                                        "arguments": call.get("arguments", {}),
                                    }
                                ],
                            }
                        except json.JSONDecodeError:
                            break

        return {"content": raw, "tool_calls": None}


# ---------------------------------------------------------------------------
# LlamaCpp Backend
# ---------------------------------------------------------------------------
class LlamaCppBackend(LLMBackend):
    """Uses llama-cpp-python to load GGUF models directly into RAM.
    Supports dynamic loading and unloading to preserve memory.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.model = None
        self.current_model_path = None

    def load_model(self, model_path: str):
        import gc
        import time
        from llama_cpp import Llama
        from deepagent.config import resolve_model_path

        # Resolve model path (download from HF if needed)
        resolved_path = resolve_model_path(model_path)

        if self.model is not None and self.current_model_path == resolved_path:
            return  # Already loaded

        if self.model is not None:
            self.unload_model()

        log.info("Loading LlamaCpp model from %s ...", resolved_path)
        start = time.time()
        self.model = Llama(
            model_path=resolved_path,
            n_ctx=self.cfg.llama_cpp.n_ctx,
            n_threads=self.cfg.llama_cpp.n_threads,
            n_gpu_layers=self.cfg.llama_cpp.n_gpu_layers,
            verbose=False,
        )
        self.current_model_path = resolved_path
        elapsed = time.time() - start
        log.info("Model loaded in %.1f s", elapsed)

    def unload_model(self):
        import gc
        if self.model is not None:
            log.info("Unloading model %s to free RAM ...", self.current_model_path)
            del self.model
            self.model = None
            self.current_model_path = None
            gc.collect()

    def generate(self, messages: list[dict], **kwargs) -> str:
        if self.model is None:
            self.load_model(self.cfg.llama_cpp.model_path)

        response_format = kwargs.pop("response_format", None)

        out = self.model.create_chat_completion(
            messages=messages,
            max_tokens=kwargs.get("max_new_tokens", self.cfg.max_new_tokens),
            temperature=kwargs.get("temperature", self.cfg.temperature),
            top_p=kwargs.get("top_p", 0.9),
            response_format=response_format,
        )
        return out["choices"][0]["message"]["content"].strip()

    def generate_with_tools(
        self, messages: list[dict], tools: list[dict], **kwargs
    ) -> dict:
        tool_description = TransformersBackend._format_tools_for_prompt(tools)
        augmented = TransformersBackend._inject_tool_prompt(messages, tool_description)
        raw = self.generate(augmented, **kwargs)
        return TransformersBackend._parse_tool_response(raw)

    def count_tokens(self, text: str) -> int:
        if self.model is None:
            return len(text) // 4  # rough estimate if unloaded
        return len(self.model.tokenize(text.encode("utf-8")))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_backend(cfg: Optional[LLMConfig] = None) -> LLMBackend:
    """Instantiate the appropriate backend from config."""
    if cfg is None:
        cfg = LLMConfig()

    if hasattr(cfg, 'backend_type') and cfg.backend_type == "llama_cpp":
        return LlamaCppBackend(cfg)

    return TransformersBackend(cfg)
