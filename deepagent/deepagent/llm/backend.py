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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        log.info("Loading model %s (quant=%s) …", cfg["model_id"], cfg["quantization"])
        start = time.time()

        # ---------- tokenizer ----------
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"], trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"

        # ---------- quantization config ----------
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": cfg["device"] if cfg["device"] != "auto" else "cpu",
        }

        if cfg["quantization"] == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif cfg["quantization"] == "8bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["dtype"] = torch.bfloat16

        # ---------- model ----------
        self.model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], **load_kwargs)
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
                    max_new_tokens=kwargs.get("max_new_tokens", self.cfg["max_new_tokens"]),
                    temperature=kwargs.get("temperature", self.cfg["temperature"]),
                    do_sample=kwargs.get("temperature", self.cfg["temperature"]) > 0,
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
# Factory
# ---------------------------------------------------------------------------
def create_backend(cfg: Optional[LLMConfig] = None) -> LLMBackend:
    """Instantiate the appropriate backend from config."""
    if cfg is None:
        cfg = LLMConfig()
    return TransformersBackend(cfg)
