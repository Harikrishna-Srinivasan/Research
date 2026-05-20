# DeepAgent

CPU friendly Autonomous research agent — uses HuggingFace Open-Source model with MCP tools.
Plans tasks, executes them, reviews its own work, creates new tools on the fly.

## Install

```bash
cd research
cd deepagent
pip install .
```

## Usage

```bash
# Run a task
deepagent run "Build a React todo app with FastAPI backend"

# Interactive chat mode
deepagent interactive

# List available tools
deepagent tools

# Run as MCP server (for external agents)
deepagent mcp-server
```

## Config

Edit `config.yaml` to change model, tools, or safety settings:

```yaml
llm:
  model_id: "google/gemma-3-4b-it" # swap any HF model here
  quantization: "none"             # "4bit", "8bit", or "none"
```

## Architecture

```
Goal → Planner → [sub-tasks] → Executor (ReAct loop + tools) → Critic → Result
                                     ↕
                              Tool Registry
                    (file, shell, web, code, arxiv, HF,
                     + dynamically created tools)
```

## Built-in Tools

| Tool | What it does |
|------|-------------|
| `read_file`, `write_file`, `find_files` | File operations |
| `run_command`, `install_package` | Shell (sandboxed) |
| `web_search`, `read_webpage` | DuckDuckGo + page reader |
| `execute_python` | Run Python code |
| `search_arxiv`, `search_huggingface` | Research APIs |
| `search_memory`, `store_memory`, `list_memories` | **Semantic memory search** (CPU-efficient, no API key) |
| `create_tool` | **Agent creates new tools at runtime** |

## Examples

```bash
deepagent run "Find latest MoE papers on ArXiv and summarize key innovations"
deepagent run "Create a social media app using Next.js, deploy to Vercel"
deepagent run "Analyze Mersenne prime patterns and test Lucas-Lehmer optimizations"
```
