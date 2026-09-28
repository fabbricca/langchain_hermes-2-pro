# LangChain function calling for Nous Hermes 2 Pro

Adapters that let [Nous Research Hermes 2 Pro](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B),
run locally through Ollama, act as a tool-calling agent in [LangChain](https://github.com/langchain-ai/langchain).

Written in May 2024, when Hermes 2 Pro was one of the first open-weight models trained for
function calling. At the time, LangChain's agents expected OpenAI's `function_call` message field.
Hermes 2 Pro uses its own format instead: tool schemas go in the system prompt inside `<tools>` tags,
and the model answers with JSON inside `<tool_call>` tags. This repository bridges the two formats.

## How it works

One agent turn:

```
prompt (tools in <tools>) ──► Hermes 2 Pro ──► "<tool_call>{...}</tool_call>"
        ▲                                               │
        │                                   output parser: extract, repair, validate
        │                                               │
   scratchpad: previous calls + <tool_response>  ◄── AgentExecutor runs the tool
```

| component | file | role |
|---|---|---|
| Prompt | `prompts/template.yaml`, `prompts/prompt.py` | Hermes 2 Pro's system prompt, with the tool schemas and the `FunctionCall` JSON schema |
| Agent | `agents/nous_hermes_functions_agent/base.py` | `create_nous_hermes_functions_agent`: scratchpad → prompt → LLM → parser, as a LangChain `Runnable` |
| Output parser | `agents/output_parsers/` | pulls every `<tool_call>` block out of the reply and turns it into `AgentAction`s (several per turn are supported). A reply without a tool call becomes `AgentFinish` |
| Repair and validation | `agents/output_parsers/utils.py` | tolerates the small model's JSON mistakes (Python literals, single quotes, trailing text). Unknown tools and wrong argument sets are sent to an error tool instead of raising an exception |
| Scratchpad | `agents/format_scratchpad/` | returns each tool result to the model inside `<tool_response>` tags, the format it was trained on |
| Tools | `tools/tools.py` | web search (DuckDuckGo), file writing, a toy word-length tool, and `handle_tools_error` |

The error tool is the main design choice. A 7B model often calls a tool with missing or extra
arguments. Instead of failing the run, the parser redirects the call to `handle_tools_error`,
which tells the model the expected and received arguments, so it can correct itself on the next turn.

## Running it

```bash
ollama pull adrienbrault/nous-hermes2pro:Q5_K_S
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py          # run from the repository root; type /bye to exit
```

## Limitations

- The JSON repair uses heuristics tuned on the failures observed in 2024, not a grammar-constrained decoder.
- It targets the LangChain agent API of mid-2024 (`AgentExecutor`). Newer LangChain versions and
  model runtimes support native tool calling for Hermes-format models, which makes most of this
  unnecessary today.

## License

MIT. See [LICENSE](LICENSE).
