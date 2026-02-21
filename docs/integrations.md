# FormalAnswer Integrations

FormalAnswer is designed to be used not just as a CLI tool, but as a robust **Reasoning Backend** for other AI agents and platforms. It provides three primary integration interfaces.

## 1. Model Context Protocol (MCP)
**Best for:** Claude Desktop, IDEs (Cursor, VS Code), and MCP-compliant agents.

FormalAnswer implements a fully compliant MCP server (`src/mcp_server.py`) that exposes its verification capabilities as a tool.

### Configuration (Claude Desktop)
To use FormalAnswer with Claude Desktop on macOS:

1.  Locate your Claude config file: `~/Library/Application Support/Claude/claude_desktop_config.json`
2.  Add the FormalAnswer server configuration:

```json
{
  "mcpServers": {
    "formal-answer": {
      "command": "uv",
      "args": [
        "run",
        "/absolute/path/to/formalanswer/src/mcp_server.py"
      ],
      "env": {
        "GOOGLE_API_KEY": "your_actual_api_key_here"
      }
    }
  }
}
```
*(Note: Ensure you provide the absolute path to the project and that you have `uv` or `python` configured correctly).*

### Available Tools
*   `formal_verify(query: str, model: str = "gemini-2.5-flash")`: Accepts a natural language query (e.g., "Design a deadlock-free traffic light"), runs the full FormalAnswer loop (Lean/TLA+/Python), and returns the verified result.

## 2. Gemini Function Calling
**Best for:** Google Vertex AI Agents, Google AI Studio, custom Gemini-based apps.

You can equip your Gemini-based agents with FormalAnswer as a custom tool.

### Tool Definition
Use the schema defined in `integrations/gemini_tool.json`:

```json
{
  "function_declarations": [
    {
      "name": "formal_verify",
      "description": "Verifies complex logic, distributed systems, or critical safety invariants using Formal Methods (Lean 4, TLA+, JAX). Use this tool when the user asks for high-assurance designs, safety proofs, or risk assessments.",
      "parameters": {
        "type": "OBJECT",
        "properties": {
          "query": {
            "type": "STRING",
            "description": "The natural language description of the system, logic, or theorem to verify."
          },
          "model": {
            "type": "STRING",
            "description": "The underlying LLM to use for reasoning (default: gemini-2.5-flash).",
            "nullable": true
          }
        },
        "required": ["query"]
      }
    }
  ]
}
```

### Implementation
Your backend should handle the function call `formal_verify` by invoking the FormalAnswer API or CLI:
```python
# Pseudo-code for tool handler
if tool_call.name == "formal_verify":
    result = subprocess.run(["./query.sh", tool_call.args["query"]])
    return result.stdout
```

## 3. OpenAI Function Calling
**Best for:** OpenAI Assistants, ChatGPT Plugins, LangChain.

FormalAnswer is compatible with the OpenAI Tool format.

### Tool Definition
Use the schema defined in `integrations/openai_tool.json`:

```json
{
  "type": "function",
  "function": {
    "name": "formal_verify",
    "description": "Verifies complex logic, distributed systems, or critical safety invariants using Formal Methods (Lean 4, TLA+, JAX).",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The natural language description of the system, logic, or theorem to verify."
        },
        "model": {
          "type": "string",
          "description": "The underlying LLM to use for reasoning (default: gemini-2.5-flash)."
        }
      },
      "required": ["query"]
    }
  }
}

## 4. TypeScript/Node.js Integration (Implementation Linking)
**Best for:** CI/CD pipelines, automated testing of JS/TS implementations against formal oracles.

FormalAnswer provides a set of TypeScript helpers in the `impl-link/` directory to help you link formal specs to actual code implementations. This allows you to use TLA+ or Lean as an **executable oracle** for your tests.

### Key Concepts
1.  **Formal Oracle:** A TLA+ spec computes expected results for a given input.
2.  **Implementation:** Your JS/TS code computes the same results.
3.  **Harness:** A Jest/Vitest test that runs both and asserts they match.

### Helper Library (`impl-link/index.ts`)
*   `findFormalWork(root: string)`: Locates the FormalAnswer toolchain and environment.
*   `runTlc(work, opts)`: Runs the TLC model checker on a spec.
*   `extractPrintedTlaStringSet(stdout, key)`: Extracts TLA+ sets from TLC output for easy diffing.
*   `runLean(work, opts)`: Runs the Lean 4 compiler.

### Example Workflow
See [impl-link/README.md](../impl-link/README.md) for a detailed "HOWTO" on linking formal proofs to implementations.

## 5. REST API (Internal)
**Best for:** High-latency reasoning tasks, distributed agents.

FormalAnswer includes a lightweight Python API wrapper (`src/api.py`) that can be exposed via FastAPI.

### Endpoint: `POST /verify`
**Request Schema:**
```json
{
  "task": "Natural language query",
  "model": "gemini-2.5-flash"
}
```

**Response Schema:**
```json
{
  "success": true,
  "answer": "The verified reasoning...",
  "artifacts": {
    "proof_dir": "path/to/results"
  }
}
```
```
