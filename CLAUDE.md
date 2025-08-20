# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trae Agent is an LLM-based agent for general purpose software engineering tasks. It provides a CLI interface that understands natural language instructions and executes complex software engineering workflows using various tools and LLM providers. The project is designed to be research-friendly with a transparent, modular architecture.

## Common Development Commands

### Setup and Installation
```bash
# Create virtual environment and install dependencies
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate
```

### Testing
```bash
# Run all tests (skips external service tests)
make test
# or
uv run pytest

# Run tests with UV (recommended)
make uv-test

# Run specific test file
uv run pytest tests/agent/test_trae_agent.py -v

# Test long context components
uv run pytest tests/agent/test_long_context_agent.py -v
uv run pytest tests/tools/test_context_tool.py -v
uv run pytest tests/tools/test_meta_tool.py -v
uv run pytest tests/tools/test_chunk_tool.py -v
```

### Code Quality
```bash
# Run pre-commit hooks
make pre-commit

# Fix formatting issues
make fix-format
# or
ruff format .
ruff check --fix .

# Run linting only
ruff check .
```

### Running the Agent
```bash
# Basic task execution
trae-cli run "Create a hello world Python script"

# Interactive mode
trae-cli interactive

# Show current configuration
trae-cli show-config

# List available tools
trae-cli tools

# Long context processing (via Python API)
python examples/long_context_example.py
```

## Architecture Overview

### Core Components

1. **Agent Layer** (`trae_agent/agent/`)
   - `trae_agent.py`: Main agent implementation specialized for software engineering
   - `long_context_agent.py`: Specialized agent for processing extremely long text contexts via chunking and problem decomposition
   - `base_agent.py`: Abstract base agent with common functionality
   - `agent_basics.py`: Core agent execution primitives and error handling

2. **Tool System** (`trae_agent/tools/`)
   - Modular tool architecture with registry-based discovery
   - Built-in tools: bash execution, file editing, sequential thinking, JSON editing, task completion
   - Long context tools: `context_tool.py` (file management), `meta_tool.py` (prompt enhancement), `chunk_tool.py` (text processing)
   - MCP (Model Context Protocol) integration for external tools
   - Each tool inherits from `base.py` and implements standard interface

3. **LLM Client Layer** (`trae_agent/utils/llm_clients/`)
   - Multi-provider support: OpenAI, Anthropic, Google, Azure, Ollama, OpenRouter, Doubao
   - Unified interface through `base_client.py` with provider-specific implementations
   - Retry logic and error handling built-in

4. **Configuration System** (`trae_agent/utils/config.py`)
   - YAML-based configuration with backward compatibility for JSON
   - Hierarchical config resolution: CLI args > environment > config file
   - Support for multiple model providers and agents in single config

5. **CLI Interface** (`trae_agent/cli.py`)
   - Click-based command interface
   - Rich console support for interactive mode
   - Trajectory recording for debugging and analysis

### Key Features

- **Lakeview**: Provides concise summarization for agent steps
- **Trajectory Recording**: Detailed logging in `trajectories/` directory
- **MCP Integration**: External tool support via Model Context Protocol
- **Multi-Console Support**: Simple text and rich interactive modes

## Configuration

### Primary Config (YAML)
Use `trae_config.yaml` (preferred) or fall back to `trae_config.json`. Configuration includes:
- Agent settings (max_steps, tools, lakeview)
- Model providers and their credentials
- MCP server configurations

### Environment Variables
API keys can be set via environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- etc.

## Tool Development

Tools are registered in `tools_registry` and must implement the `Tool` interface from `tools/base.py`. Each tool defines:
- Name and description
- Input/output schemas
- Execution logic
- Provider-specific behavior if needed

## Testing Strategy

- Unit tests in `tests/` mirror the source structure
- Integration tests with external services can be skipped via environment flags
- Pre-commit hooks ensure code quality
- Coverage reporting configured in `pyproject.toml`

## Important Implementation Details

- Agent execution is async-first throughout the codebase
- Tools maintain persistent state within agent sessions
- Git operations are built-in for patch generation and diff tracking
- Trajectory files are auto-generated with timestamps unless specified
- MCP clients require proper async cleanup to prevent context leaks

## LongContextAgent

### Overview
The LongContextAgent (`trae_agent/agent/long_context_agent.py`) is a specialized agent designed for processing extremely long text contexts that exceed standard LLM token limits. It focuses on **problem decomposition and Chain-of-Thought (CoT) simplification** rather than simple text splitting.

### Key Features
- **Strategic Problem Decomposition**: Breaks complex queries into simpler, independent sub-problems
- **CoT Length Optimization**: Reduces reasoning complexity per chunk to improve accuracy
- **Semantic Chunking**: Respects document structure and maintains context continuity
- **Meta-Prompt Enhancement**: Automatically optimizes prompts for chunk-based processing
- **Result Integration**: Synthesizes partial results into comprehensive answers

### Core Tools
1. **ContextFileManagerTool** (`trae_agent/tools/context_tool.py`)
   - Saves long context to temporary files
   - Manages chunking strategies and boundaries
   - Provides chunk analysis and planning capabilities

2. **MetaPromptTool** (`trae_agent/tools/meta_tool.py`)
   - Transforms simple prompts into detailed, optimized instructions
   - Uses LLM to enhance prompt quality for chunk processing
   - Focuses on reducing CoT complexity per sub-task

3. **TextChunkTool** (`trae_agent/tools/chunk_tool.py`)
   - Processes specific text segments with enhanced prompts
   - Maintains awareness of partial context limitations
   - Formats results for easy integration

### Usage Example
```python
from trae_agent.agent.long_context_agent import LongContextAgent
from trae_agent.utils.config import AgentConfig

# Create agent with config
agent = LongContextAgent(agent_config)

# Setup task with long context
task = "Analyze the document and answer the query"
extra_args = {
    "context": very_long_text,  # KB to MB of text
    "query": "What are the key findings?"
}

# Execute
agent.new_task(task, extra_args)
execution = await agent.execute_task()
```

### Workflow
1. Save context to temporary file (ContextFileManagerTool)
2. Analyze query and plan decomposition strategy (Sequential Thinking)
3. Generate optimized prompts for chunk processing (MetaPromptTool)
4. Process chunks systematically (TextChunkTool)
5. Integrate results and validate completeness

### Testing
- Run `python examples/long_context_example.py` for a complete demo
- Unit tests available for all components in `tests/` directory
- Supports evaluation with parquet datasets via `evaluation/long_context_eval.py`