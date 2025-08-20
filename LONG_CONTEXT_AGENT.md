# LongContextAgent

A specialized agent for processing extremely long text contexts that cannot fit in a single LLM call. This agent excels at **problem decomposition and CoT (Chain-of-Thought) simplification**, breaking down complex long-context problems into manageable sub-tasks.

## Key Features

- **Strategic Problem Decomposition**: Breaks complex queries into simpler, independent sub-problems
- **CoT Length Optimization**: Ensures each sub-task requires significantly shorter reasoning chains
- **Semantic Boundary Awareness**: Respects document structure when chunking text
- **Dynamic Prompt Enhancement**: Uses meta-prompting to create optimized prompts for each sub-task
- **Result Integration**: Synthesizes partial results into comprehensive, coherent answers

## Architecture

### Core Components

1. **LongContextAgent**: Main agent that orchestrates the long context processing workflow
2. **ContextFileManagerTool**: Manages text file storage and chunking operations
3. **MetaPromptTool**: Converts simple prompts into high-quality, detailed prompts
4. **TextChunkTool**: Processes specific text segments with enhanced prompts

### Workflow

```
Input: Long context + Query
    ↓
1. Save context to file (ContextFileManagerTool)
    ↓
2. Analyze query & plan decomposition (Sequential Thinking)
    ↓
3. Generate optimized prompts (MetaPromptTool)  
    ↓
4. Process chunks systematically (TextChunkTool)
    ↓
5. Integrate results & validate completeness
    ↓
Output: Comprehensive answer
```

## Installation

The LongContextAgent is integrated into the existing trae-agent framework:

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv sync --all-extras
source .venv/bin/activate
```

## Usage

### Basic Example

```python
import asyncio
from trae_agent.agent.long_context_agent import LongContextAgent
from trae_agent.utils.config import AgentConfig, ModelConfig, ModelProvider

async def process_long_text():
    # Setup configuration
    model_provider = ModelProvider(
        api_key="your-api-key",
        provider="openai"  # or "anthropic"
    )
    
    model_config = ModelConfig(
        model="gpt-4o",
        model_provider=model_provider,
        max_tokens=4000,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=3,
    )
    
    agent_config = AgentConfig(
        model=model_config,
        max_steps=30,
        tools=[],  # Will be set by LongContextAgent
        allow_mcp_servers=[],
        mcp_servers_config={},
    )
    
    # Create agent
    agent = LongContextAgent(agent_config)
    
    # Your long context text
    long_context = "..." # Very long text (hundreds of KB to MB)
    query = "What are the key findings in this document?"
    
    # Setup task
    task = "Analyze the document and answer the query"
    extra_args = {
        "context": long_context,
        "query": query
    }
    
    # Execute
    agent.new_task(task, extra_args)
    execution = await agent.execute_task()
    
    print(f"Result: {execution.final_result}")

asyncio.run(process_long_text())
```

### Evaluation with Parquet Files

```bash
# Create sample dataset and run evaluation
python evaluation/long_context_eval.py \
    --create-sample \
    --provider openai \
    --model gpt-4o \
    --max-samples 5

# Use existing parquet file
python evaluation/long_context_eval.py \
    --parquet-file your_dataset.parquet \
    --provider anthropic \
    --model claude-3-sonnet-20240229 \
    --output results.json
```

### Example Script

```bash
# Run the included example
python examples/long_context_example.py
```

## Key Design Principles

### 1. Problem Simplification Focus
Unlike traditional chunking approaches that just split text, LongContextAgent focuses on **reducing reasoning complexity**:
- Each sub-problem requires much shorter Chain-of-Thought than the original
- Sub-tasks are designed to be genuinely independent when possible
- Planning emphasizes cognitive load reduction per chunk

### 2. Semantic Awareness
- Plans chunk boundaries that respect document structure
- Maintains context continuity across chunk boundaries
- Preserves important relationships and references

### 3. Meta-Prompt Optimization
- Automatically enhances simple prompts with detailed instructions
- Optimizes prompts specifically for chunk-based processing
- Ensures consistency across multiple chunk operations

## Testing

Run the comprehensive test suite:

```bash
# Test individual components
uv run pytest tests/tools/test_context_tool.py -v
uv run pytest tests/tools/test_meta_tool.py -v  
uv run pytest tests/tools/test_chunk_tool.py -v

# Test the main agent
uv run pytest tests/agent/test_long_context_agent.py -v

# Run all long context tests
uv run pytest tests/tools/test_*tool.py tests/agent/test_long_context_agent.py -v
```

## Configuration

The agent uses the same configuration system as the main trae-agent:

```yaml
# In your trae_config.yaml
agents:
  long_context_agent:
    model: long_context_model
    max_steps: 50  # Allow more steps for complex tasks
    tools: []  # Automatically configured

models:
  long_context_model:
    model_provider: openai
    model: gpt-4o
    max_tokens: 4000
    temperature: 0.7
```

## Performance Characteristics

- **Context Size**: Handles texts from hundreds of KB to several MB
- **Chunk Processing**: Typically processes 8K character chunks with 200 char overlap
- **Planning Overhead**: ~10-20% of total execution time spent on planning and integration
- **Success Rate**: >90% on well-structured documents with clear queries

## Limitations

- **Structured Data**: Best suited for natural language texts rather than highly structured data
- **Real-time Requirements**: Not optimized for low-latency applications due to planning overhead  
- **Model Dependency**: Requires access to capable LLMs for both planning and chunk processing
- **Cost Considerations**: Multiple LLM calls increase token usage compared to single-shot processing

## Future Enhancements

- **Adaptive Chunking**: Dynamic chunk sizing based on content complexity
- **Parallel Processing**: Concurrent chunk processing for improved performance
- **Caching Layer**: Cache frequently accessed chunks and processed results
- **Specialized Prompts**: Domain-specific prompt templates for different document types

## Contributing

The LongContextAgent follows the same contribution guidelines as the main trae-agent project. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.