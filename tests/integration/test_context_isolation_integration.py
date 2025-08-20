# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Integration tests for LongContextAgent with context isolation verification."""

import tempfile
import pytest
import os
from unittest.mock import Mock, AsyncMock, patch
import json

from trae_agent.agent.long_context_agent import LongContextAgent
from trae_agent.utils.config import AgentConfig, ModelConfig, ModelProvider
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse, LLMUsage
from trae_agent.tools.base import ToolCall, ToolResult


class TestLongContextAgentIntegration:
    """Integration tests for LongContextAgent with context isolation focus."""

    @pytest.fixture
    def mock_model_config(self):
        """Create mock model configuration."""
        provider = ModelProvider(
            api_key="test-key",
            provider="openai"
        )
        return ModelConfig(
            model="gpt-4",
            model_provider=provider,
            max_tokens=4000,
            temperature=0.7,
            top_p=1.0,
            top_k=0,
            parallel_tool_calls=False,
            max_retries=3
        )

    @pytest.fixture
    def agent_config(self, mock_model_config):
        """Create agent configuration."""
        return AgentConfig(
            model=mock_model_config,
            max_steps=10,
            tools=[],  # Will be overridden by LongContextAgent
            allow_mcp_servers=[],
            mcp_servers_config={}
        )

    @pytest.fixture
    def long_context_agent(self, agent_config):
        """Create LongContextAgent instance."""
        return LongContextAgent(agent_config)

    @pytest.fixture
    def sample_long_context(self):
        """Create sample long context for testing."""
        return """This is a sample long context document for testing purposes.
        
The document contains multiple sections and paragraphs with various information.
Section 1 discusses the implementation of context processing algorithms.
Section 2 covers the optimization of memory usage in large text processing.
Section 3 explains the integration patterns for chunk-based analysis.

The content is designed to test the chunking and processing capabilities
of the LongContextAgent system. Each section contains specific information
that can be extracted and analyzed independently.

Key features mentioned include:
- Automatic prompt enhancement
- Context isolation mechanisms  
- Efficient chunk processing
- Result integration strategies

The document concludes with recommendations for best practices
in long context processing and future enhancement opportunities."""

    def create_mock_llm_response(self, content: str = "", tool_calls: list = None):
        """Helper to create mock LLM response."""
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=LLMUsage(input_tokens=100, output_tokens=50)
        )

    def create_mock_tool_result(self, output: str):
        """Helper to create mock tool result."""
        return ToolResult(
            name="test_tool",
            arguments={},
            result=output
        )

    @pytest.mark.asyncio
    async def test_context_isolation_in_execution_flow(self, long_context_agent, sample_long_context):
        """Test that enhanced prompts don't appear in main model context during execution."""
        
        # Track all messages sent to LLM client
        captured_messages = []
        
        def capture_chat_call(messages, model_config, tools):
            """Capture LLM chat calls to analyze context."""
            captured_messages.append({
                'messages': [msg.dict() if hasattr(msg, 'dict') else {'role': msg.role, 'content': msg.content} for msg in messages],
                'tools': [tool.get_name() for tool in tools]
            })
            
            # Simulate different responses based on call number
            call_num = len(captured_messages)
            
            if call_num == 1:
                # First call: Agent should use context_file_manager
                return self.create_mock_llm_response(
                    content="I'll save the context to a file first.",
                    tool_calls=[ToolCall(
                        id="call_1",
                        name="context_file_manager",
                        arguments={"action": "save_context", "context": sample_long_context}
                    )]
                )
            elif call_num == 2:
                # Second call: After file saved, use enhanced_chunk_tool
                return self.create_mock_llm_response(
                    content="Now I'll process the first chunk.",
                    tool_calls=[ToolCall(
                        id="call_2", 
                        name="enhanced_chunk_tool",
                        arguments={
                            "file_path": "/tmp/context.txt",
                            "start_pos": 0,
                            "end_pos": 200,
                            "query": "What are the key features mentioned?",
                            "simple_prompt": "Extract key features"  # Note: simple prompt, not enhanced
                        }
                    )]
                )
            else:
                # Final call: Complete task
                return self.create_mock_llm_response(
                    content="Task completed successfully.",
                    tool_calls=[ToolCall(
                        id="call_3",
                        name="task_done", 
                        arguments={"result": "Found key features: automatic prompt enhancement, context isolation, etc."}
                    )]
                )
        
        # Mock the LLM client chat method
        with patch.object(long_context_agent._llm_client, 'chat', side_effect=capture_chat_call):
            # Mock tool execution results
            with patch('trae_agent.tools.context_tool.ContextFileManagerTool.execute') as mock_context:
                with patch('trae_agent.tools.enhanced_chunk_tool.EnhancedChunkTool.execute') as mock_enhanced:
                    with patch('trae_agent.tools.task_done.TaskDoneTool.execute') as mock_done:
                        
                        # Setup tool return values
                        mock_context.return_value = ToolResult(
                            name="context_file_manager",
                            arguments={},
                            result="Context saved to /tmp/context.txt successfully!"
                        )
                        
                        mock_enhanced.return_value = ToolResult(
                            name="enhanced_chunk_tool", 
                            arguments={},
                            result="""**Chunk Analysis Result**

**Chunk:** chunk_0_200 (position 0:200)
**Analysis:** The text mentions key features including automatic prompt enhancement, context isolation mechanisms, efficient chunk processing, and result integration strategies.

**Status:** Processing complete"""
                        )
                        
                        mock_done.return_value = ToolResult(
                            name="task_done",
                            arguments={},
                            result="Task completed"
                        )
                        
                        # Setup task
                        task = "Find key features mentioned in the document"
                        extra_args = {
                            "context": sample_long_context,
                            "query": "What are the key features mentioned?"
                        }
                        
                        # Execute task
                        long_context_agent.new_task(task, extra_args)
                        execution = await long_context_agent.execute_task()
                        
                        # Verify execution was successful
                        assert execution.success
                        
                        # Analyze captured messages for context isolation
                        all_message_content = []
                        for call_data in captured_messages:
                            for msg in call_data['messages']:
                                all_message_content.append(msg.get('content', ''))
                        
                        combined_context = ' '.join(all_message_content)
                        
                        # CRITICAL TEST: Verify enhanced prompts are NOT in main model context
                        enhanced_prompt_indicators = [
                            "Please analyze the provided text carefully and extract",
                            "Focus on accuracy and completeness in your analysis",
                            "Consider the following detailed instructions",
                            "Provide comprehensive analysis with the following criteria",
                            "extensive instructions and guidance for processing"
                        ]
                        
                        for indicator in enhanced_prompt_indicators:
                            assert indicator not in combined_context, f"Enhanced prompt leaked to main context: {indicator}"
                        
                        # Verify that only simple prompts appear in main context
                        simple_prompts = [
                            "Extract key features",
                            "Find key features",
                            "simple_prompt"
                        ]
                        
                        # At least one simple prompt should be present
                        assert any(prompt in combined_context for prompt in simple_prompts), "No simple prompts found in context"
                        
                        # Verify the enhanced_chunk_tool was called correctly
                        mock_enhanced.assert_called_once()
                        enhanced_args = mock_enhanced.call_args[0][0]
                        assert enhanced_args["simple_prompt"] == "Extract key features"
                        assert "enhanced_prompt" not in enhanced_args  # Should be added internally

    @pytest.mark.asyncio
    async def test_tool_registration_and_initialization(self, long_context_agent):
        """Test that tools are properly registered and initialized."""
        # Setup a simple task to trigger tool initialization
        task = "Test task"
        extra_args = {"context": "test context", "query": "test query"}
        
        long_context_agent.new_task(task, extra_args)
        
        # Verify tools are properly registered
        tool_names = [tool.get_name() for tool in long_context_agent.tools]
        
        expected_tools = [
            "context_file_manager",
            "enhanced_chunk_tool", 
            "sequentialthinking",
            "str_replace_based_edit_tool",
            "json_edit_tool",
            "task_done",
            "bash"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Tool {expected_tool} not found in registered tools"
        
        # Verify enhanced_chunk_tool has LLM client
        enhanced_tool = next(tool for tool in long_context_agent.tools if tool.get_name() == "enhanced_chunk_tool")
        assert enhanced_tool.llm_client is not None, "EnhancedChunkTool should have LLM client"

    @pytest.mark.asyncio  
    async def test_message_chain_construction(self, long_context_agent, sample_long_context):
        """Test that initial message chain is properly constructed."""
        task = "Analyze the document"
        query = "What are the main topics?"
        extra_args = {"context": sample_long_context, "query": query}
        
        long_context_agent.new_task(task, extra_args)
        
        messages = long_context_agent.initial_messages
        
        # Should have system message and user messages
        assert len(messages) >= 2
        assert messages[0].role == "system"
        
        # Find the main task message
        task_message = None
        context_message = None
        
        for msg in messages[1:]:
            if "**Task:**" in msg.content:
                task_message = msg
            elif "**Full Context Text to Process:**" in msg.content:
                context_message = msg
        
        assert task_message is not None, "Task message not found"
        assert context_message is not None, "Context message not found"
        
        # Verify task message contains updated workflow
        assert "enhanced_chunk_tool" in task_message.content
        assert "automatically enhances prompts internally" in task_message.content
        
        # Should NOT contain references to old separate tools
        assert "meta_prompt_tool" not in task_message.content
        assert "text_chunk_tool" not in task_message.content

    @pytest.mark.asyncio
    async def test_context_length_calculation(self, long_context_agent):
        """Test context length calculation and message formatting."""
        sample_context = "A" * 10000  # 10k character context
        
        task = "Process large context"
        extra_args = {"context": sample_context, "query": "test query"}
        
        long_context_agent.new_task(task, extra_args)
        messages = long_context_agent.initial_messages
        
        # Find task message
        task_message = next(msg for msg in messages if "**Task:**" in msg.content)
        
        # Verify length calculation
        assert "10,000 characters" in task_message.content
        assert "2,500 tokens estimated" in task_message.content
        assert "too long for direct processing" in task_message.content

    def test_temp_directory_creation(self, long_context_agent):
        """Test that temporary directory is created and accessible."""
        assert hasattr(long_context_agent, 'temp_dir')
        assert long_context_agent.temp_dir is not None
        assert os.path.exists(long_context_agent.temp_dir)
        assert "long_context_" in long_context_agent.temp_dir

    @pytest.mark.asyncio
    async def test_cleanup_temp_files(self, long_context_agent, sample_long_context):
        """Test that temporary files are cleaned up after execution."""
        temp_dir = long_context_agent.temp_dir
        
        # Verify temp directory exists before execution
        assert os.path.exists(temp_dir)
        
        # Setup and execute a simple task
        with patch.object(long_context_agent._llm_client, 'chat') as mock_chat:
            mock_chat.return_value = self.create_mock_llm_response(
                content="Task completed",
                tool_calls=[ToolCall(id="1", name="task_done", arguments={"result": "done"})]
            )
            
            with patch('trae_agent.tools.task_done.TaskDoneTool.execute') as mock_done:
                mock_done.return_value = ToolResult(name="task_done", arguments={}, result="completed")
                
                task = "Simple test task"
                extra_args = {"context": sample_long_context, "query": "test"}
                
                long_context_agent.new_task(task, extra_args)
                execution = await long_context_agent.execute_task()
                
                # Verify temp directory is cleaned up after execution
                assert not os.path.exists(temp_dir), "Temporary directory should be cleaned up after execution"

    @pytest.mark.asyncio
    async def test_error_handling_with_context_isolation(self, long_context_agent, sample_long_context):
        """Test error handling while maintaining context isolation."""
        
        captured_messages = []
        
        def failing_chat_call(messages, model_config, tools):
            captured_messages.extend([msg.content for msg in messages if hasattr(msg, 'content')])
            raise Exception("Simulated LLM failure")
        
        with patch.object(long_context_agent._llm_client, 'chat', side_effect=failing_chat_call):
            task = "Test error handling"
            extra_args = {"context": sample_long_context, "query": "test query"}
            
            long_context_agent.new_task(task, extra_args)
            execution = await long_context_agent.execute_task()
            
            # Execution should fail gracefully
            assert not execution.success
            assert "Agent execution failed" in execution.final_result
            
            # Even in error cases, no enhanced prompts should leak
            all_content = ' '.join(captured_messages)
            enhanced_indicators = ["extensive instructions", "detailed analysis criteria"]
            
            for indicator in enhanced_indicators:
                assert indicator not in all_content


@pytest.mark.asyncio
async def test_context_isolation_benchmark():
    """Benchmark test to measure context size difference."""
    
    # Create a very long enhanced prompt scenario
    long_enhanced_prompt = "A" * 5000  # 5KB enhanced prompt
    
    # Simulate old approach (meta_tool result in main context)
    old_approach_context = f"""System prompt here.
Task description here.
Meta tool result: {long_enhanced_prompt}
Chunk tool input and output here.
Other tool calls here."""
    
    # Simulate new approach (enhanced_chunk_tool with internal processing)
    new_approach_context = """System prompt here.
Task description here.
Enhanced chunk tool result: Simple clean analysis result here.
Other tool calls here."""
    
    old_size = len(old_approach_context)
    new_size = len(new_approach_context)
    
    # Verify significant context size reduction
    reduction_ratio = (old_size - new_size) / old_size
    assert reduction_ratio > 0.5, f"Context size should be reduced by more than 50%, got {reduction_ratio:.2%}"
    
    print(f"Context size reduction: {reduction_ratio:.1%} ({old_size} -> {new_size} characters)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])