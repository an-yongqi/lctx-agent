# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

from trae_agent.agent.long_context_agent import LongContextAgent
from trae_agent.utils.config import AgentConfig, ModelConfig, ModelProvider


class TestLongContextAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock model provider and config with dummy API key
        self.mock_provider = ModelProvider(
            api_key="test-dummy-api-key",
            provider="openai"
        )
        
        self.mock_model_config = ModelConfig(
            model="gpt-4",
            model_provider=self.mock_provider,
            max_tokens=4000,
            temperature=0.7,
            top_p=1.0,
            top_k=0,
            parallel_tool_calls=False,
            max_retries=3,
        )
        
        self.agent_config = AgentConfig(
            model=self.mock_model_config,
            max_steps=10,
            tools=["sequentialthinking", "task_done"],  # Use only existing tools
            allow_mcp_servers=[],
            mcp_servers_config={},
        )
        
        # Mock LLMClient to avoid actual API calls
        self.llm_client_patcher = patch("trae_agent.agent.base_agent.LLMClient")
        mock_llm_client_class = self.llm_client_patcher.start()
        mock_llm_client_instance = MagicMock()
        mock_llm_client_instance.provider.value = "openai"
        mock_llm_client_class.return_value = mock_llm_client_instance
        
        self.agent = LongContextAgent(self.agent_config)

    def tearDown(self):
        # Stop the patch
        self.llm_client_patcher.stop()
        
        # Clean up agent's temp directory
        self.agent._cleanup_temp_files()

    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent.temp_dir)
        self.assertIsNone(self.agent.context_file_path)
        self.assertEqual(self.agent.max_steps, 10)

    def test_new_task_setup(self):
        """Test new task setup with context and query."""
        task = "Analyze the provided text and summarize key findings"
        context = "This is a long context text that needs to be processed in chunks. " * 100
        query = "What are the main themes in this text?"
        
        extra_args = {
            "context": context,
            "query": query
        }
        
        self.agent.new_task(task, extra_args)
        
        # Check task setup
        self.assertEqual(self.agent._task, task)
        self.assertGreater(len(self.agent._tools), 0)
        self.assertGreater(len(self.agent._initial_messages), 0)
        
        # Check system prompt is set
        system_message = self.agent._initial_messages[0]
        self.assertEqual(system_message.role, "system")
        self.assertIn("long context", system_message.content.lower())
        
        # Check user messages contain task info
        user_messages = [msg for msg in self.agent._initial_messages if msg.role == "user"]
        self.assertGreater(len(user_messages), 0)
        
        # First user message should contain task and query
        first_user_msg = user_messages[0].content
        self.assertIn(task, first_user_msg)
        self.assertIn(query, first_user_msg)
        self.assertIn(f"{len(context):,}", first_user_msg)  # Check for comma-formatted number

    def test_tool_registration(self):
        """Test that custom tools are properly registered."""
        task = "Test task"
        self.agent.new_task(task, {"context": "test", "query": "test"})
        
        # Check that our custom tools are available
        tool_names = [tool.get_name() for tool in self.agent._tools]
        self.assertIn("context_file_manager", tool_names)
        self.assertIn("sequentialthinking", tool_names)
        self.assertIn("task_done", tool_names)

    def test_system_prompt_content(self):
        """Test system prompt contains expected content."""
        system_prompt = self.agent.get_system_prompt()
        
        key_concepts = [
            "long context",
            "problem decomposition",
            "cot simplification",
            "chunk",
            "context_file_manager",
            "meta_prompt_tool",
            "text_chunk_tool",
            "sequentialthinking"
        ]
        
        for concept in key_concepts:
            self.assertIn(concept.lower(), system_prompt.lower())

    def test_temp_directory_creation(self):
        """Test that temp directory is created and cleaned up."""
        import os
        
        temp_dir = self.agent.temp_dir
        self.assertTrue(os.path.exists(temp_dir))
        self.assertTrue(temp_dir.startswith("/tmp/long_context_") or 
                       temp_dir.startswith(tempfile.gettempdir()))
        
        # Test cleanup
        self.agent._cleanup_temp_files()
        self.assertFalse(os.path.exists(temp_dir))

    def test_task_completion_detection(self):
        """Test task completion detection."""
        # Mock LLM response with task_done tool call
        mock_response = Mock()
        mock_response.tool_calls = [Mock()]
        mock_response.tool_calls[0].name = "task_done"
        
        result = self.agent.llm_indicates_task_completed(mock_response)
        self.assertTrue(result)
        
        # Test without task_done
        mock_response.tool_calls[0].name = "other_tool"
        result = self.agent.llm_indicates_task_completed(mock_response)
        self.assertFalse(result)
        
        # Test with no tool calls
        mock_response.tool_calls = None
        result = self.agent.llm_indicates_task_completed(mock_response)
        self.assertFalse(result)

    def test_tool_initialization_with_dependencies(self):
        """Test that tools are initialized with proper dependencies."""
        task = "Test task with all tools"
        extra_args = {"context": "test context", "query": "test query"}
        
        # Use all available tools
        tool_names = [
            "context_file_manager", 
            "meta_prompt_tool",
            "text_chunk_tool",
            "sequentialthinking",
            "task_done"
        ]
        
        self.agent.new_task(task, extra_args, tool_names)
        
        # Verify tools are initialized
        self.assertEqual(len(self.agent._tools), len(tool_names))
        
        # Check that context_file_manager has agent reference
        context_tool = next((t for t in self.agent._tools if t.get_name() == "context_file_manager"), None)
        self.assertIsNotNone(context_tool)
        self.assertIs(context_tool.agent, self.agent)

    async def test_workflow_message_structure(self):
        """Test that workflow messages are properly structured."""
        task = "Analyze scientific paper"
        context = "Long scientific paper content here..." * 200
        query = "What are the key findings and methodology?"
        
        extra_args = {"context": context, "query": query}
        self.agent.new_task(task, extra_args)
        
        # Check message structure
        messages = self.agent._initial_messages
        self.assertEqual(len(messages), 3)  # system + 2 user messages
        
        # System message
        self.assertEqual(messages[0].role, "system")
        
        # First user message with task and workflow
        self.assertEqual(messages[1].role, "user")
        self.assertIn("**Task:**", messages[1].content)
        self.assertIn("**Your Workflow:**", messages[1].content)
        self.assertIn(query, messages[1].content)
        
        # Second user message with context
        self.assertEqual(messages[2].role, "user")
        self.assertIn("**Full Context Text to Process:**", messages[2].content)
        self.assertIn(context, messages[2].content)


if __name__ == "__main__":
    unittest.main()