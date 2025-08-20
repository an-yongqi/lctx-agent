# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import unittest
from unittest.mock import AsyncMock, Mock, patch

from trae_agent.tools.base import ToolCallArguments
from trae_agent.tools.meta_tool import MetaPromptTool
from trae_agent.utils.llm_clients.llm_basics import LLMResponse
from trae_agent.utils.llm_clients.llm_client import LLMClient
from trae_agent.utils.config import ModelProvider


class TestMetaPromptTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create dummy provider and model config
        mock_provider = ModelProvider(
            api_key="test-dummy-api-key",
            provider="openai"
        )
        
        # Create mock LLM client directly
        self.mock_llm_client = Mock()
        self.mock_llm_client.provider = mock_provider
        
        # Create tool with the mock client
        self.tool = MetaPromptTool(model_provider="openai", llm_client=self.mock_llm_client)

    async def test_tool_initialization(self):
        self.assertEqual(self.tool.get_name(), "meta_prompt_tool")
        self.assertIn("prompt", self.tool.get_description().lower())
        
        params = self.tool.get_parameters()
        param_names = [p.name for p in params]
        
        self.assertIn("simple_prompt", param_names)
        self.assertIn("task_context", param_names)

    async def test_enhance_prompt_success(self):
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Enhanced detailed prompt with specific instructions and context"
        self.mock_llm_client.chat = Mock(return_value=mock_response)
        
        arguments = ToolCallArguments({
            "simple_prompt": "Summarize the text",
            "task_context": "Analyzing scientific papers"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 0)
        self.assertIn("Enhanced Prompt Generated Successfully:", result.output)
        self.assertIn("Enhanced detailed prompt", result.output)
        self.assertIn("Optimization Summary:", result.output)
        
        # Verify LLM client was called
        self.mock_llm_client.chat.assert_called_once()

    async def test_enhance_prompt_missing_simple_prompt(self):
        arguments = ToolCallArguments({
            "task_context": "Some context"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("simple_prompt is required", result.error)

    async def test_enhance_prompt_no_llm_client(self):
        tool_without_llm = MetaPromptTool(model_provider="openai", llm_client=None)
        
        arguments = ToolCallArguments({
            "simple_prompt": "Test prompt"
        })
        
        result = await tool_without_llm.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("LLM client not available", result.error)

    async def test_enhance_prompt_llm_failure(self):
        # Mock LLM to return None
        self.mock_llm_client.chat = Mock(return_value=None)
        
        arguments = ToolCallArguments({
            "simple_prompt": "Test prompt"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("Failed to generate enhanced prompt", result.error)

    async def test_enhance_prompt_llm_exception(self):
        # Mock LLM to raise exception
        self.mock_llm_client.chat = Mock(side_effect=Exception("LLM error"))
        
        arguments = ToolCallArguments({
            "simple_prompt": "Test prompt"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("Error generating meta prompt", result.error)
        self.assertIn("LLM error", result.error)

    async def test_enhance_prompt_with_all_params(self):
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Comprehensive enhanced prompt with all considerations"
        self.mock_llm_client.chat = Mock(return_value=mock_response)
        
        arguments = ToolCallArguments({
            "simple_prompt": "Analyze data",
            "task_context": "Financial analysis task"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 0)
        self.assertIn("Comprehensive enhanced prompt", result.output)
        
        # Verify the call included all parameters
        call_args = self.mock_llm_client.chat.call_args[0][0]  # messages
        user_message = call_args[1].content
        
        self.assertIn("Analyze data", user_message)
        self.assertIn("Financial analysis task", user_message)

    async def test_system_prompt_content(self):
        # Mock LLM response to check what system prompt was used
        mock_response = Mock()
        mock_response.content = "Test response"
        self.mock_llm_client.chat = Mock(return_value=mock_response)
        
        arguments = ToolCallArguments({
            "simple_prompt": "Test"
        })
        
        await self.tool.execute(arguments)
        
        # Check that system prompt contains key concepts
        call_args = self.mock_llm_client.chat.call_args[0][0]  # messages
        system_message = call_args[0].content
        
        self.assertIn("prompt engineer", system_message.lower())
        self.assertIn("chunk processing", system_message.lower())
        self.assertIn("sub-agent", system_message.lower())


if __name__ == "__main__":
    unittest.main()