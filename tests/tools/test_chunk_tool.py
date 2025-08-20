# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from trae_agent.tools.base import ToolCallArguments
from trae_agent.tools.chunk_tool import TextChunkTool
from trae_agent.utils.llm_clients.llm_basics import LLMResponse
from trae_agent.utils.llm_clients.llm_client import LLMClient
from trae_agent.utils.config import ModelProvider


class TestTextChunkTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create dummy provider and model config
        mock_provider = ModelProvider(
            api_key="test-dummy-api-key",
            provider="openai"
        )
        
        # Create mock LLM client directly
        self.mock_llm_client = Mock()
        self.mock_llm_client.provider = mock_provider
        
        # Create mock model config for the LLM client
        from trae_agent.utils.config import ModelConfig
        self.mock_llm_client.model_config = ModelConfig(
            model="test-model",
            model_provider=mock_provider,
            max_tokens=4096,
            temperature=0.5,
            top_p=1.0,
            top_k=0,
            parallel_tool_calls=False,
            max_retries=3,
        )
        
        # Create tool with the mock client
        self.tool = TextChunkTool(model_provider="openai", llm_client=self.mock_llm_client)
        
        # Create test file with content
        self.test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
        self.test_content = """This is the first paragraph of test content.
It contains multiple sentences and provides context.

This is the second paragraph with different information.
It helps test the chunking functionality properly.

Finally, this is the third paragraph.
It concludes our test document content."""
        self.test_file.write(self.test_content)
        self.test_file.close()

    def tearDown(self):
        # Clean up test file
        if os.path.exists(self.test_file.name):
            os.unlink(self.test_file.name)

    async def test_tool_initialization(self):
        self.assertEqual(self.tool.get_name(), "text_chunk_tool")
        self.assertIn("chunk", self.tool.get_description().lower())
        
        params = self.tool.get_parameters()
        param_names = [p.name for p in params]
        
        self.assertIn("file_path", param_names)
        self.assertIn("start_pos", param_names)
        self.assertIn("end_pos", param_names)
        self.assertIn("query", param_names)
        self.assertIn("enhanced_prompt", param_names)
        self.assertIn("chunk_id", param_names)

    async def test_process_chunk_success(self):
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Analysis result: The first paragraph discusses test content with multiple sentences."
        self.mock_llm_client.chat = Mock(return_value=mock_response)
        
        arguments = ToolCallArguments({
            "file_path": self.test_file.name,
            "start_pos": 0,
            "end_pos": 100,
            "query": "What does the first paragraph discuss?",
            "enhanced_prompt": "Analyze the content and provide a clear summary of the main topics discussed.",
            "chunk_id": "chunk_1"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 0)
        self.assertIn("Chunk Processing Result", result.output)
        self.assertIn("chunk_1", result.output)
        self.assertIn("Analysis result", result.output)
        self.assertIn("**Processing Status:** Complete", result.output)
        
        # Verify LLM client was called
        self.mock_llm_client.chat.assert_called_once()

    async def test_process_chunk_missing_file(self):
        arguments = ToolCallArguments({
            "file_path": "/nonexistent/file.txt",
            "start_pos": 0,
            "end_pos": 100,
            "query": "Test query",
            "enhanced_prompt": "Test prompt"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("File not found", result.error)

    async def test_process_chunk_invalid_positions(self):
        arguments = ToolCallArguments({
            "file_path": self.test_file.name,
            "start_pos": -1,
            "end_pos": 100,
            "query": "Test query",
            "enhanced_prompt": "Test prompt"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("Invalid position range", result.error)

    async def test_process_chunk_end_before_start(self):
        arguments = ToolCallArguments({
            "file_path": self.test_file.name,
            "start_pos": 100,
            "end_pos": 50,
            "query": "Test query",
            "enhanced_prompt": "Test prompt"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("Invalid position range", result.error)

    async def test_process_chunk_no_llm_client(self):
        tool_without_llm = TextChunkTool(model_provider="openai", llm_client=None)
        
        arguments = ToolCallArguments({
            "file_path": self.test_file.name,
            "start_pos": 0,
            "end_pos": 100,
            "query": "Test query",
            "enhanced_prompt": "Test prompt"
        })
        
        result = await tool_without_llm.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("LLM client not available", result.error)

    async def test_process_chunk_missing_parameters(self):
        arguments = ToolCallArguments({
            "file_path": self.test_file.name,
            "start_pos": 0,
            # Missing end_pos, query, enhanced_prompt
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("Required parameters", result.error)

    async def test_process_chunk_empty_content(self):
        # Create empty file
        empty_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        empty_file.close()
        
        try:
            arguments = ToolCallArguments({
                "file_path": empty_file.name,
                "start_pos": 0,
                "end_pos": 100,
                "query": "Test query",
                "enhanced_prompt": "Test prompt"
            })
            
            result = await self.tool.execute(arguments)
            
            self.assertEqual(result.error_code, 1)
            self.assertIn("No content found", result.error)
        finally:
            os.unlink(empty_file.name)

    async def test_process_chunk_llm_failure(self):
        # Mock LLM to return None
        self.mock_llm_client.chat = Mock(return_value=None)
        
        arguments = ToolCallArguments({
            "file_path": self.test_file.name,
            "start_pos": 0,
            "end_pos": 50,
            "query": "Test query",
            "enhanced_prompt": "Test prompt"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("Failed to process text chunk", result.error)

    async def test_process_chunk_llm_exception(self):
        # Mock LLM to raise exception
        self.mock_llm_client.chat = Mock(side_effect=Exception("LLM processing error"))
        
        arguments = ToolCallArguments({
            "file_path": self.test_file.name,
            "start_pos": 0,
            "end_pos": 50,
            "query": "Test query",
            "enhanced_prompt": "Test prompt"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("Error processing text chunk", result.error)
        self.assertIn("LLM processing error", result.error)

    async def test_chunk_content_extraction(self):
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Extracted content analysis"
        self.mock_llm_client.chat = Mock(return_value=mock_response)
        
        # Test extracting specific range
        start_pos = 20
        end_pos = 80
        expected_chunk = self.test_content[start_pos:end_pos]
        
        arguments = ToolCallArguments({
            "file_path": self.test_file.name,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "query": "Analyze this chunk",
            "enhanced_prompt": "Process the provided text"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 0)
        
        # Verify the chunk size is reported correctly
        self.assertIn(f"**Chunk Size:** {len(expected_chunk):,} characters", result.output)
        
        # Verify LLM received the correct chunk
        call_args = self.mock_llm_client.chat.call_args[0][0]  # messages
        user_message = call_args[1].content
        self.assertIn(expected_chunk, user_message)

    async def test_default_chunk_id(self):
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Test response"
        self.mock_llm_client.chat = Mock(return_value=mock_response)
        
        arguments = ToolCallArguments({
            "file_path": self.test_file.name,
            "start_pos": 10,
            "end_pos": 50,
            "query": "Test query",
            "enhanced_prompt": "Test prompt"
            # No chunk_id provided
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 0)
        # Should use default chunk_id format
        self.assertIn("chunk_10_50", result.output)


if __name__ == "__main__":
    unittest.main()