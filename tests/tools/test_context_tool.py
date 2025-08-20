# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import tempfile
import unittest
from unittest.mock import Mock

from trae_agent.tools.base import ToolCallArguments
from trae_agent.tools.context_tool import ContextFileManagerTool


class TestContextFileManagerTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock agent with temp_dir
        self.mock_agent = Mock()
        self.mock_agent.temp_dir = tempfile.mkdtemp(prefix="test_context_")
        self.mock_agent.context_file_path = None
        
        self.tool = ContextFileManagerTool(model_provider="openai", agent=self.mock_agent)
        
        # Sample context text for testing
        self.sample_context = """This is a long context text for testing.
It contains multiple paragraphs and sentences.
The purpose is to test the context file management functionality.
This text should be saved to a file and then processed in chunks.
Each chunk should maintain semantic boundaries when possible.
The tool should provide accurate statistics about the text.
This includes character count, word count, and estimated tokens.
The chunking strategy should be flexible and configurable."""

    def tearDown(self):
        # Clean up temp files
        if self.mock_agent.context_file_path and os.path.exists(self.mock_agent.context_file_path):
            os.remove(self.mock_agent.context_file_path)
        if os.path.exists(self.mock_agent.temp_dir):
            os.rmdir(self.mock_agent.temp_dir)

    async def test_tool_initialization(self):
        self.assertEqual(self.tool.get_name(), "context_file_manager")
        self.assertIn("context", self.tool.get_description().lower())
        
        params = self.tool.get_parameters()
        param_names = [p.name for p in params]
        
        self.assertIn("action", param_names)
        self.assertIn("context", param_names)
        self.assertIn("chunk_size", param_names)
        self.assertIn("overlap", param_names)

    async def test_save_context_success(self):
        arguments = ToolCallArguments({
            "action": "save_context",
            "context": self.sample_context
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 0)
        self.assertIsNotNone(result.output)
        self.assertIn("Context saved successfully!", result.output)
        self.assertIn("File path:", result.output)
        self.assertIn("Statistics:", result.output)
        self.assertIn(f"{len(self.sample_context):,} characters", result.output)
        
        # Verify agent state was updated
        self.assertIsNotNone(self.mock_agent.context_file_path)
        self.assertTrue(os.path.exists(self.mock_agent.context_file_path))
        
        # Verify file content
        with open(self.mock_agent.context_file_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        self.assertEqual(saved_content, self.sample_context)

    async def test_save_context_missing_context(self):
        arguments = ToolCallArguments({
            "action": "save_context"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("Context text is required", result.error)

    async def test_get_chunk_info_success(self):
        # First save context
        save_args = ToolCallArguments({
            "action": "save_context",
            "context": self.sample_context
        })
        await self.tool.execute(save_args)
        
        # Then get chunk info
        chunk_args = ToolCallArguments({
            "action": "get_chunk_info",
            "chunk_size": 100,
            "overlap": 20
        })
        
        result = await self.tool.execute(chunk_args)
        
        self.assertEqual(result.error_code, 0)
        self.assertIn("Chunk Analysis:", result.output)
        self.assertIn(f"Total size: {len(self.sample_context):,} characters", result.output)
        self.assertIn("Chunk size: 100 characters", result.output)
        self.assertIn("Overlap: 20 characters", result.output)
        self.assertIn("Chunk boundaries (start:end):", result.output)

    async def test_get_chunk_info_no_context_file(self):
        arguments = ToolCallArguments({
            "action": "get_chunk_info"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("No context file found", result.error)

    async def test_plan_chunks_success(self):
        arguments = ToolCallArguments({
            "action": "plan_chunks",
            "chunk_size": 5000,
            "overlap": 100
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 0)
        self.assertIn("Chunking Strategy Planning:", result.output)
        self.assertIn("Recommended Settings:", result.output)
        self.assertIn("Chunk size: 5,000 characters", result.output)
        self.assertIn("Overlap: 100 characters", result.output)
        self.assertIn("Semantic Considerations:", result.output)
        self.assertIn("Processing Tips:", result.output)

    async def test_invalid_action(self):
        arguments = ToolCallArguments({
            "action": "invalid_action"
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 1)
        self.assertIn("Unknown action: invalid_action", result.error)

    async def test_context_statistics_accuracy(self):
        test_context = "Hello world!\nThis is line 2.\nFinal line."
        
        arguments = ToolCallArguments({
            "action": "save_context",
            "context": test_context
        })
        
        result = await self.tool.execute(arguments)
        
        self.assertEqual(result.error_code, 0)
        
        # Verify statistics
        expected_chars = len(test_context)
        expected_words = len(test_context.split())
        expected_lines = test_context.count('\n') + 1
        
        self.assertIn(f"{expected_chars:,} characters", result.output)
        self.assertIn(f"Lines: {expected_lines:,}", result.output)
        self.assertIn(f"Words: {expected_words:,}", result.output)


if __name__ == "__main__":
    unittest.main()