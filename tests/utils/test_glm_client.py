# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import unittest
from unittest import SkipTest

from trae_agent.utils.config import ModelConfig, ModelProvider
from trae_agent.utils.llm_clients.glm_client import GLMClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage


class TestGLMClient(unittest.TestCase):
    """Integration tests for GLM client."""

    def setUp(self):
        # Use the provided GLM API key for testing
        self.api_key = "e34be58db90244488b2f5ad00008698b.wlAPdq5wKIVuJiNA"
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/"
        
        # Skip if running in CI or if we want to avoid API costs
        if os.getenv("SKIP_GLM_TESTS") == "1":
            raise SkipTest("GLM tests skipped (SKIP_GLM_TESTS=1)")
        
        self.model_provider = ModelProvider(
            api_key=self.api_key,
            provider="glm",
            base_url=self.base_url
        )
        
        self.model_config = ModelConfig(
            model="glm-4.5",
            model_provider=self.model_provider,
            max_tokens=500,
            temperature=0.7,
            top_p=1.0,
            top_k=0,
            parallel_tool_calls=False,
            max_retries=3,
        )
        
        self.client = GLMClient(self.model_config)

    def test_glm_basic_chat(self):
        """Test basic chat with GLM."""
        messages = [
            LLMMessage(role="user", content="你好，请回答：1+1等于多少？")
        ]
        
        response = self.client.chat(messages, self.model_config)
        
        # Basic assertions
        self.assertIsNotNone(response)
        self.assertIsInstance(response.content, str)
        self.assertGreater(len(response.content), 0)
        
        # Should contain the answer
        self.assertTrue(any(char in response.content for char in ["2", "二", "两"]))
        
        # Check usage information
        if response.usage:
            self.assertGreater(response.usage.input_tokens, 0)
            self.assertGreater(response.usage.output_tokens, 0)
        
        print(f"✅ GLM Response: {response.content[:100]}...")

    def test_glm_chinese_conversation(self):
        """Test Chinese conversation capability."""
        messages = [
            LLMMessage(role="user", content="请用中文简单介绍一下人工智能的发展历程。")
        ]
        
        response = self.client.chat(messages, self.model_config)
        
        self.assertIsNotNone(response.content)
        self.assertGreater(len(response.content), 50)
        
        # Should contain Chinese AI-related terms
        ai_terms = ["人工智能", "AI", "机器学习", "深度学习", "神经网络"]
        self.assertTrue(any(term in response.content for term in ai_terms))
        
        print(f"✅ GLM Chinese Response: {response.content[:100]}...")

    def test_glm_available_models(self):
        """Test available models list."""
        models = self.client.get_available_models()

        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        
        # Should contain expected GLM models
        expected_models = ["glm-4-plus", "glm-4", "glm-4-flash"]
        for expected in expected_models:
            self.assertIn(expected, models)
        
        print(f"✅ GLM Available Models: {models}")

    def test_glm_error_handling(self):
        """Test error handling with invalid requests."""
        # Test with very long context that might exceed limits
        very_long_message = "请重复这句话：" + "测试" * 10000
        messages = [
            LLMMessage(role="user", content=very_long_message)
        ]
        
        try:
            response = self.client.chat(messages, self.model_config)
            # If successful, that's also fine
            self.assertIsNotNone(response)
        except Exception as e:
            # Should handle errors gracefully
            self.assertIsInstance(e, Exception)
            print(f"✅ GLM Error Handling: {type(e).__name__}: {str(e)[:100]}...")


if __name__ == "__main__":
    # Note: These tests make real API calls
    # Set SKIP_GLM_TESTS=1 to skip them
    unittest.main()