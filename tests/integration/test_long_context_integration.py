# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Integration tests for LongContextAgent that use real API calls.
These tests are skipped by default and only run when API keys are available.
"""

import os
import unittest
import asyncio
from unittest import SkipTest

from trae_agent.agent.long_context_agent import LongContextAgent
from trae_agent.utils.config import AgentConfig, ModelConfig, ModelProvider


class TestLongContextIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests requiring real API calls."""

    def setUp(self):
        # Skip if no API keys available
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise SkipTest("API key not available - skipping integration tests")
        
        self.provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
        self.model = "gpt-4o-mini" if self.provider == "openai" else "claude-3-haiku-20240307"
        
        # Use smaller, cheaper models for integration tests
        model_provider = ModelProvider(
            api_key=self.api_key,
            provider=self.provider
        )
        
        model_config = ModelConfig(
            model=self.model,
            model_provider=model_provider,
            max_tokens=1000,  # Smaller token limit for cost control
            temperature=0.3,
            top_p=1.0,
            top_k=0,
            parallel_tool_calls=False,
            max_retries=2,
        )
        
        self.agent_config = AgentConfig(
            model=model_config,
            max_steps=10,  # Limit steps for cost control
            tools=[],
            allow_mcp_servers=[],
            mcp_servers_config={},
        )

    async def test_basic_long_context_processing(self):
        """Test basic long context processing with real API."""
        # Use a moderately long context to test functionality without excessive cost
        context = """
        The history of artificial intelligence spans several decades, with key milestones including:
        
        1950s: Alan Turing proposes the Turing Test as a measure of machine intelligence.
        1956: The term "artificial intelligence" is coined at the Dartmouth Conference.
        1960s: Early AI programs like ELIZA demonstrate natural language processing capabilities.
        1970s-80s: Expert systems emerge as practical AI applications in specialized domains.
        1990s: Machine learning approaches gain prominence, statistical methods improve.
        2000s: Big data and computational power enable more sophisticated AI applications.
        2010s: Deep learning revolutionizes computer vision, natural language processing.
        2020s: Large language models like GPT and BERT achieve remarkable capabilities.
        
        Key challenges throughout AI history include:
        - The frame problem: difficulty in representing changing situations
        - Knowledge representation: how to encode human knowledge in machines
        - Common sense reasoning: understanding implicit knowledge humans take for granted
        - Scalability: making AI systems work on real-world problems
        - Ethics and safety: ensuring AI systems are beneficial and aligned with human values
        
        Current applications of AI include:
        - Search engines and recommendation systems
        - Computer vision for image recognition and analysis  
        - Natural language processing for translation and chatbots
        - Autonomous vehicles and robotics
        - Medical diagnosis and drug discovery
        - Financial trading and risk assessment
        
        Future directions in AI research focus on:
        - Artificial general intelligence (AGI) that matches human cognitive abilities
        - Explainable AI that can explain its reasoning processes
        - Robust AI that performs reliably in diverse environments
        - Efficient AI that requires less computational resources
        - Cooperative AI that works effectively with humans
        """ * 5  # Repeat to make it longer
        
        query = "What are the key milestones in AI history and what are the main current applications?"
        
        agent = LongContextAgent(self.agent_config)
        
        task = "Analyze the AI history text and answer the query"
        extra_args = {
            "context": context,
            "query": query
        }
        
        agent.new_task(task, extra_args)
        execution = await agent.execute_task()
        
        # Basic assertions
        self.assertIsNotNone(execution.final_result)
        self.assertTrue(len(execution.final_result) > 100)  # Should have substantial content
        self.assertGreater(len(execution.steps), 2)  # Should have multiple steps
        
        # Should mention key concepts from the query
        result_lower = execution.final_result.lower()
        self.assertTrue(any(term in result_lower for term in ["turing", "dartmouth", "1950s", "1960s"]))
        self.assertTrue(any(term in result_lower for term in ["search", "vision", "language", "autonomous"]))
        
        print(f"Integration test completed successfully with {self.provider}")
        print(f"Steps: {len(execution.steps)}, Time: {execution.execution_time:.2f}s")


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/ -v -s
    unittest.main()