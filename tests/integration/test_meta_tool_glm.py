# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest import SkipTest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trae_agent.tools.base import ToolCallArguments
from trae_agent.tools.meta_tool import MetaPromptTool
from trae_agent.utils.config import ModelConfig, ModelProvider
from trae_agent.utils.llm_clients.glm_client import GLMClient


class TestMetaPromptToolWithGLM(unittest.IsolatedAsyncioTestCase):
    """Integration tests for MetaPromptTool using real GLM API."""

    def setUp(self):
        # Skip if running in CI or if we want to avoid API costs
        if os.getenv("SKIP_GLM_TESTS") == "1":
            raise SkipTest("GLM tests skipped (SKIP_GLM_TESTS=1)")
            
        # GLM API configuration
        self.api_key = "e34be58db90244488b2f5ad00008698b.wlAPdq5wKIVuJiNA"
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/"
        
        # Create GLM model config
        model_provider = ModelProvider(
            api_key=self.api_key,
            provider="glm",
            base_url=self.base_url
        )
        
        model_config = ModelConfig(
            model="glm-4.5",
            model_provider=model_provider,
            max_tokens=1500,
            temperature=0.2,  # Lower temperature for more consistent prompt enhancement
            top_p=1.0,
            top_k=0,
            parallel_tool_calls=False,
            max_retries=3,
        )
        
        # Create real GLM client
        llm_client = GLMClient(model_config)
        
        # Create tool with real GLM client
        self.tool = MetaPromptTool(model_provider="glm", llm_client=llm_client)

    async def test_document_summarization_enhancement(self):
        """Test enhancing a document summarization prompt for chunk processing."""
        print(f"\n🔬 Testing Document Summarization Enhancement with GLM-4.5")
        print("🎯 Task: Document chunk summarization for sub-agent")
        
        arguments = ToolCallArguments({
            "simple_prompt": "总结这段文字的主要内容",
            "task_context": "处理100页学术论文的子段落，需要为每个1000字片段提取核心观点，要求30秒内处理完成"
        })
        
        print(f"📝 Original prompt: {arguments['simple_prompt']}")
        print(f"🎯 Context: {arguments['task_context']}")
        print("⏳ Enhancing prompt for chunk processing...")
        
        result = await self.tool.execute(arguments)
        
        # Verify the enhanced prompt
        self.assertIsNone(result.error, f"Tool execution failed: {result.error}")
        self.assertIsNotNone(result.output)
        self.assertGreater(len(result.output), len(arguments["simple_prompt"]))
        
        print(f"✅ Enhanced Prompt Result:")
        print(f"📊 Original length: {len(arguments['simple_prompt'])} characters")
        print(f"📈 Enhanced length: {len(result.output)} characters")
        print(f"🤖 GLM Enhanced Prompt:")
        print("-" * 50)
        print(result.output)
        print("-" * 50)
        
        # Check for chunk-specific and academic processing instructions
        academic_terms = ["片段", "chunk", "简洁", "关键", "观点", "核心", "学术", "论文", "段落"]
        found_terms = [term for term in academic_terms if term in result.output]
        self.assertGreater(len(found_terms), 2,
                          f"Enhanced prompt should contain academic chunk processing instructions, found: {found_terms}")

    async def test_information_extraction_enhancement(self):
        """Test enhancing an information extraction prompt for chunk processing."""
        print(f"\n🔬 Testing Information Extraction Enhancement with GLM-4.5")
        
        arguments = ToolCallArguments({
            "simple_prompt": "提取重要信息",
            "task_context": "从500页技术手册中提取API接口信息，每个片段500字，需要识别函数名、参数、返回值"
        })
        
        print(f"🔍 Extraction prompt: {arguments['simple_prompt']}")
        print("⏳ Enhancing extraction prompt for sub-agent...")
        
        result = await self.tool.execute(arguments)
        
        # Verify extraction enhancement
        self.assertIsNone(result.error, f"Tool execution failed: {result.error}")
        self.assertIsNotNone(result.output)
        self.assertGreater(len(result.output), 50)
        
        # Check for API extraction and technical terms
        api_terms = ["提取", "API", "接口", "函数", "参数", "返回值", "技术", "手册", "片段"]
        found_terms = [term for term in api_terms if term in result.output]
        
        self.assertGreater(len(found_terms), 3,
                          f"API extraction prompt should contain relevant terms: {found_terms}")
        
        print(f"✅ Technical API Extraction Enhancement Result:")
        print(f"🔍 Found API extraction terms: {found_terms}")
        print(f"🤖 Enhanced API Extraction Prompt:")
        print("-" * 50)
        print(result.output)
        print("-" * 50)

    async def test_classification_task_enhancement(self):
        """Test enhancing a classification prompt for chunk processing."""
        print(f"\n🔬 Testing Classification Task Enhancement with GLM-4.5")
        
        arguments = ToolCallArguments({
            "simple_prompt": "判断文本类型",
            "task_context": "对客服聊天记录进行情感分类，每段300字对话片段，需分类为积极/消极/中性，并行处理1000个片段"
        })
        
        print(f"🏷️ Classification prompt: {arguments['simple_prompt']}")
        print("⏳ Enhancing classification prompt for chunk processing...")
        
        result = await self.tool.execute(arguments)
        
        # Verify classification enhancement
        self.assertIsNone(result.error, f"Tool execution failed: {result.error}")
        self.assertIsNotNone(result.output)
        self.assertGreater(len(result.output), 50)
        
        # Check for sentiment classification and customer service terms
        sentiment_terms = ["分类", "情感", "积极", "消极", "中性", "客服", "对话", "并行", "片段"]
        found_terms = [term for term in sentiment_terms if term in result.output]
        
        self.assertGreater(len(found_terms), 3,
                          f"Sentiment classification prompt should contain relevant terms: {found_terms}")
        
        print(f"✅ Customer Service Sentiment Classification Enhancement Result:")
        print(f"🏷️ Found sentiment classification terms: {found_terms}")
        print(f"🤖 Enhanced Customer Service Classification Prompt:")
        print("-" * 50)
        print(result.output)
        print("-" * 50)

    async def test_entity_recognition_enhancement(self):
        """Test enhancing an entity recognition prompt for chunk processing."""
        print(f"\n🔬 Testing Entity Recognition Enhancement with GLM-4.5")
        
        arguments = ToolCallArguments({
            "simple_prompt": "识别重要实体",
            "task_context": "从新闻语料库提取命名实体，每个800字新闻片段识别人名、公司名、地名，需要处理可能截断的实体边界"
        })
        
        print(f"🏃 Entity recognition prompt: {arguments['simple_prompt']}")
        print("⏳ Enhancing entity recognition prompt for chunk processing...")
        
        result = await self.tool.execute(arguments)
        
        # Verify entity recognition enhancement
        self.assertIsNone(result.error, f"Tool execution failed: {result.error}")
        self.assertIsNotNone(result.output)
        self.assertGreater(len(result.output), 50)
        
        # Check for news entity recognition and boundary handling terms
        news_entity_terms = ["实体", "新闻", "人名", "公司", "地名", "边界", "截断", "语料", "片段"]
        found_terms = [term for term in news_entity_terms if term in result.output]
        
        self.assertGreater(len(found_terms), 3,
                          f"News entity recognition prompt should contain relevant terms: {found_terms}")
        
        print(f"✅ News Entity Recognition with Boundary Handling Result:")
        print(f"🏃 Found news entity terms: {found_terms}")
        print(f"🤖 Enhanced News Entity Recognition Prompt:")
        print("-" * 50)
        print(result.output)
        print("-" * 50)

    async def test_sentiment_analysis_enhancement(self):
        """Test enhancing a sentiment analysis prompt for chunk processing."""
        print(f"\n🔬 Testing Sentiment Analysis Enhancement with GLM-4.5")
        
        arguments = ToolCallArguments({
            "simple_prompt": "分析情感倾向",
            "task_context": "对电商评论进行情感挖掘，每个200字评论片段，输出情感得分和关键词，10秒内完成单片段分析"
        })
        
        print(f"😊 Sentiment analysis prompt: {arguments['simple_prompt']}")
        print(f"📄 Context: {arguments['task_context']}")
        print("⏳ Enhancing sentiment analysis prompt for chunk processing...")
        
        result = await self.tool.execute(arguments)
        
        # Verify sentiment analysis enhancement
        self.assertIsNone(result.error, f"Tool execution failed: {result.error}")
        self.assertIsNotNone(result.output)
        self.assertGreater(len(result.output), 50)
        
        # Check for e-commerce sentiment analysis terms
        ecommerce_terms = ["情感", "电商", "评论", "得分", "关键词", "挖掘", "片段", "分析"]
        found_terms = [term for term in ecommerce_terms if term in result.output]
        
        self.assertGreater(len(found_terms), 3,
                          f"E-commerce sentiment analysis prompt should contain relevant terms: {found_terms}")
        
        print(f"✅ E-commerce Review Sentiment Mining Enhancement Result:")
        print(f"😊 Found e-commerce sentiment terms: {found_terms}")
        print(f"🤖 Enhanced E-commerce Sentiment Mining Prompt:")
        print("-" * 50)
        print(result.output)
        print("-" * 50)

    async def test_chunk_boundary_handling_enhancement(self):
        """Test enhancing a prompt for handling chunk boundary cases."""
        print(f"\n🔬 Testing Chunk Boundary Handling Enhancement with GLM-4.5")
        
        # Test chunk boundary handling prompt
        arguments = ToolCallArguments({
            "simple_prompt": "处理不完整的文本",
            "task_context": "处理法律合同条款切分，每个1200字片段可能在条款中间截断，需要标记不完整信息并指示后续处理"
        })
        
        print("🔀 Testing chunk boundary handling enhancement...")
        print(f"📝 Original prompt: {arguments['simple_prompt']}")
        print(f"🎯 Context: {arguments['task_context']}")
        
        result1 = await self.tool.execute(arguments)
        result2 = await self.tool.execute(arguments)
        
        # Both results should be meaningful
        self.assertIsNone(result1.error, f"First result failed: {result1.error}")
        self.assertIsNone(result2.error, f"Second result failed: {result2.error}")
        self.assertIsNotNone(result1.output)
        self.assertIsNotNone(result2.output)
        self.assertGreater(len(result1.output), 50)
        self.assertGreater(len(result2.output), 50)
        
        # Check for legal contract boundary handling terms
        legal_terms = ["法律", "合同", "条款", "截断", "边界", "标记", "不完整", "后续", "片段"]
        found_in_result1 = [term for term in legal_terms if term in result1.output]
        found_in_result2 = [term for term in legal_terms if term in result2.output]
        
        self.assertGreater(len(found_in_result1), 2)
        self.assertGreater(len(found_in_result2), 2)
        
        print(f"✅ Legal Contract Chunk Boundary Enhancement Results:")
        print(f"📝 First result length: {len(result1.output)}")
        print(f"📝 Second result length: {len(result2.output)}")
        print(f"🔀 Legal boundary terms in result 1: {found_in_result1}")
        print(f"🔀 Legal boundary terms in result 2: {found_in_result2}")
        print(f"🤖 Enhanced Legal Contract Boundary Handling Prompt:")
        print("-" * 50)
        print(result1.output)
        print("-" * 50)


if __name__ == "__main__":
    # Note: These tests make real API calls to GLM
    # Set SKIP_GLM_TESTS=1 to skip them
    unittest.main()