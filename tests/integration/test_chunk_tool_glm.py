# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import SkipTest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trae_agent.tools.base import ToolCallArguments
from trae_agent.tools.chunk_tool import TextChunkTool
from trae_agent.utils.config import ModelConfig, ModelProvider
from trae_agent.utils.llm_clients.glm_client import GLMClient


class TestTextChunkToolWithGLM(unittest.IsolatedAsyncioTestCase):
    """Integration tests for TextChunkTool using real GLM API."""

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
            max_tokens=1000,
            temperature=0.3,
            top_p=1.0,
            top_k=0,
            parallel_tool_calls=False,
            max_retries=3,
        )
        
        # Create real GLM client
        llm_client = GLMClient(model_config)
        
        # Create tool with real GLM client
        self.tool = TextChunkTool(model_provider="glm", llm_client=llm_client)
        
        # Create test file with Chinese content
        self.test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
        self.test_content = """人工智能的发展历程可以分为几个重要阶段：

1950年代：艾伦·图灵提出图灵测试，标志着人工智能理论基础的建立。这个测试提出了一个重要问题：机器能思考吗？

1956年：达特茅斯会议正式提出"人工智能"这一概念。约翰·麦卡锡、马文·明斯基等计算机科学家聚集在一起，讨论机器智能的可能性。

1960-70年代：专家系统兴起，如MYCIN医疗诊断系统。这些系统能够在特定领域内模拟人类专家的决策过程。

1980年代：机器学习理论逐渐发展，神经网络重获关注。反向传播算法的提出为深度学习奠定了基础。

1990年代：互联网普及推动了数据积累和算法改进。大量数据的可获得性为机器学习算法提供了更好的训练素材。

2000年代：深度学习突破，图像识别和自然语言处理取得重大进展。深度卷积神经网络在图像识别任务中表现出色。

2010年代：大语言模型出现，如GPT、BERT等。这些模型在自然语言理解和生成任务中展现出惊人的能力。

2020年代：ChatGPT等对话式AI广泛应用，AGI成为新目标。人工智能开始在更多领域展现出接近人类的能力。

人工智能技术在各行业的应用包括：
- 医疗：辅助诊断、药物发现、个性化治疗
- 金融：风险评估、算法交易、客户服务  
- 教育：个性化学习、智能辅导
- 交通：自动驾驶、智能导航
- 制造业：质量控制、预测性维护
- 娱乐：内容推荐、游戏AI

未来发展趋势包括更强的推理能力、多模态融合、边缘计算部署等。"""
        
        self.test_file.write(self.test_content)
        self.test_file.close()

    def tearDown(self):
        # Clean up test file
        if os.path.exists(self.test_file.name):
            os.unlink(self.test_file.name)

    async def test_chunk_analysis_with_glm(self):
        """Test chunk analysis using real GLM API."""
        print(f"\n🔬 Testing TextChunkTool with GLM-4.5")
        print(f"📄 Test file: {self.test_file.name}")
        print(f"📝 Content length: {len(self.test_content)} characters")
        
        # Get first ~500 characters for analysis (simulating document chunk processing)
        start_pos = 0
        end_pos = 500
        query = "从这个文档片段中提取关键时间节点和历史事件，用于构建AI发展时间线"
        enhanced_prompt = "作为文档分析子代理，分析这个AI历史文档片段。只处理片段内容，提取：1)明确时间点(年代) 2)关键人物 3)重要事件。输出简洁，便于后续合并处理。如果信息不完整，标记为部分信息。"
        
        arguments = ToolCallArguments({
            "file_path": self.test_file.name,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "query": query,
            "enhanced_prompt": enhanced_prompt,
            "chunk_id": "ai_history_chunk_001"
        })
        
        print(f"🎯 Query: {query}")
        print(f"📍 Chunk position: {start_pos}-{end_pos}")
        print("⏳ Sub-agent analyzing document chunk with GLM...")
        
        result = await self.tool.execute(arguments)
        
        # Verify the result
        self.assertIsNone(result.error, f"Tool execution failed: {result.error}")
        self.assertIsNotNone(result.output)
        self.assertGreater(len(result.output), 50)
        
        print(f"✅ Sub-agent Chunk Analysis Result:")
        print(f"📊 Response length: {len(result.output)} characters")
        print(f"🤖 GLM Chunk Analysis: {result.output}")
        
        # Check for chunk processing effectiveness and key extraction
        chunk_terms = ["时间", "事件", "片段", "简洁", "合并"] + ["1950年代", "图灵", "1956年", "达特茅斯", "人工智能"]
        found_terms = [term for term in chunk_terms if term in result.output]
        
        self.assertGreater(len(found_terms), 2, 
                          f"Expected chunk processing to extract key information, found: {found_terms}")
        
        print(f"✅ Found chunk processing and key terms: {found_terms}")

    async def test_simple_chunk_processing(self):
        """Test simple chunk processing with GLM for sub-agent workflow."""
        print(f"\n🔬 Testing Sub-agent Chunk Processing with GLM-4.5")
        
        # Test a smaller chunk - first 200 characters (realistic chunk size)
        start_pos = 0
        end_pos = 200
        query = "作为子代理，快速处理这个文档片段"
        enhanced_prompt = "作为文档处理子代理，快速分析这个文档片段。只提取核心信息，输出简洁，便于与其他片段结果合并。如果内容不完整，标明边界信息。"
        
        arguments = ToolCallArguments({
            "file_path": self.test_file.name,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "query": query,
            "enhanced_prompt": enhanced_prompt
        })
        
        print(f"📍 Sub-agent processing chunk: {start_pos}-{end_pos}")
        result = await self.tool.execute(arguments)
        
        # Verify result for sub-agent chunk processing
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.output)
        
        # Check for sub-agent processing indicators
        subagent_terms = ["核心", "简洁", "片段", "合并", "边界"]
        found_subagent_terms = [term for term in subagent_terms if term in result.output]
        
        print(f"✅ Sub-agent chunk processing result: {result.output[:100]}...")
        print(f"🤖 Found sub-agent processing terms: {found_subagent_terms}")


if __name__ == "__main__":
    # Note: These tests make real API calls to GLM
    # Set SKIP_GLM_TESTS=1 to skip them
    unittest.main()