#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
GLM Tools Integration Test
Simple integration test for GLM with our tools.
"""

import asyncio
import os
import tempfile
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trae_agent.tools.base import ToolCallArguments
from trae_agent.tools.meta_tool import MetaPromptTool
from trae_agent.tools.chunk_tool import TextChunkTool
from trae_agent.utils.config import ModelConfig, ModelProvider
from trae_agent.utils.llm_clients.glm_client import GLMClient


async def test_glm_meta_tool():
    """Test MetaPromptTool with GLM."""
    print("🔧 Testing MetaPromptTool with GLM-4.5...")
    
    # GLM configuration
    api_key = "e34be58db90244488b2f5ad00008698b.wlAPdq5wKIVuJiNA"
    base_url = "https://open.bigmodel.cn/api/paas/v4/"
    
    model_provider = ModelProvider(
        api_key=api_key,
        provider="glm",
        base_url=base_url
    )
    
    model_config = ModelConfig(
        model="glm-4.5",
        model_provider=model_provider,
        max_tokens=800,
        temperature=0.2,
        top_p=1.0,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=3,
    )
    
    llm_client = GLMClient(model_config)
    tool = MetaPromptTool(model_provider="glm", llm_client=llm_client)
    
    arguments = ToolCallArguments({
        "simple_prompt": "总结文档",
        "task_context": "学术研究"
    })
    
    try:
        result = await tool.execute(arguments)
        
        if result.error:
            print(f"❌ MetaTool Error: {result.error}")
            return False
        else:
            print(f"✅ MetaTool Success!")
            print(f"📝 Enhanced prompt: {result.output[:100]}...")
            return True
            
    except Exception as e:
        print(f"❌ MetaTool Exception: {e}")
        return False


async def test_glm_chunk_tool():
    """Test TextChunkTool with GLM."""
    print("\n🔧 Testing TextChunkTool with GLM-4.5...")
    
    # GLM configuration
    api_key = "e34be58db90244488b2f5ad00008698b.wlAPdq5wKIVuJiNA"
    base_url = "https://open.bigmodel.cn/api/paas/v4/"
    
    model_provider = ModelProvider(
        api_key=api_key,
        provider="glm",
        base_url=base_url
    )
    
    model_config = ModelConfig(
        model="glm-4.5",
        model_provider=model_provider,
        max_tokens=800,
        temperature=0.3,
        top_p=1.0,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=3,
    )
    
    llm_client = GLMClient(model_config)
    tool = TextChunkTool(model_provider="glm", llm_client=llm_client)
    
    # Create test file
    test_content = """人工智能发展历程简述：
    
1950年代：艾伦·图灵提出图灵测试。
1956年：达特茅斯会议确立人工智能概念。
1960-70年代：专家系统兴起。
1980年代：机器学习理论发展。
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        arguments = ToolCallArguments({
            "file_path": temp_file,
            "start_pos": 0,
            "end_pos": 100,
            "query": "主要内容是什么？",
            "enhanced_prompt": "请简要概述这段文字的主要内容。",
            "chunk_id": "test_chunk"
        })
        
        result = await tool.execute(arguments)
        
        if result.error:
            print(f"❌ ChunkTool Error: {result.error}")
            return False
        else:
            print(f"✅ ChunkTool Success!")
            print(f"📊 Analysis result: {result.output[:100]}...")
            return True
            
    except Exception as e:
        print(f"❌ ChunkTool Exception: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def main():
    """Main test function."""
    print("🚀 GLM Tools Integration Tests")
    print("=" * 50)
    
    results = []
    
    # Test MetaPromptTool
    meta_result = await test_glm_meta_tool()
    results.append(("MetaPromptTool", meta_result))
    
    # Test TextChunkTool
    chunk_result = await test_glm_chunk_tool()
    results.append(("TextChunkTool", chunk_result))
    
    # Summary
    print(f"\n{'=' * 50}")
    print("📊 Test Results Summary")
    print("=" * 50)
    
    for tool_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{tool_name:15}: {status}")
    
    total_passed = sum(r[1] for r in results)
    print(f"\n🎯 Overall: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("🎉 All GLM tools integration tests passed!")
    else:
        print("⚠️ Some tests failed. Please check the output above.")


if __name__ == "__main__":
    asyncio.run(main())