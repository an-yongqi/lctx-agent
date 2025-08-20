#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
GLM (ChatGLM) integration test script.
This script tests the GLM client with real API calls.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trae_agent.utils.config import ModelConfig, ModelProvider
from trae_agent.utils.llm_clients.llm_client import LLMClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage


async def test_glm_basic_chat():
    """Test basic chat functionality with GLM."""
    
    print("🧪 Testing GLM Basic Chat")
    print("=" * 50)
    
    # Configure GLM client
    model_provider = ModelProvider(
        api_key="e34be58db90244488b2f5ad00008698b.wlAPdq5wKIVuJiNA",
        provider="glm",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    
    model_config = ModelConfig(
        model="glm-4.5",  # Use fastest model for testing
        model_provider=model_provider,
        max_tokens=500,
        temperature=0.7,
        top_p=1.0,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=3,
    )
    
    # Create client
    client = LLMClient(model_config)
    
    # Test simple conversation
    messages = [
        LLMMessage(role="user", content="你好！请简单介绍一下ChatGLM模型。")
    ]
    
    print("💬 发送消息: 你好！请简单介绍一下ChatGLM模型。")
    print("⏳ 等待响应...")
    
    try:
        response = client.chat(messages, model_config)
        
        print("\n✅ 响应成功!")
        print(f"🤖 GLM回复: {response.content}")
        
        if response.usage:
            print(f"📊 Token使用情况:")
            print(f"   输入tokens: {response.usage.input_tokens}")
            print(f"   输出tokens: {response.usage.output_tokens}")
        
        print(f"🎯 使用模型: {response.model}")
        
        return True
        
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")
        return False


async def test_glm_with_tools():
    """Test GLM with tool calling (if supported)."""
    
    print("\n🔧 Testing GLM Tool Calling")
    print("=" * 50)
    
    model_provider = ModelProvider(
        api_key="e34be58db90244488b2f5ad00008698b.wlAPdq5wKIVuJiNA",
        provider="glm",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    
    model_config = ModelConfig(
        model="glm-4.5",  # Use more advanced model for tool calling
        model_provider=model_provider,
        max_tokens=1000,
        temperature=0.3,
        top_p=1.0,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=3,
    )
    
    client = LLMClient(model_config)
    
    # Simple tool-like conversation
    messages = [
        LLMMessage(role="user", content="请帮我计算 15 * 23 + 45 的结果，并解释计算步骤。")
    ]
    
    print("💬 发送消息: 请帮我计算 15 * 23 + 45 的结果，并解释计算步骤。")
    print("⏳ 等待响应...")
    
    try:
        response = client.chat(messages, model_config)
        
        print("\n✅ 响应成功!")
        print(f"🤖 GLM回复: {response.content}")
        
        if response.usage:
            print(f"📊 Token使用情况:")
            print(f"   输入tokens: {response.usage.input_tokens}")
            print(f"   输出tokens: {response.usage.output_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")
        return False


async def test_long_context_agent_with_glm():
    """Test our LongContextAgent with GLM."""
    
    print("\n🚀 Testing LongContextAgent with GLM")
    print("=" * 50)
    
    from trae_agent.agent.long_context_agent import LongContextAgent
    from trae_agent.utils.config import AgentConfig
    
    # Configure GLM for long context agent
    model_provider = ModelProvider(
        api_key="e34be58db90244488b2f5ad00008698b.wlAPdq5wKIVuJiNA",
        provider="glm",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    
    model_config = ModelConfig(
        model="glm-4.5",
        model_provider=model_provider,
        max_tokens=2000,
        temperature=0.3,
        top_p=1.0,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=3,
    )
    
    agent_config = AgentConfig(
        model=model_config,
        max_steps=50,
        tools=[],  # Will be set by LongContextAgent
        allow_mcp_servers=[],
        mcp_servers_config={},
    )
    
    # Create agent
    agent = LongContextAgent(agent_config)
    
    # Simple long context test
    long_context = """
    人工智能的发展历程可以分为几个重要阶段：

    1950年代：艾伦·图灵提出图灵测试，标志着人工智能理论基础的建立。
    1956年：达特茅斯会议正式提出"人工智能"这一概念。
    1960-70年代：专家系统兴起，如MYCIN医疗诊断系统。
    1980年代：机器学习理论逐渐发展，神经网络重获关注。
    1990年代：互联网普及推动了数据积累和算法改进。
    2000年代：深度学习突破，图像识别和自然语言处理取得重大进展。
    2010年代：大语言模型出现，如GPT、BERT等。
    2020年代：ChatGPT等对话式AI广泛应用，AGI成为新目标。

    人工智能技术在各行业的应用包括：
    - 医疗：辅助诊断、药物发现、个性化治疗
    - 金融：风险评估、算法交易、客户服务  
    - 教育：个性化学习、智能辅导
    - 交通：自动驾驶、智能导航
    - 制造业：质量控制、预测性维护
    - 娱乐：内容推荐、游戏AI

    未来发展趋势包括更强的推理能力、多模态融合、边缘计算部署等。
    """ * 3  # 重复几次使内容更长
    
    query = "请总结人工智能发展的主要阶段，并分析其在不同行业的应用情况。"
    
    print(f"📄 长文本长度: {len(long_context):,} 字符")
    print(f"❓ 查询: {query}")
    print("⏳ LongContextAgent 处理中...")
    
    try:
        task = "分析人工智能发展历程并总结应用领域"
        extra_args = {
            "context": long_context,
            "query": query
        }
        
        agent.new_task(task, extra_args)
        execution = await agent.execute_task()
        
        print(f"\n✅ 处理完成!")
        print(f"🎯 成功状态: {execution.success}")
        print(f"⏱️ 执行时间: {execution.execution_time:.2f} 秒")
        print(f"🔢 执行步数: {len(execution.steps)}")
        print(f"📝 最终结果:\n{execution.final_result}")
        
        return execution.success
        
    except Exception as e:
        print(f"❌ LongContextAgent 执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    
    print("🔬 GLM (ChatGLM) Integration Tests")
    print("=" * 60)
    print("🔑 使用API Key: e34be58db90244488b2f5ad00008698b.wlAPdq5wKIVuJiNA")
    print("🌐 Base URL: https://open.bigmodel.cn/api/paas/v4/")
    print()
    
    results = []
    
    # Test 1: Basic chat
    result1 = await test_glm_basic_chat()
    results.append(("Basic Chat", result1))
    
    # Test 2: Advanced conversation
    result2 = await test_glm_with_tools()
    results.append(("Advanced Conversation", result2))
    
    # Test 3: Long Context Agent
    result3 = await test_long_context_agent_with_glm()
    results.append(("LongContextAgent", result3))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20}: {status}")
    
    total_passed = sum(results[i][1] for i in range(len(results)))
    print(f"\n🎯 总体结果: {total_passed}/{len(results)} 测试通过")
    
    if total_passed == len(results):
        print("🎉 所有测试都通过！GLM 集成成功！")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    asyncio.run(main())