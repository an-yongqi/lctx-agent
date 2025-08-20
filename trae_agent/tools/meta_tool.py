# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Meta prompt generation tool using LLM."""

from typing import override

from trae_agent.prompt.meta_tool_prompt import META_PROMPT_SYSTEM_PROMPT
from trae_agent.tools.base import Tool, ToolExecResult, ToolParameter
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.llm_basics import LLMMessage
from trae_agent.utils.llm_clients.llm_client import LLMClient


class MetaPromptTool(Tool):
    """Tool for converting simple prompts to high-quality prompts using LLM."""
    
    def __init__(self, model_provider: str | None = None, llm_client: LLMClient | None = None):
        super().__init__(model_provider)
        self.llm_client = llm_client
    
    @override
    def get_name(self) -> str:
        return "meta_prompt_tool"
    
    @override  
    def get_description(self) -> str:
        return "Transform simple, basic prompts into detailed, high-quality prompts optimized for better LLM performance and chunk processing."
    
    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="simple_prompt", 
                type="string",
                description="The simple, basic prompt to enhance",
                required=True,
            ),
            ToolParameter(
                name="task_context",
                type="string", 
                description="Context about the overall task and goals",
                required=False,
            ),
        ]
    
    @override
    async def execute(self, arguments) -> ToolExecResult:
        simple_prompt = arguments.get("simple_prompt")
        task_context = arguments.get("task_context", "")
        
        if not simple_prompt:
            return ToolExecResult(
                error="simple_prompt is required",
                error_code=1
            )
        
        if not self.llm_client:
            return ToolExecResult(
                error="LLM client not available for meta_prompt_tool",
                error_code=1
            )
        
        try:
            # Create user message
            user_message = f"""Transform this simple prompt into a high-quality, detailed prompt optimized for chunk-based processing:

Simple Prompt: {simple_prompt}"""

            if task_context:
                user_message += f"\n\nTask or Goal: {task_context}"

            # Call LLM
            messages = [
                LLMMessage(role="system", content=META_PROMPT_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_message)
            ]
            
            # Use current model configuration for meta-prompt generation
            if hasattr(self.llm_client, 'model_config'):
                base_model_config = self.llm_client.model_config
                model_config = ModelConfig(
                    model=base_model_config.model,
                    model_provider=base_model_config.model_provider,
                    max_tokens=1500,
                    temperature=0.7,
                    top_p=base_model_config.top_p,
                    top_k=base_model_config.top_k,
                    parallel_tool_calls=base_model_config.parallel_tool_calls,
                    max_retries=base_model_config.max_retries,
                )
            else:
                # Fallback configuration
                model_config = ModelConfig(
                    model="claude-3-haiku-20240307",
                    model_provider=self.llm_client.provider,
                    max_tokens=1500,
                    temperature=0.7,
                    top_p=1.0,
                    top_k=0,
                    parallel_tool_calls=False,
                    max_retries=3,
                )
            
            response = self.llm_client.chat(messages, model_config, [])
            
            if not response or not response.content:
                return ToolExecResult(
                    error="Failed to generate enhanced prompt",
                    error_code=1
                )
            
            enhanced_prompt = response.content.strip()
            
            result = f"""Enhanced Prompt Generated Successfully:

{enhanced_prompt}

---
Optimization Summary:
- Original prompt length: {len(simple_prompt)} characters  
- Enhanced prompt length: {len(enhanced_prompt)} characters
- Optimized for chunk-based processing: Yes
- CoT complexity reduction: Applied"""
            
            return ToolExecResult(output=result)
            
        except Exception as e:
            return ToolExecResult(
                error=f"Error generating meta prompt: {str(e)}",
                error_code=1
            )