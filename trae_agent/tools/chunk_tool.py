# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Text chunk processing tool using LLM."""

import os
from typing import override

from trae_agent.tools.base import Tool, ToolExecResult, ToolParameter
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.llm_basics import LLMMessage
from trae_agent.utils.llm_clients.llm_client import LLMClient


class TextChunkTool(Tool):
    """Tool for processing specific text chunks using LLM."""
    
    def __init__(self, model_provider: str | None = None, llm_client: LLMClient | None = None):
        super().__init__(model_provider)
        self.llm_client = llm_client
    
    @override
    def get_name(self) -> str:
        return "text_chunk_tool"
    
    @override
    def get_description(self) -> str:
        return "Process a specific chunk of text file[start:end] based on query and enhanced prompt using LLM."
    
    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the context file",
                required=True,
            ),
            ToolParameter(
                name="start_pos",
                type="integer", 
                description="Start character position in the file",
                required=True,
            ),
            ToolParameter(
                name="end_pos",
                type="integer",
                description="End character position in the file", 
                required=True,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="The specific query/question to answer based on this chunk",
                required=True,
            ),
            ToolParameter(
                name="enhanced_prompt",
                type="string",
                description="Enhanced prompt (from meta_prompt_tool) for processing this chunk",
                required=True,
            ),
            ToolParameter(
                name="chunk_id",
                type="string",
                description="Identifier for this chunk (e.g., 'chunk_1', 'section_A')",
                required=False,
            ),
        ]
    
    @override
    async def execute(self, arguments) -> ToolExecResult:
        file_path = arguments.get("file_path")
        start_pos = arguments.get("start_pos")
        end_pos = arguments.get("end_pos")
        query = arguments.get("query")
        enhanced_prompt = arguments.get("enhanced_prompt")
        chunk_id = arguments.get("chunk_id", f"chunk_{start_pos}_{end_pos}")
        
        # Validate inputs
        if not all([file_path, isinstance(start_pos, int), isinstance(end_pos, int), query, enhanced_prompt]):
            return ToolExecResult(
                error="Required parameters: file_path, start_pos, end_pos, query, enhanced_prompt",
                error_code=1
            )
        
        if start_pos < 0 or end_pos <= start_pos:
            return ToolExecResult(
                error="Invalid position range: start_pos must be >= 0 and end_pos must be > start_pos",
                error_code=1
            )
        
        if not self.llm_client:
            return ToolExecResult(
                error="LLM client not available for text_chunk_tool",
                error_code=1
            )
        
        try:
            # Read the specified chunk from file
            if not os.path.exists(file_path):
                return ToolExecResult(
                    error=f"File not found: {file_path}",
                    error_code=1
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                f.seek(start_pos)
                chunk_text = f.read(end_pos - start_pos)
            
            if not chunk_text.strip():
                return ToolExecResult(
                    error=f"No content found in range {start_pos}:{end_pos}",
                    error_code=1
                )
            
            # Create system prompt for chunk processing
            system_prompt = """You are an expert text analysis agent specialized in processing text chunks as part of a larger long-context analysis task.

**Key Guidelines:**
- Focus exclusively on the provided text chunk
- Answer based solely on information available in this chunk
- If information is not available in this chunk, clearly state this limitation
- Maintain objectivity and accuracy in your analysis
- Follow the enhanced prompt instructions precisely
- Be concise but thorough in your response
- Consider that this chunk is part of a larger document
- Format your response for easy integration with other chunk results

**Important:** This is a partial view of a larger text. Focus on extracting and analyzing information present in this specific chunk while acknowledging the partial nature of the context."""

            # Create user message with chunk and query
            user_message = f"""**Enhanced Instructions:** {enhanced_prompt}

**Query to Answer:** {query}

**Text Chunk Information:**
- Chunk ID: {chunk_id}
- Position: {start_pos}:{end_pos}
- Size: {len(chunk_text)} characters

**Text Chunk Content:**
{chunk_text}

---

Please process this text chunk according to the enhanced instructions and answer the query. Remember that this is only a partial view of a larger document."""
            
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_message)
            ]
            
            # Use current model configuration for chunk processing
            if hasattr(self.llm_client, 'model_config'):
                base_model_config = self.llm_client.model_config
                model_config = ModelConfig(
                    model=base_model_config.model,
                    model_provider=base_model_config.model_provider,
                    max_tokens=2500,
                    temperature=0.3,
                    top_p=base_model_config.top_p,
                    top_k=base_model_config.top_k,
                    parallel_tool_calls=base_model_config.parallel_tool_calls,
                    max_retries=base_model_config.max_retries,
                )
            else:
                # Fallback configuration
                model_config = ModelConfig(
                    model="claude-3-sonnet-20240229",
                    model_provider=self.llm_client.provider,
                    max_tokens=2500,
                    temperature=0.3,
                    top_p=1.0,
                    top_k=0,
                    parallel_tool_calls=False,
                    max_retries=3,
                )
            
            response = self.llm_client.chat(messages, model_config, [])
            
            if not response or not response.content:
                return ToolExecResult(
                    error="Failed to process text chunk",
                    error_code=1
                )
            
            # Format the result with metadata
            result = f"""**Chunk Processing Result**

**Chunk Info:** {chunk_id} (position {start_pos}:{end_pos})
**Query:** {query}
**Chunk Size:** {len(chunk_text):,} characters

**Analysis Result:**
{response.content.strip()}

---
**Processing Status:** Complete
**Tokens Used:** ~{len(chunk_text) // 4} input + ~{len(response.content) // 4} output"""
            
            return ToolExecResult(output=result)
            
        except Exception as e:
            return ToolExecResult(
                error=f"Error processing text chunk: {str(e)}",
                error_code=1
            )