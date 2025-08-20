# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Enhanced chunk processing tool that combines meta-prompting and text processing internally."""

import os
from typing import override

from trae_agent.tools.base import Tool, ToolExecResult, ToolParameter
from trae_agent.tools.meta_tool import MetaPromptTool
from trae_agent.tools.chunk_tool import TextChunkTool
from trae_agent.utils.llm_clients.llm_client import LLMClient


class EnhancedChunkTool(Tool):
    """
    Enhanced chunk processing tool that combines meta-prompting and chunk processing internally.
    
    This tool solves the context pollution problem by:
    1. Internally calling MetaPromptTool to enhance prompts
    2. Using the enhanced prompt with TextChunkTool
    3. Only returning the final processed result to the main model
    4. Never exposing the long enhanced prompt to the main model's context
    """
    
    def __init__(self, model_provider: str | None = None, llm_client: LLMClient | None = None):
        super().__init__(model_provider)
        self.llm_client = llm_client
        # Internal tool instances - these won't be exposed to main model
        self.meta_tool = MetaPromptTool(model_provider, llm_client) if llm_client else None
        self.chunk_tool = TextChunkTool(model_provider, llm_client) if llm_client else None
    
    @override
    def get_name(self) -> str:
        return "enhanced_chunk_tool"
    
    @override
    def get_description(self) -> str:
        return "Process text chunk with automatically enhanced prompt. Combines meta-prompting and chunk processing internally to avoid context pollution."
    
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
                name="simple_prompt",
                type="string",
                description="Simple prompt that will be enhanced internally (e.g., 'Find family relationships', 'Extract key information')",
                required=True,
            ),
            ToolParameter(
                name="task_context",
                type="string",
                description="Context about the overall task and goals",
                required=False,
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
        """
        Execute enhanced chunk processing with internal prompt enhancement.
        
        Internal flow:
        1. Use MetaPromptTool to enhance the simple_prompt (internal only)
        2. Use TextChunkTool with enhanced prompt (internal only) 
        3. Clean and return only the final analysis result
        """
        if not self.llm_client:
            return ToolExecResult(
                error="LLM client not available for enhanced_chunk_tool",
                error_code=1
            )
        
        if not self.meta_tool or not self.chunk_tool:
            return ToolExecResult(
                error="Internal tools not properly initialized",
                error_code=1
            )
        
        # Extract arguments
        file_path = arguments.get("file_path")
        start_pos = arguments.get("start_pos")
        end_pos = arguments.get("end_pos")
        query = arguments.get("query")
        simple_prompt = arguments.get("simple_prompt")
        task_context = arguments.get("task_context", "")
        chunk_id = arguments.get("chunk_id", f"chunk_{start_pos}_{end_pos}")
        
        # Validate required arguments
        if not all([file_path, isinstance(start_pos, int), isinstance(end_pos, int), query, simple_prompt]):
            return ToolExecResult(
                error="Required parameters: file_path, start_pos, end_pos, query, simple_prompt",
                error_code=1
            )
        
        try:
            # Step 1: Internal meta-prompt enhancement (not exposed to main model)
            meta_args = {
                "simple_prompt": simple_prompt,
                "task_context": task_context
            }
            
            meta_result = await self.meta_tool.execute(meta_args)
            if meta_result.error is not None:
                return ToolExecResult(
                    error=f"Internal prompt enhancement failed: {meta_result.error}",
                    error_code=1
                )
            
            # Extract enhanced prompt from meta_tool result (internal processing)
            enhanced_prompt = self._extract_enhanced_prompt(meta_result.output)
            
            # Step 2: Internal chunk processing with enhanced prompt (not exposed to main model)
            chunk_args = {
                "file_path": file_path,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "query": query,
                "enhanced_prompt": enhanced_prompt,
                "chunk_id": chunk_id
            }
            
            chunk_result = await self.chunk_tool.execute(chunk_args)
            if chunk_result.error is not None:
                return ToolExecResult(
                    error=f"Internal chunk processing failed: {chunk_result.error}",
                    error_code=chunk_result.error_code
                )
            
            # Step 3: Clean output to remove internal processing details
            clean_output = self._clean_chunk_output(chunk_result.output, chunk_id, start_pos, end_pos)
            
            return ToolExecResult(output=clean_output)
            
        except Exception as e:
            return ToolExecResult(
                error=f"Enhanced chunk processing failed: {str(e)}",
                error_code=1
            )
    
    def _extract_enhanced_prompt(self, meta_output: str) -> str:
        """
        Extract the actual enhanced prompt from MetaPromptTool output.
        
        Args:
            meta_output: Output from MetaPromptTool
            
        Returns:
            The enhanced prompt text
        """
        try:
            lines = meta_output.split('\n')
            
            # Find the start of enhanced prompt
            prompt_start_idx = None
            for i, line in enumerate(lines):
                if 'Enhanced Prompt Generated Successfully:' in line:
                    prompt_start_idx = i + 2  # Skip title and empty line
                    break
            
            if prompt_start_idx is None:
                # Fallback: look for other indicators
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('Enhanced Prompt') and not line.startswith('---'):
                        prompt_start_idx = i
                        break
            
            if prompt_start_idx is None:
                return meta_output.strip()  # Return entire output if no structure found
            
            # Extract enhanced prompt until separator
            enhanced_lines = []
            for line in lines[prompt_start_idx:]:
                if line.strip() == '---':
                    break
                enhanced_lines.append(line)
            
            return '\n'.join(enhanced_lines).strip()
            
        except Exception:
            # Fallback to entire output if parsing fails
            return meta_output.strip()
    
    def _clean_chunk_output(self, chunk_output: str, chunk_id: str, start_pos: int, end_pos: int) -> str:
        """
        Clean chunk output to remove internal processing details and create a clean result.
        
        Args:
            chunk_output: Raw output from TextChunkTool
            chunk_id: Chunk identifier
            start_pos: Start position
            end_pos: End position
            
        Returns:
            Cleaned output with only essential information
        """
        try:
            lines = chunk_output.split('\n')
            
            # Find the analysis result section
            result_start_idx = None
            for i, line in enumerate(lines):
                if '**Analysis Result:**' in line:
                    result_start_idx = i + 1
                    break
            
            if result_start_idx is None:
                # Fallback: look for main content after chunk info
                for i, line in enumerate(lines):
                    if '**Chunk Processing Result**' in line:
                        # Skip metadata, find actual content
                        for j in range(i, len(lines)):
                            if not lines[j].startswith('**') and lines[j].strip():
                                result_start_idx = j
                                break
                        break
            
            if result_start_idx is None:
                # Last fallback: return content after first few metadata lines
                result_start_idx = min(5, len(lines))
            
            # Extract analysis content
            result_lines = []
            for line in lines[result_start_idx:]:
                # Stop at processing status or token usage info
                if line.startswith('---') or '**Processing Status:**' in line or '**Tokens Used:**' in line:
                    break
                result_lines.append(line)
            
            # Create clean output
            analysis_content = '\n'.join(result_lines).strip()
            
            # Format the final clean result
            clean_result = f"""**Chunk Analysis Result**

**Chunk:** {chunk_id} (position {start_pos}:{end_pos})
**Size:** {end_pos - start_pos:,} characters

**Analysis:**
{analysis_content}

**Status:** Processing complete"""
            
            return clean_result
            
        except Exception:
            # Fallback: return original output if cleaning fails
            return chunk_output