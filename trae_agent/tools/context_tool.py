# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Context file management tool for long context processing."""

import os
import tempfile
from typing import override

from trae_agent.tools.base import Tool, ToolExecResult, ToolParameter


class ContextFileManagerTool(Tool):
    """Tool for managing context files and chunking operations."""
    
    def __init__(self, model_provider: str | None = None, agent=None):
        super().__init__(model_provider)
        self.agent = agent  # Reference to main agent for state management
    
    @override
    def get_name(self) -> str:
        return "context_file_manager"
    
    @override
    def get_description(self) -> str:
        return "Save long context text to file and manage text chunking operations. Returns file path and file statistics."
    
    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Action to perform: 'save_context', 'get_chunk_info', 'plan_chunks'",
                enum=["save_context", "get_chunk_info", "plan_chunks"],
                required=True,
            ),
            ToolParameter(
                name="context",
                type="string", 
                description="Long context text to save (required for 'save_context')",
                required=False,
            ),
            ToolParameter(
                name="chunk_size",
                type="integer",
                description="Size of each chunk in characters (default: 8000)",
                required=False,
            ),
            ToolParameter(
                name="overlap",
                type="integer",
                description="Overlap between chunks in characters (default: 200)",
                required=False,
            ),
        ]
    
    @override
    async def execute(self, arguments) -> ToolExecResult:
        action = arguments.get("action")
        
        try:
            if action == "save_context":
                return await self._save_context(arguments)
            elif action == "get_chunk_info":
                return await self._get_chunk_info(arguments)
            elif action == "plan_chunks":
                return await self._plan_chunks(arguments)
            else:
                return ToolExecResult(
                    error=f"Unknown action: {action}",
                    error_code=1
                )
                
        except Exception as e:
            return ToolExecResult(
                error=f"Error in context_file_manager: {str(e)}",
                error_code=1
            )
    
    async def _save_context(self, arguments) -> ToolExecResult:
        """Save context text to a temporary file."""
        context = arguments.get("context")
        if not context:
            return ToolExecResult(
                error="Context text is required for save_context action",
                error_code=1
            )
        
        # Create temp file in agent's temp directory
        if self.agent and hasattr(self.agent, 'temp_dir'):
            temp_dir = self.agent.temp_dir
        else:
            temp_dir = tempfile.gettempdir()
        
        # Save context to file
        context_file = os.path.join(temp_dir, "context.txt")
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write(context)
        
        # Update agent state
        if self.agent:
            self.agent.context_file_path = context_file
        
        # Get file statistics
        file_size = len(context)
        lines = context.count('\n') + 1
        words = len(context.split())
        
        result = f"""Context saved successfully!
File path: {context_file}
Statistics:
- File size: {file_size:,} characters
- Lines: {lines:,}
- Words: {words:,}
- Estimated tokens: {file_size // 4:,} (rough estimate)

The context file is now ready for chunk-based processing."""
        
        return ToolExecResult(output=result)
    
    async def _get_chunk_info(self, arguments) -> ToolExecResult:
        """Get information about how the context would be chunked."""
        chunk_size = arguments.get("chunk_size", 8000)
        overlap = arguments.get("overlap", 200)
        
        # Get context file path
        if self.agent and hasattr(self.agent, 'context_file_path') and self.agent.context_file_path:
            context_file = self.agent.context_file_path
        else:
            return ToolExecResult(
                error="No context file found. Please save context first using 'save_context' action.",
                error_code=1
            )
        
        if not os.path.exists(context_file):
            return ToolExecResult(
                error=f"Context file not found: {context_file}",
                error_code=1
            )
        
        # Read file and calculate chunks
        with open(context_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        total_size = len(content)
        num_chunks = max(1, (total_size - overlap) // (chunk_size - overlap))
        
        result = f"""Chunk Analysis:
Context file: {context_file}
Total size: {total_size:,} characters
Chunk size: {chunk_size:,} characters
Overlap: {overlap:,} characters  
Estimated chunks: {num_chunks}

Chunk boundaries (start:end):"""

        # Calculate actual chunk boundaries
        for i in range(num_chunks):
            start = i * (chunk_size - overlap)
            end = min(start + chunk_size, total_size)
            result += f"\nChunk {i+1}: {start}:{end}"
            
            if end >= total_size:
                break
        
        return ToolExecResult(output=result)
    
    async def _plan_chunks(self, arguments) -> ToolExecResult:
        """Plan semantic chunking strategy."""
        chunk_size = arguments.get("chunk_size", 8000)
        overlap = arguments.get("overlap", 200)
        
        result = f"""Chunking Strategy Planning:

**Recommended Settings:**
- Chunk size: {chunk_size:,} characters (~{chunk_size//4:,} tokens)
- Overlap: {overlap:,} characters (to preserve context continuity)

**Semantic Considerations:**
1. Try to break at paragraph boundaries when possible
2. Maintain context for references and pronouns
3. Consider document structure (sections, chapters, etc.)
4. Preserve tables and code blocks within single chunks

**Processing Tips:**
1. Use meta_prompt_tool to create consistent prompts across chunks
2. Design prompts that work well with partial information  
3. Plan for result integration and consistency checking
4. Consider chunk dependencies for complex reasoning tasks

Use get_chunk_info action to see exact chunk boundaries before processing."""
        
        return ToolExecResult(output=result)