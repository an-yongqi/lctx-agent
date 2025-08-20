# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""LongContextAgent for handling long text context processing tasks."""

import os
import tempfile
from typing import override

from trae_agent.agent.base_agent import BaseAgent
from trae_agent.agent.agent_basics import AgentExecution
from trae_agent.tools import tools_registry
from trae_agent.tools.base import Tool, ToolExecutor
from trae_agent.utils.config import AgentConfig
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
from trae_agent.prompt.long_context_agent_prompt import LONG_CONTEXT_AGENT_SYSTEM_PROMPT


LongContextAgentToolNames = [
    "context_file_manager", 
    "enhanced_chunk_tool",
    "str_replace_based_edit_tool",
    "sequentialthinking",
    "json_edit_tool",
    "task_done",
    "bash",
]


class LongContextAgent(BaseAgent):
    """Agent specialized for processing long context input problems."""
    
    def __init__(self, agent_config: AgentConfig):
        """Initialize LongContextAgent."""
        self.temp_dir: str = tempfile.mkdtemp(prefix="long_context_")
        self.context_file_path: str | None = None
        super().__init__(agent_config)
    
    @override
    def new_task(
        self,
        task: str,
        extra_args: dict[str, str] | None = None,
        tool_names: list[str] | None = None,
    ):
        """Create a new long context processing task."""
        self._task = task
        
        if tool_names is None:
            tool_names = LongContextAgentToolNames
            
        # Import and register our custom tools
        from trae_agent.tools.context_tool import ContextFileManagerTool
        from trae_agent.tools.enhanced_chunk_tool import EnhancedChunkTool
        
        # Update tools registry with our custom tools
        tools_registry.update({
            "context_file_manager": ContextFileManagerTool,
            "enhanced_chunk_tool": EnhancedChunkTool,
        })
        
        provider = self._model_config.model_provider.provider
        self._tools: list[Tool] = []
        
        for tool_name in tool_names:
            if tool_name == "context_file_manager":
                # Context tool needs agent reference for state management
                tool = tools_registry[tool_name](model_provider=provider, agent=self)
            elif tool_name == "enhanced_chunk_tool":
                # Enhanced chunk tool needs LLM client for internal processing
                tool = tools_registry[tool_name](model_provider=provider, llm_client=self._llm_client)
            else:
                # Standard tools
                tool = tools_registry[tool_name](model_provider=provider)
            self._tools.append(tool)
        
        self._tool_caller = ToolExecutor(self._tools)
        
        # Setup initial messages
        self._initial_messages = [
            LLMMessage(role="system", content=self.get_system_prompt())
        ]
        
        # Process input parameters
        context = extra_args.get("context", "") if extra_args else ""
        query = extra_args.get("query", "") if extra_args else ""
        
        # Add user task message  
        user_message = f"""**Task:** {task}

**Query to Answer:**
{query}

**Context Information:**
- Context length: {len(context):,} characters (~{len(context)//4:,} tokens estimated)
- This context is too long for direct processing and needs to be handled in chunks

**Your Workflow:**
1. Use context_file_manager to save the context to a file
2. Use sequentialthinking to analyze the query and plan your decomposition strategy
3. Plan how to break down the problem to reduce CoT complexity per chunk  
4. Process chunks systematically using enhanced_chunk_tool (automatically enhances prompts internally)
5. Integrate results and ensure complete coverage of the query

Please start by saving the context to a file, then proceed with your analysis and planning."""
        
        self._initial_messages.append(
            LLMMessage(role="user", content=user_message)
        )
        
        # If we have context, include it for the agent to process via tools
        if context:
            context_message = f"""**Full Context Text to Process:**

{context}

**Note:** This context text should be saved using the context_file_manager tool before processing."""
            self._initial_messages.append(
                LLMMessage(role="user", content=context_message)
            )
        
        # Setup trajectory recording
        if self._trajectory_recorder:
            self._trajectory_recorder.start_recording(
                task=task,
                provider=self._llm_client.provider.value,
                model=self._model_config.model,
                max_steps=self._max_steps,
            )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for LongContextAgent."""
        return LONG_CONTEXT_AGENT_SYSTEM_PROMPT
    
    @override
    async def execute_task(self) -> AgentExecution:
        """Execute the long context processing task."""
        execution = await super().execute_task()
        
        # Cleanup temporary files
        self._cleanup_temp_files()
        
        if self._trajectory_recorder:
            self._trajectory_recorder.finalize_recording(
                success=execution.success, 
                final_result=execution.final_result
            )
        
        return execution
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @override
    def llm_indicates_task_completed(self, llm_response: LLMResponse) -> bool:
        """Check if the LLM indicates that the task is completed."""
        if llm_response.tool_calls is None:
            return False
        return any(tool_call.name == "task_done" for tool_call in llm_response.tool_calls)
    
    @override
    async def cleanup_mcp_clients(self) -> None:
        """Clean up MCP clients."""
        pass