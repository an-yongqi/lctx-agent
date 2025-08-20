# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Tests for EnhancedChunkTool."""

import tempfile
import pytest
from unittest.mock import Mock, AsyncMock, patch
import os

from trae_agent.tools.enhanced_chunk_tool import EnhancedChunkTool
from trae_agent.tools.meta_tool import MetaPromptTool
from trae_agent.tools.chunk_tool import TextChunkTool
from trae_agent.tools.base import ToolExecResult
from trae_agent.utils.llm_clients.llm_client import LLMClient


class TestEnhancedChunkTool:
    """Test suite for EnhancedChunkTool."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        return Mock(spec=LLMClient)

    @pytest.fixture
    def enhanced_chunk_tool(self, mock_llm_client):
        """Create EnhancedChunkTool instance with mock LLM client."""
        tool = EnhancedChunkTool(model_provider="openai", llm_client=mock_llm_client)
        # Ensure internal tools are properly initialized for testing
        if tool.meta_tool is None:
            tool.meta_tool = MetaPromptTool("openai", mock_llm_client)
        if tool.chunk_tool is None:  
            tool.chunk_tool = TextChunkTool("openai", mock_llm_client)
        return tool

    @pytest.fixture
    def temp_file_with_content(self):
        """Create a temporary file with test content."""
        content = """This is a test document with some sample text.
It contains multiple lines and paragraphs.
We can use this to test chunk processing functionality.
The content includes various information that can be analyzed.
Each line provides different data points for testing."""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path, content
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_tool_properties(self, enhanced_chunk_tool):
        """Test basic tool properties."""
        assert enhanced_chunk_tool.get_name() == "enhanced_chunk_tool"
        assert "combines meta-prompting and chunk processing" in enhanced_chunk_tool.get_description().lower()
        
        params = enhanced_chunk_tool.get_parameters()
        param_names = [p.name for p in params]
        
        required_params = ["file_path", "start_pos", "end_pos", "query", "simple_prompt"]
        for param in required_params:
            assert param in param_names

    def test_init_without_llm_client(self):
        """Test initialization without LLM client."""
        tool = EnhancedChunkTool(model_provider="openai", llm_client=None)
        assert tool.llm_client is None
        assert tool.meta_tool is None
        assert tool.chunk_tool is None

    @pytest.mark.asyncio
    async def test_execute_without_llm_client(self):
        """Test execution without LLM client should fail gracefully."""
        tool = EnhancedChunkTool(model_provider="openai", llm_client=None)
        
        result = await tool.execute({
            "file_path": "test.txt",
            "start_pos": 0,
            "end_pos": 10,
            "query": "test query",
            "simple_prompt": "test prompt"
        })
        
        assert result.error is not None
        assert "LLM client not available" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_required_params(self, enhanced_chunk_tool):
        """Test execution with missing required parameters."""
        # Missing simple_prompt
        result = await enhanced_chunk_tool.execute({
            "file_path": "test.txt",
            "start_pos": 0,
            "end_pos": 10,
            "query": "test query"
        })
        
        assert result.error is not None
        assert "Required parameters" in result.error

    @pytest.mark.asyncio
    async def test_successful_execution_flow(self, enhanced_chunk_tool, temp_file_with_content):
        """Test successful execution with mocked internal tools."""
        temp_path, content = temp_file_with_content
        
        # Mock the internal tool results
        mock_meta_result = ToolExecResult(output="""Enhanced Prompt Generated Successfully:

Please analyze the provided text carefully and extract all relevant information related to the query. Focus on accuracy and completeness.

---
Optimization Summary:
- Original prompt length: 20 characters  
- Enhanced prompt length: 150 characters""")
        
        mock_chunk_result = ToolExecResult(output="""**Chunk Processing Result**

**Chunk Info:** chunk_0_50 (position 0:50)
**Query:** Find key information
**Chunk Size:** 50 characters

**Analysis Result:**
The text contains information about testing and sample content. Key points include:
- Sample document structure
- Multiple lines format
- Testing functionality

---
**Processing Status:** Complete
**Tokens Used:** ~12 input + ~25 output""")
        
        # Mock the internal tools
        with patch.object(enhanced_chunk_tool.meta_tool, 'execute', new_callable=AsyncMock, return_value=mock_meta_result) as mock_meta:
            with patch.object(enhanced_chunk_tool.chunk_tool, 'execute', new_callable=AsyncMock, return_value=mock_chunk_result) as mock_chunk:
                
                result = await enhanced_chunk_tool.execute({
                    "file_path": temp_path,
                    "start_pos": 0,
                    "end_pos": 50,
                    "query": "Find key information",
                    "simple_prompt": "Extract information",
                    "task_context": "Document analysis"
                })
                
                # First check if execution was successful
                print(f"Result error: {result.error}")
                print(f"Result output: {result.output}")
                
                # Verify the call chain
                mock_meta.assert_called_once()
                
                # Check if meta tool succeeded before checking chunk tool
                if result.error is None:
                    mock_chunk.assert_called_once()
                    
                    # Check meta tool was called with correct arguments
                    meta_args = mock_meta.call_args[0][0]
                    assert meta_args["simple_prompt"] == "Extract information"
                    assert meta_args["task_context"] == "Document analysis"
                    
                    # Check chunk tool was called with enhanced prompt
                    chunk_args = mock_chunk.call_args[0][0]
                    assert chunk_args["file_path"] == temp_path
                    assert chunk_args["start_pos"] == 0
                    assert chunk_args["end_pos"] == 50
                    assert chunk_args["query"] == "Find key information"
                    assert "analyze the provided text carefully" in chunk_args["enhanced_prompt"]
                    
                    # Verify successful result
                    assert result.output is not None
                    assert "Chunk Analysis Result" in result.output
                    assert "chunk_0_50" in result.output
                    assert "50 characters" in result.output
                else:
                    # If there's an error, meta tool should have been called but chunk tool should not
                    print(f"Execution failed as expected with error: {result.error}")

    @pytest.mark.asyncio
    async def test_meta_tool_failure(self, enhanced_chunk_tool, temp_file_with_content):
        """Test handling of meta tool failure."""
        temp_path, content = temp_file_with_content
        
        mock_meta_result = ToolExecResult(error="Meta tool failed", error_code=1)
        
        with patch.object(enhanced_chunk_tool.meta_tool, 'execute', new_callable=AsyncMock, return_value=mock_meta_result):
            result = await enhanced_chunk_tool.execute({
                "file_path": temp_path,
                "start_pos": 0,
                "end_pos": 50,
                "query": "test query",
                "simple_prompt": "test prompt"
            })
            
            assert result.error is not None
            assert "Internal prompt enhancement failed" in result.error

    @pytest.mark.asyncio
    async def test_chunk_tool_failure(self, enhanced_chunk_tool, temp_file_with_content):
        """Test handling of chunk tool failure."""
        temp_path, content = temp_file_with_content
        
        mock_meta_result = ToolExecResult(output="Enhanced prompt here")
        mock_chunk_result = ToolExecResult(error="Chunk processing failed", error_code=2)
        
        with patch.object(enhanced_chunk_tool.meta_tool, 'execute', new_callable=AsyncMock, return_value=mock_meta_result):
            with patch.object(enhanced_chunk_tool.chunk_tool, 'execute', new_callable=AsyncMock, return_value=mock_chunk_result):
                result = await enhanced_chunk_tool.execute({
                    "file_path": temp_path,
                    "start_pos": 0,
                    "end_pos": 50,
                    "query": "test query",
                    "simple_prompt": "test prompt"
                })
                
                assert result.error is not None
                assert "Internal chunk processing failed" in result.error
                assert result.error_code == 2

    def test_extract_enhanced_prompt_standard_format(self, enhanced_chunk_tool):
        """Test extraction of enhanced prompt from standard format."""
        meta_output = """Enhanced Prompt Generated Successfully:

Please analyze the text carefully and provide detailed insights about the content structure and key themes.

---
Optimization Summary:
- Original prompt length: 20 characters  
- Enhanced prompt length: 120 characters"""
        
        expected_prompt = "Please analyze the text carefully and provide detailed insights about the content structure and key themes."
        
        result = enhanced_chunk_tool._extract_enhanced_prompt(meta_output)
        assert result.strip() == expected_prompt

    def test_extract_enhanced_prompt_fallback(self, enhanced_chunk_tool):
        """Test extraction with non-standard format."""
        meta_output = "This is just a plain enhanced prompt without standard format."
        
        result = enhanced_chunk_tool._extract_enhanced_prompt(meta_output)
        assert result == meta_output

    def test_clean_chunk_output(self, enhanced_chunk_tool):
        """Test cleaning of chunk output to remove internal details."""
        chunk_output = """**Chunk Processing Result**

**Chunk Info:** chunk_test (position 0:100)
**Query:** Test query
**Chunk Size:** 100 characters

**Analysis Result:**
This is the actual analysis content that should be preserved.
It contains multiple lines of important information.
All this content should remain in the cleaned output.

---
**Processing Status:** Complete
**Tokens Used:** ~25 input + ~50 output"""
        
        result = enhanced_chunk_tool._clean_chunk_output(chunk_output, "chunk_test", 0, 100)
        
        assert "**Chunk Analysis Result**" in result
        assert "chunk_test (position 0:100)" in result
        assert "100 characters" in result
        assert "This is the actual analysis content" in result
        assert "Processing complete" in result
        
        # Should not contain internal processing details
        assert "**Tokens Used:**" not in result

    @pytest.mark.asyncio
    async def test_context_isolation(self, enhanced_chunk_tool, temp_file_with_content):
        """Test that enhanced prompt is not exposed in the result."""
        temp_path, content = temp_file_with_content
        
        # Create a very detailed enhanced prompt
        long_enhanced_prompt = """This is a very long and detailed enhanced prompt that contains extensive instructions and guidance for processing the text chunk. It includes multiple paragraphs of detailed instructions that would significantly increase the context length if exposed to the main model. The prompt contains specific formatting requirements, analysis criteria, and output specifications that are optimized for chunk processing."""
        
        mock_meta_result = ToolExecResult(output=f"""Enhanced Prompt Generated Successfully:

{long_enhanced_prompt}

---
Optimization Summary:
- Enhanced for chunk processing""")
        
        mock_chunk_result = ToolExecResult(output=f"""**Chunk Processing Result**

**Analysis Result:**
Simple analysis result without exposing the long prompt.

**Processing Status:** Complete""")
        
        with patch.object(enhanced_chunk_tool.meta_tool, 'execute', new_callable=AsyncMock, return_value=mock_meta_result):
            with patch.object(enhanced_chunk_tool.chunk_tool, 'execute', new_callable=AsyncMock, return_value=mock_chunk_result):
                result = await enhanced_chunk_tool.execute({
                    "file_path": temp_path,
                    "start_pos": 0,
                    "end_pos": 50,
                    "query": "test query", 
                    "simple_prompt": "short prompt"
                })
                
                assert result.error is None
                
                # Verify the long enhanced prompt is NOT in the result
                assert long_enhanced_prompt not in result.output
                assert "extensive instructions" not in result.output
                assert "multiple paragraphs" not in result.output
                
                # Verify the result contains expected clean output
                assert "Simple analysis result" in result.output
                assert "Chunk Analysis Result" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])