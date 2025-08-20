# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""GLM (ChatGLM) client wrapper with tool integrations"""

import openai

from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class GLMProvider(ProviderConfig):
    """GLM provider configuration."""

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """Create OpenAI client with GLM base URL."""
        return openai.OpenAI(base_url=base_url, api_key=api_key)

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "GLM"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "glm"

    def get_extra_headers(self) -> dict[str, str]:
        """Get any extra headers needed for the API call."""
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        # GLM-4 series models support tool calling
        tool_calling_models = [
            "glm-4-plus",
            "glm-4-0520", 
            "glm-4",
            "glm-4-air",
            "glm-4-airx",
        ]
        return model_name in tool_calling_models


class GLMClient(OpenAICompatibleClient):
    """GLM client implementation using OpenAI-compatible interface."""

    def __init__(self, model_config: ModelConfig):
        """Initialize GLM client."""
        super().__init__(model_config, GLMProvider())
        # Add provider and model_config attributes for compatibility with tools
        self.provider = model_config.model_provider
        self.model_config = model_config

    def get_available_models(self) -> list[str]:
        """Get list of available GLM models."""
        return [
            "glm-4-plus",
            "glm-4-0520", 
            "glm-4",
            "glm-4-air",
            "glm-4-airx",
            "glm-4-flash",
            "glm-3-turbo",
        ]