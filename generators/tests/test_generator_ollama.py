"""
Test cases for generator_ollama.py

This module contains unit tests for the Ollama generator functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Optional

# Import the modules to test
from generators.generator_ollama import generator_Ollama
from internal_models import MessageConfig, ModelConfig, GenerationResult
from pydantic import BaseModel, Field


class TestModel(BaseModel):
    """Simple test model for testing."""
    name: str = Field(description="A name field")
    value: int = Field(description="A numeric value")


class TestGeneratorOllama:
    """Test class for Ollama generator functionality."""
    
    @pytest.fixture
    def sample_message_config(self):
        """Sample MessageConfig for testing."""
        return MessageConfig(
            system_message="Extract data from text",
            pre_prompt="Return as JSON",
            text="Test data with name John and value 42",
            pydantic_model=TestModel
        )
    
    @pytest.fixture
    def sample_model_config(self):
        """Sample ModelConfig for testing.""" 
        return ModelConfig(
            model_name="llama3",
            temperature=0.1,
            max_tokens=1000
        )
    
    @pytest.mark.asyncio
    async def test_generator_ollama_success(self, sample_message_config, sample_model_config):
        """Test successful generation and parsing."""
        # Mock the Ollama client response
        mock_response = MagicMock()
        mock_response.message.content = '{"name": "John", "value": 42}'
        
        with patch('generators.generator_ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            result = await generator_Ollama(sample_message_config, sample_model_config)
            
            assert result.generation_success is True
            assert result.parsing_success is True
            assert result.parsed_response is not None
            assert result.parsed_response.name == "John"
            assert result.parsed_response.value == 42
    
    @pytest.mark.asyncio
    async def test_generator_ollama_parsing_failure(self, sample_message_config, sample_model_config):
        """Test generation success but parsing failure."""
        # Mock the Ollama client response with invalid JSON
        mock_response = MagicMock()
        mock_response.message.content = 'Invalid JSON response'
        
        with patch('generators.generator_ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            result = await generator_Ollama(sample_message_config, sample_model_config)
            
            assert result.generation_success is True
            assert result.parsing_success is False
            assert result.parsed_response is None
            assert result.raw_response == 'Invalid JSON response'
    
    @pytest.mark.asyncio
    async def test_generator_ollama_exception(self, sample_message_config, sample_model_config):
        """Test exception handling."""
        with patch('generators.generator_ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.side_effect = Exception("Connection error")
            mock_client_class.return_value = mock_client
            
            result = await generator_Ollama(sample_message_config, sample_model_config)
            
            assert result.generation_success is False
            assert result.parsing_success is False
            assert result.error is not None
            assert "Connection error" in result.error


if __name__ == "__main__":
    pytest.main([__file__])
