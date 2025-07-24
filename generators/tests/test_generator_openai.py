"""
Test cases for generator_openai.py

This module contains unit tests for the OpenAI generator functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Optional

# Import the modules to test
from generators.generator_openai import generator_OpenAI
from internal_models import MessageConfig, ModelConfig, GenerationResult
from pydantic import BaseModel, Field


class TestModel(BaseModel):
    """Simple test model for testing."""
    name: str = Field(description="A name field")
    value: int = Field(description="A numeric value")


class TestGeneratorOpenAI:
    """Test class for OpenAI generator functionality."""
    
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
            model_name="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000
        )
    
    @pytest.mark.asyncio
    async def test_generator_openai_success(self, sample_message_config, sample_model_config):
        """Test successful generation and parsing."""
        # Mock the OpenAI client response
        mock_choice = MagicMock()
        mock_choice.message.content = '{"name": "John", "value": 42}'
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        with patch('generators.generator_openai.AsyncOpenAI') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            # Mock the API key check
            with patch('generators.generator_openai.API_KEY', 'test-key'):
                result = await generator_OpenAI(sample_message_config, sample_model_config)
                
                assert result.generation_success is True
                assert result.parsing_success is True
                assert result.parsed_response is not None
                assert result.parsed_response.name == "John"
                assert result.parsed_response.value == 42
    
    @pytest.mark.asyncio
    async def test_generator_openai_parsing_failure(self, sample_message_config, sample_model_config):
        """Test generation success but parsing failure."""
        # Mock the OpenAI client response with invalid JSON
        mock_choice = MagicMock()
        mock_choice.message.content = 'Invalid JSON response'
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        with patch('generators.generator_openai.AsyncOpenAI') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            with patch('generators.generator_openai.API_KEY', 'test-key'):
                result = await generator_OpenAI(sample_message_config, sample_model_config)
                
                assert result.generation_success is True
                assert result.parsing_success is False
                assert result.parsed_response is None
                assert result.raw_response == 'Invalid JSON response'
    
    @pytest.mark.asyncio
    async def test_generator_openai_no_api_key(self, sample_message_config, sample_model_config):
        """Test API key validation."""
        with patch('generators.generator_openai.API_KEY', ''):
            result = await generator_OpenAI(sample_message_config, sample_model_config)
            
            assert result.generation_success is False
            assert result.parsing_success is False
            assert result.error is not None
            assert "API key not configured" in result.error
    
    @pytest.mark.asyncio
    async def test_generator_openai_exception(self, sample_message_config, sample_model_config):
        """Test exception handling."""
        with patch('generators.generator_openai.AsyncOpenAI') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.side_effect = Exception("API error")
            mock_client_class.return_value = mock_client
            
            with patch('generators.generator_openai.API_KEY', 'test-key'):
                result = await generator_OpenAI(sample_message_config, sample_model_config)
                
                assert result.generation_success is False
                assert result.parsing_success is False
                assert result.error is not None
                assert "API error" in result.error


if __name__ == "__main__":
    pytest.main([__file__])
