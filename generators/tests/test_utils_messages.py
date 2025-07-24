"""
Test cases for utils_messages.py

This module contains unit tests for message utility functions.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, mock_open
from typing import List, Dict, Any

# Import the modules to test
from generators.utils.utils_messages import (
    create_system_message,
    create_user_message,
    prepare_llm_messages,
    prepare_messages_Utils_async,
    load_few_shot_examples,
    Message
)
from internal_models import MessageConfig
from pydantic import BaseModel


class TestModel(BaseModel):
    """Simple test model for testing."""
    name: str
    value: int


class TestUtilsMessages:
    """Test class for message utility functions."""
    
    def test_create_system_message(self):
        """Test system message creation."""
        system_prompt = "You are a helpful assistant"
        message = create_system_message(system_prompt)
        
        assert message.role == "system"
        assert message.content == system_prompt
    
    def test_create_user_message(self):
        """Test user message creation."""
        pre_prompt = "Extract data from:"
        text = "Sample text data"
        message = create_user_message(pre_prompt, text)
        
        assert message.role == "user"
        assert pre_prompt in message.content
        assert text in message.content
    
    def test_create_user_message_no_preprompt(self):
        """Test user message creation without pre-prompt."""
        text = "Sample text data"
        message = create_user_message(None, text)
        
        assert message.role == "user"
        assert message.content == text
    
    def test_prepare_llm_messages(self):
        """Test basic message preparation."""
        system_prompt = "You are a helpful assistant"
        text = "Extract data from this text"
        pre_prompt = "Return as JSON"
        
        messages = prepare_llm_messages(
            system_prompt=system_prompt,
            unstructured_text=text,
            pre_prompt=pre_prompt
        )
        
        assert len(messages) == 2  # System + User
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
        assert messages[1]["role"] == "user"
        assert pre_prompt in messages[1]["content"]
        assert text in messages[1]["content"]
    
    @pytest.mark.asyncio
    async def test_prepare_messages_utils_async(self):
        """Test async message preparation with MessageConfig."""
        message_config = MessageConfig(
            system_message="Extract data",
            pre_prompt="Return as JSON",
            text="Sample text with name John and value 42",
            pydantic_model=TestModel
        )
        
        messages = await prepare_messages_Utils_async(message_config)
        
        assert len(messages) >= 2  # At least system + user
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == message_config.system_message
        assert messages[-1]["role"] == "user"
    
    @pytest.mark.asyncio
    async def test_prepare_messages_utils_async_with_dict(self):
        """Test async message preparation with dict input."""
        config_dict = {
            "system_message": "Extract data",
            "pre_prompt": "Return as JSON", 
            "text": "Sample text",
            "pydantic_model": TestModel
        }
        
        messages = await prepare_messages_Utils_async(config_dict)
        
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
    
    def test_load_few_shot_examples_json(self):
        """Test loading few-shot examples from JSON."""
        sample_examples = [
            {"role": "user", "content": "Extract from: Test text 1"},
            {"role": "assistant", "content": '{"name": "Test", "value": 1}'}
        ]
        
        with patch("builtins.open", mock_open(read_data=json.dumps(sample_examples))):
            messages = load_few_shot_examples("test.json")
            
            assert len(messages) == 2
            assert all(isinstance(msg, Message) for msg in messages)
            assert messages[0].role == "user"
            assert messages[1].role == "assistant"


if __name__ == "__main__":
    pytest.main([__file__])
