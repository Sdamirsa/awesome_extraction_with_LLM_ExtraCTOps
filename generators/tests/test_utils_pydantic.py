"""
Test cases for utils_pydantic.py

This module contains unit tests for Pydantic utility functions.
"""

import pytest
import asyncio
import json
from typing import Optional, List
from pydantic import BaseModel, Field

# Import the modules to test
from generators.utils.utils_pydantic import (
    advanced_parser_Utils_async,
    parse_json_to_pydantic_async,
    extract_structured_data_async,
    pydantic_to_schema,
    generate_extraction_prompt,
    extract_json_from_markdown
)


class TestModel(BaseModel):
    """Simple test model for testing."""
    name: str = Field(description="A person's name")
    age: int = Field(description="A person's age")
    email: Optional[str] = Field(None, description="Email address")


class TestUtilsPydantic:
    """Test class for Pydantic utility functions."""
    
    @pytest.mark.asyncio
    async def test_parse_json_to_pydantic_async_success(self):
        """Test successful JSON to Pydantic parsing."""
        json_data = {"name": "John", "age": 30, "email": "john@example.com"}
        
        instance, error = await parse_json_to_pydantic_async(json_data, TestModel)
        
        assert instance is not None
        assert error is None
        assert instance.name == "John"
        assert instance.age == 30
        assert instance.email == "john@example.com"
    
    @pytest.mark.asyncio
    async def test_parse_json_to_pydantic_async_string_input(self):
        """Test JSON string to Pydantic parsing."""
        json_string = '{"name": "Jane", "age": 25}'
        
        instance, error = await parse_json_to_pydantic_async(json_string, TestModel)
        
        assert instance is not None
        assert error is None
        assert instance.name == "Jane"
        assert instance.age == 25
    
    @pytest.mark.asyncio
    async def test_parse_json_to_pydantic_async_validation_error(self):
        """Test validation error handling."""
        invalid_data = {"name": "John", "age": "not_a_number"}
        
        instance, error = await parse_json_to_pydantic_async(invalid_data, TestModel)
        
        assert instance is None
        assert error is not None
        assert "Validation error" in error
    
    @pytest.mark.asyncio
    async def test_parse_json_to_pydantic_async_partial_success(self):
        """Test partial model creation when some fields are missing."""
        partial_data = {"name": "Bob"}  # Missing required age field
        
        instance, error = await parse_json_to_pydantic_async(partial_data, TestModel)
        
        # This should fail since age is required
        assert instance is None
        assert error is not None
    
    @pytest.mark.asyncio
    async def test_extract_structured_data_async_json(self):
        """Test extraction from clean JSON."""
        content = '{"name": "Alice", "age": 28}'
        
        result = await extract_structured_data_async(content)
        
        assert result is not None
        assert result["name"] == "Alice"
        assert result["age"] == 28
    
    @pytest.mark.asyncio
    async def test_extract_structured_data_async_markdown(self):
        """Test extraction from markdown code block."""
        content = '''Here's the data:
        ```json
        {"name": "Bob", "age": 35}
        ```
        '''
        
        result = await extract_structured_data_async(content)
        
        assert result is not None
        assert result["name"] == "Bob"
        assert result["age"] == 35
    
    @pytest.mark.asyncio
    async def test_extract_structured_data_async_no_json(self):
        """Test extraction failure when no JSON is found."""
        content = "This is just plain text with no JSON data."
        
        result = await extract_structured_data_async(content)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_advanced_parser_utils_async_success(self):
        """Test full pipeline with successful parsing."""
        content = '{"name": "Charlie", "age": 40}'
        
        instance, original = await advanced_parser_Utils_async(content, TestModel)
        
        assert instance is not None
        assert original is None
        assert instance.name == "Charlie"
        assert instance.age == 40
    
    @pytest.mark.asyncio
    async def test_advanced_parser_utils_async_existing_model(self):
        """Test pipeline when content is already the correct model type."""
        existing_model = TestModel(name="David", age=45)
        
        instance, original = await advanced_parser_Utils_async(existing_model, TestModel)
        
        assert instance is existing_model
        assert original is None
    
    @pytest.mark.asyncio
    async def test_advanced_parser_utils_async_dict_input(self):
        """Test pipeline with dictionary input."""
        content = {"name": "Eve", "age": 32}
        
        instance, original = await advanced_parser_Utils_async(content, TestModel)
        
        assert instance is not None
        assert original is None
        assert instance.name == "Eve"
        assert instance.age == 32
    
    @pytest.mark.asyncio
    async def test_advanced_parser_utils_async_failure(self):
        """Test pipeline failure handling."""
        content = "This cannot be parsed into structured data"
        
        instance, original = await advanced_parser_Utils_async(content, TestModel)
        
        assert instance is None
        assert original == content
    
    @pytest.mark.asyncio 
    async def test_pydantic_to_schema(self):
        """Test Pydantic model to JSON schema conversion."""
        schema = await pydantic_to_schema(TestModel)
        
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"
    
    @pytest.mark.asyncio
    async def test_generate_extraction_prompt(self):
        """Test extraction prompt generation."""
        prompt = await generate_extraction_prompt(TestModel)
        
        assert "TestModel" in prompt
        assert "name" in prompt
        assert "age" in prompt
        assert "JSON" in prompt
    
    def test_extract_json_from_markdown(self):
        """Test JSON extraction from markdown code blocks."""
        markdown_text = '''
        Here's some text.
        
        ```json
        {"name": "Test", "value": 123}
        ```
        
        More text here.
        '''
        
        result = extract_json_from_markdown(markdown_text)
        
        assert result is not None
        assert result["name"] == "Test"
        assert result["value"] == 123
    
    def test_extract_json_from_markdown_no_json(self):
        """Test markdown extraction when no JSON block exists."""
        markdown_text = "Just plain markdown text with no code blocks."
        
        result = extract_json_from_markdown(markdown_text)
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
