from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field


class MessageConfig(BaseModel):
    """Configuration for message content to be sent to Ollama."""
    system_message: str = Field(description="System instructions for the LLM")
    pre_prompt: Optional[str] = Field(default=None, description="Instructions to prepend to unstructured text")
    few_shot_json_path: Optional[str] = Field(default=None, description="Path to JSON file with few-shot examples")
    image_paths: Optional[List[str]] = Field(default=None, description="Paths to images for multimodal input")
    text: str = Field(description="Unstructured text for extraction")
    pydantic_model: Union[str, Type[BaseModel]] = Field(description="Pydantic model for extraction (class or import path)")


class ModelConfig(BaseModel):
    """Configuration for Ollama model parameters."""
    model_name: str = Field(description="Name of the Ollama model to use")
    temperature: float = Field(default=0.2, description="Temperature for generation (0.0-1.0)")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling value")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling value")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    logprobs: Optional[int] = Field(default=None, description="Number of log probabilities to generate")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences to end generation")
    
    def to_options_dict(self) -> Dict[str, Any]:
        """Convert to LLM options dictionary, excluding None values."""
        options = {
            "temperature": self.temperature
        }
        
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.seed is not None:
            options["seed"] = self.seed
        if self.stop is not None:
            options["stop"] = self.stop
            
        return options

class GenerationResult(BaseModel):
    """Result of a generation attempt."""
    execution_time: float = Field(description="Time taken for execution in seconds")
    generation_success: bool = Field(description="Whether generation was successful")
    parsing_success: bool = Field(description="Whether parsing was successful")
    raw_response: Optional[str] = Field(default=None, description="Raw text response from LLM")
    parsed_response: Optional[BaseModel] = Field(default=None, description="Structured data extracted from response")
    error: Optional[str] = Field(default=None, description="Error message if any")