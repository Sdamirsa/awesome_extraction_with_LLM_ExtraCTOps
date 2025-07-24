"""
Utility functions for handling message preparation for LLM input.
These functions help create structured messages containing text, images, and few-shot examples for LLM API calls.

Served by: Direct call

Path to venv, if required: "the_venvs/venv_utils_message"

Libraries to import:
- json
- os
- base64
- typing
- pandas
- pydantic
- pillow
"""

###################
####  imports  ####
###################
import subprocess
import sys
import json
import os
import base64
import asyncio
from typing import List, Dict, Any, Union, Optional

libraries = [
    "pandas",
    "pydantic",
    "pillow"
]

for lib in libraries:
    try:
        __import__(lib.replace("-", "_"))  # crude guess for pip-import mismatch
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

import pandas as pd
from pydantic import BaseModel

# Import models from generators_models.py where applicable
try:
    from internal_models import MessageConfig
except ImportError:
    the_logger.debug("Could not import MessageConfig from generators_models.py")

###################
####  logging  ####
###################
from utils import the_logger

the_logger.info("Initializing message utils for LLM input")

###################################
####  1. Message Models  #########
###################################
class Message(BaseModel):
    """Base message model for LLM interactions."""
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ImageContent(BaseModel):
    """Model for image content in multimodal messages."""
    type: str = "image_url"
    image_url: Dict[str, str]

###################################
####  2. Message Creation  #######
###################################
def create_system_message(system_prompt: str) -> Message:
    """
    Create a system message for LLM input.
    
    Args:
        system_prompt: The system prompt text
    
    Returns:
        Message object with role 'system'
    """
    the_logger.info("Creating system message")
    return Message(role="system", content=system_prompt)

def load_few_shot_examples(file_path: str) -> List[Message]:
    """
    Load few-shot examples from a JSON or Excel file.
    
    Args:
        file_path: Path to JSON or Excel file containing examples
    
    Returns:
        List of Message objects for few-shot learning
    """
    the_logger.info(f"Loading few-shot examples from {file_path}")
    
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            examples = json.load(f)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
        examples = df.to_dict('records')
    else:
        raise ValueError("Unsupported file format. Use JSON or Excel files.")
    
    messages = []
    for example in examples:
        if 'role' in example and 'content' in example:
            messages.append(Message(**example))
        else:
            the_logger.warning(f"Skipping invalid example: {example}")
    
    return messages

def create_user_message(pre_prompt: str, unstructured_text: str) -> Message:
    """
    Create a user message combining pre-prompt and unstructured text.
    
    Args:
        pre_prompt: Instructions or context to prepend
        unstructured_text: The main text content to process
    
    Returns:
        Message object with role 'user'
    """
    the_logger.info("Creating user message with pre-prompt and unstructured text")
    content = f"{pre_prompt}\n\n{unstructured_text}" if pre_prompt else unstructured_text
    return Message(role="user", content=content)

###################################
####  3. Image Handling  #########
###################################
def encode_image(image_path: str) -> str:
    """
    Encode an image to base64.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def encode_image_async(image_path: str) -> str:
    """
    Encode an image to base64 asynchronously.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Base64 encoded string of the image
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, encode_image, image_path)

def create_multimodal_message(text_content: str, image_paths: List[str], use_base64: bool = True) -> Message:
    """
    Create a multimodal message with text and images.
    
    Args:
        text_content: The text portion of the message
        image_paths: List of paths to image files
        use_base64: Whether to encode images as base64 (True) or provide paths (False)
    
    Returns:
        Message object with mixed content types
    """
    the_logger.info(f"Creating multimodal message with {len(image_paths)} images")
    
    content = [{"type": "text", "text": text_content}]
    
    for img_path in image_paths:
        if use_base64:
            try:
                base64_image = encode_image(img_path)
                image_content = ImageContent(
                    image_url={"url": f"data:image/jpeg;base64,{base64_image}"}
                )
                content.append(image_content.dict())
            except Exception as e:
                the_logger.error(f"Failed to encode image {img_path}: {str(e)}")
        else:
            content.append({"type": "image_url", "image_url": {"url": img_path}})
    
    return Message(role="user", content=content)

async def create_multimodal_message_async(text_content: str, image_paths: List[str], use_base64: bool = True) -> Message:
    """
    Create a multimodal message with text and images asynchronously.
    
    Args:
        text_content: The text portion of the message
        image_paths: List of paths to image files
        use_base64: Whether to encode images as base64 (True) or provide paths (False)
    
    Returns:
        Message object with mixed content types
    """
    the_logger.info(f"Creating multimodal message with {len(image_paths)} images (async)")
    
    content = [{"type": "text", "text": text_content}]
    
    if use_base64:
        encoding_tasks = []
        for img_path in image_paths:
            task = asyncio.create_task(encode_image_async(img_path))
            encoding_tasks.append((img_path, task))
        
        for img_path, task in encoding_tasks:
            try:
                base64_image = await task
                image_content = ImageContent(
                    image_url={"url": f"data:image/jpeg;base64,{base64_image}"}
                )
                content.append(image_content.dict())
            except Exception as e:
                the_logger.error(f"Failed to encode image {img_path}: {str(e)}")
    else:
        for img_path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": img_path}})
    
    return Message(role="user", content=content)

###################################
####  4. Main Functions  #########
###################################
def prepare_llm_messages(
    system_prompt: str,
    unstructured_text: str,
    pre_prompt: Optional[str] = None,
    few_shot_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    use_base64: bool = True
) -> List[Dict[str, Any]]:
    """
    Prepare a complete message object for LLM input.
    
    Args:
        system_prompt: The system instructions
        unstructured_text: The main text to process
        pre_prompt: Optional instructions to prepend to unstructured text
        few_shot_path: Optional path to few-shot examples file
        image_paths: Optional list of paths to images
        use_base64: Whether to encode images as base64
    
    Returns:
        List of message dictionaries ready for LLM API
    """
    the_logger.info("Preparing complete message for LLM")
    
    messages = [create_system_message(system_prompt).dict()]
    
    # Add few-shot examples if provided
    if few_shot_path:
        few_shot_messages = load_few_shot_examples(few_shot_path)
        messages.extend([m.dict() for m in few_shot_messages])
    
    # Add the main content (with or without images)
    if image_paths:
        user_message = create_multimodal_message(
            text_content=f"{pre_prompt}\n\n{unstructured_text}" if pre_prompt else unstructured_text,
            image_paths=image_paths,
            use_base64=use_base64
        )
    else:
        user_message = create_user_message(pre_prompt, unstructured_text)
    
    messages.append(user_message.dict())
    
    return messages

async def prepare_messages_Utils_async(
    message_config: Union[MessageConfig, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Prepare a complete message object for LLM input asynchronously using MessageConfig.
    
    Args:
        message_config: Configuration object containing all message parameters
        
    Returns:
        List of message dictionaries ready for LLM API
    """
    the_logger.info("Preparing complete message for LLM using MessageConfig (async)")
    
    # Convert dict to MessageConfig if needed
    if isinstance(message_config, dict):
        message_config = MessageConfig(**message_config)
    
    messages = [create_system_message(message_config.system_message).dict()]
    
    # Add few-shot examples if provided
    if message_config.few_shot_json_path:
        few_shot_messages = load_few_shot_examples(message_config.few_shot_json_path)
        messages.extend([m.dict() for m in few_shot_messages])
    
    # Add the main content (with or without images)
    if message_config.image_paths:
        user_message = await create_multimodal_message_async(
            text_content=f"{message_config.pre_prompt}\n\n{message_config.text}" if message_config.pre_prompt else message_config.text,
            image_paths=message_config.image_paths,
            use_base64=True  # Default to base64 for broader compatibility
        )
    else:
        user_message = create_user_message(message_config.pre_prompt, message_config.text)
    
    messages.append(user_message.dict())
    
    return messages

# Keep the old function for backward compatibility but make it use the new one
async def prepare_messages_Utils_async_legacy(
    system_prompt: str,
    unstructured_text: str,
    pre_prompt: Optional[str] = None,
    few_shot_path: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    use_base64: bool = True
) -> List[Dict[str, Any]]:
    """
    Legacy version that uses individual parameters instead of MessageConfig.
    
    Args:
        system_prompt: The system instructions
        unstructured_text: The main text to process
        pre_prompt: Optional instructions to prepend to unstructured text
        few_shot_path: Optional path to few-shot examples file
        image_paths: Optional list of paths to images
        use_base64: Whether to encode images as base64
    
    Returns:
        List of message dictionaries ready for LLM API
    """
    the_logger.warning("Using legacy prepare_messages_Utils_async function. Consider switching to the MessageConfig version.")
    
    config = MessageConfig(
        system_message=system_prompt,
        text=unstructured_text,
        pre_prompt=pre_prompt,
        few_shot_json_path=few_shot_path,
        image_paths=image_paths
    )
    
    return await prepare_messages_Utils_async(config)

###################################
####      Main function        ####
###################################
if __name__ == "__main__":
    the_logger.info("Message utility test run")
    # Test functionality
    test_system = "You are a helpful assistant that extracts information from text."
    test_text = "Patient John Doe, 45 years old, was admitted on January 15, 2023 with complaints of chest pain."
    
    messages = prepare_llm_messages(
        system_prompt=test_system,
        unstructured_text=test_text,
        pre_prompt="Extract patient information from the following text:"
    )
    
    print(json.dumps(messages, indent=2))
    
    # Test async functionality
    async def test_async():
        messages_async = await prepare_messages_Utils_async_legacy(
            system_prompt=test_system,
            unstructured_text=test_text,
            pre_prompt="Extract patient information from the following text:"
        )
        print("\nAsync version:")
        print(json.dumps(messages_async, indent=2))
    
    # Test the new function with MessageConfig
    async def test_config_async():
        config = MessageConfig(
            system_message="You are a helpful assistant that extracts information from text.",
            text="Patient John Doe, 45 years old, was admitted on January 15, 2023 with complaints of chest pain.",
            pre_prompt="Extract patient information from the following text:",
            pydantic_model="the_pydantics.example_schema.ExampleModel"
        )
        
        messages_async = await prepare_messages_Utils_async(config)
        print("\nAsync version with MessageConfig:")
        print(json.dumps(messages_async, indent=2))
    
    # Run the async tests
    asyncio.run(test_async())
    asyncio.run(test_config_async())

###################################
####  Example use in terminal  ####
###################################
"""
python utils_message.py
"""

###################################
####  Example use in notebook  ####
###################################
"""
# Updated example using MessageConfig
import asyncio
from generators.utils.utils_messages import prepare_messages_Utils_async
from generators.generators_models import MessageConfig

# Create a MessageConfig
config = MessageConfig(
    system_message="You are an AI assistant that extracts structured data from text.",
    text="Patient record text goes here...",
    pre_prompt="Extract the following fields into JSON...",
    pydantic_model="the_pydantics.example_schema.ExampleModel",
    image_paths=["path/to/image1.jpg"]  # Optional
)

async def process_with_config():
    # Use the new function with MessageConfig
    messages = await prepare_messages_Utils_async(config)
    return messages

# Run the async function
messages = asyncio.run(process_with_config())
"""