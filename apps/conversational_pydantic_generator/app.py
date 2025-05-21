"""
Conversational Pydantic Generator App

# Description
    [A streamlit application that uses multiple LLM agents to help generate Pydantic models through conversation.]

    - Arguments:
        - User Input (text): Conversational description of the data model needs
        - LLM Provider Selection: OpenAI, Azure OpenAI, or OpenAI-compatible endpoints
        - API Configuration: API keys and endpoints based on the selected provider

    - Environment Arguments:
        - None required for UI, API keys needed for LLM functionality

    - Returns
        - Pydantic Schema: Structured definition of classes and fields based on conversation

# Features:
    - Conversational interface to describe your data model needs
    - Schema Generator agent that creates structured Pydantic classes from conversation
    - Advisor agent that provides recommendations to improve the schema
    - Structure Optimization agent to analyze and improve the model hierarchy
    - Final verification agent to ensure the schema meets best practices
    - Cost tracking and reporting for API usage
    - Support for OpenAI, Azure OpenAI, and OpenAI-compatible endpoints

# Engine:
    - Serve (utils/data/main-function/sub-function): main-function
    - Served by (API/Direct/Subprocess): Direct
    - Path to venv, if require separate venv: the_venvs/venv_streamlit
    - libraries to import: [streamlit,pydantic,openai,python-dotenv]

# Identity
    - Last Status (future/in-progress/complete/published): published
    - Publish Date: 2025-05-22
    - Version: 0.1
    - License: MIT
    - Author: Seyed Amir Ahmad Safavi-Naini, sdamirsa@gmail.com
    - Source: https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps

# Changelog
    - 2025-05-22: version 0.1

# To-do:
    - [] Improve error handling for unreliable API connections
    - [] Add support for more LLM providers
    - [] Implement caching to reduce API costs
"""

import streamlit as st
import json
import os
import re
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from enum import Enum
import time
from datetime import datetime
from openai import OpenAI, AzureOpenAI
import tempfile
import base64
import traceback

# --- App Configuration ---
st.set_page_config(
    page_title="Pydantic Model Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Models for Structured Output ---
class PydanticField(BaseModel):
    """A field in a Pydantic model"""
    name: str = Field(..., description="Name of the field")
    field_type: str = Field(..., description="Data type of the field (e.g., str, int, float, bool, List[str], etc.)")
    parent_class: str = Field(..., description="Name of the Pydantic class this field belongs to")
    description: str = Field(..., description="Description of what this field represents")
    is_optional: bool = Field(True, description="Whether this field is optional")
    enum_values: Optional[List[Dict[str, str]]] = Field(None, description="If this is an Enum field, list of enum values and their descriptions")
    default_value: Optional[Any] = Field(None, description="Default value for this field, if any")

class EnumDefinition(BaseModel):
    """Definition of an Enum class"""
    name: str = Field(..., description="Name of the Enum class")
    base_type: str = Field("str", description="Base type for the Enum (usually 'str')")
    description: str = Field(..., description="Description of what this enum represents")
    values: List[Dict[str, str]] = Field(..., description="List of enum values and their descriptions")

class PydanticClass(BaseModel):
    """A Pydantic class definition"""
    name: str = Field(..., description="Name of the class")
    description: str = Field(..., description="Description of what this class represents")
    is_base_model: bool = Field(True, description="Whether this class inherits from BaseModel")
    parent_class: Optional[str] = Field(None, description="Parent class if this inherits from another custom class")
    is_enum: bool = Field(False, description="Whether this is an Enum class")

class SchemaDefinition(BaseModel):
    """Complete schema definition with all classes and fields"""
    main_class: str = Field(..., description="Name of the main (top-level) Pydantic class")
    classes: List[PydanticClass] = Field(..., description="List of all Pydantic classes in the schema")
    fields: List[PydanticField] = Field(..., description="List of all fields across all classes")
    enums: List[EnumDefinition] = Field([], description="List of all Enum definitions")

class AdvisorRecommendation(BaseModel):
    """Recommendation from the advisor agent"""
    type: Literal["add_field", "modify_field", "remove_field", "add_class", "modify_class", "restructure", "general", "good_practice"] 
    description: str = Field(..., description="Detailed description of the recommendation")
    priority: Literal["high", "medium", "low"] = Field(..., description="Priority of this recommendation")
    reason: str = Field(..., description="Reasoning behind this recommendation")
    example: Optional[str] = Field(None, description="Example code or explanation if applicable")

class AdvisorAnalysis(BaseModel):
    """Analysis and recommendations from the advisor agent"""
    recommendations: List[AdvisorRecommendation] = Field(..., description="List of recommendations for improving the schema")
    missing_information: List[str] = Field([], description="List of information that appears to be missing from the conversation")
    general_feedback: str = Field(..., description="General feedback on the schema")

class FinalVerification(BaseModel):
    """Final verification of the schema"""
    approved: bool = Field(..., description="Whether the schema is approved")
    reasons: Optional[List[str]] = Field(None, description="Reasons for approval or rejection")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions for improvement if not approved")

class StructureOptimization(BaseModel):
    """Results from the structure optimization agent"""
    optimized_schema: Optional[SchemaDefinition] = Field(None, description="The optimized schema, if improvements were found")
    changes_made: List[str] = Field([], description="List of changes made to improve the structure")
    reasoning: str = Field(..., description="Reasoning behind the structural optimizations")
    is_optimized: bool = Field(..., description="Whether the schema was optimized or kept the same")

# --- Session State Initialization ---
def init_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "schema" not in st.session_state:
        st.session_state.schema = None
    
    if "advisor_analysis" not in st.session_state:
        st.session_state.advisor_analysis = None
    
    if "verification_result" not in st.session_state:
        st.session_state.verification_result = None
    
    if "structure_optimization" not in st.session_state:
        st.session_state.structure_optimization = None
    
    if "final_code" not in st.session_state:
        st.session_state.final_code = None
    
    if "just_reset" not in st.session_state:
        st.session_state.just_reset = False
    
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "OpenAI"
    
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    
    if "azure_api_key" not in st.session_state:
        st.session_state.azure_api_key = ""
    
    if "azure_endpoint" not in st.session_state:
        st.session_state.azure_endpoint = ""
        
    if "azure_api_version" not in st.session_state:
        st.session_state.azure_api_version = "2023-07-01-preview"
    
    if "base_url" not in st.session_state:
        st.session_state.base_url = ""
    
    if "llm_model_conversation" not in st.session_state:
        st.session_state.llm_model_conversation = "gpt-4.1-mini"
    
    if "llm_model_schema" not in st.session_state:
        st.session_state.llm_model_schema = "gpt-4.1-mini"
    
    if "llm_model_advisor" not in st.session_state:
        st.session_state.llm_model_advisor = "gpt-4.1-mini"
    
    if "llm_model_verifier" not in st.session_state:
        st.session_state.llm_model_verifier = "gpt-4.1-mini"
    
    if "llm_model_optimizer" not in st.session_state:
        st.session_state.llm_model_optimizer = "gpt-4.1-mini"
        
    # Cost tracking
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    
    if "cost_breakdown" not in st.session_state:
        st.session_state.cost_breakdown = {
            "conversation": 0.0,
            "schema_generation": 0.0,
            "advisor": 0.0,
            "verification": 0.0,
            "structure_optimization": 0.0
        }
        
    if "api_call_count" not in st.session_state:
        st.session_state.api_call_count = {
            "conversation": 0,
            "schema_generation": 0,
            "advisor": 0,
            "verification": 0,
            "structure_optimization": 0
        }
        
    # Pricing settings
    if "pricing_settings" not in st.session_state:
        # Default pricing per 1 million tokens as of May 2025
        st.session_state.pricing_settings = {
            "gpt-4o": {"prompt": 2.5, "completion": 10.0},
            "gpt-4.1": {"prompt": 2.0, "completion": 8.0},
            "gpt-4.1-mini": {"prompt": 0.4, "completion": 1.6},
            "gpt-4.1-nano": {"prompt": 0.1, "completion": 0.4},
            "o3-mini": {"prompt": 1.1, "completion": 4.4},
            "o4-mini": {"prompt": 1.1, "completion": 4.4},
            "default": {"prompt": 1000.0, "completion": 2000.0},
            "o3": {"prompt": 10.0, "completion": 40.0},
        }

def reset_app_state():
    """Reset the app state for conversation and schema-related data only, preserving API settings"""
    conversation_keys = [
        "chat_history", 
        "schema", 
        "advisor_analysis", 
        "verification_result", 
        "structure_optimization",
        "final_code"
    ]
    
    # Reset conversation-related keys
    for key in conversation_keys:
        if key in st.session_state:
            st.session_state[key] = [] if key == "chat_history" else None
    
    # Reset cost tracking
    st.session_state.total_cost = 0.0
    st.session_state.cost_breakdown = {
        "conversation": 0.0,
        "schema_generation": 0.0,
        "advisor": 0.0,
        "verification": 0.0,
        "structure_optimization": 0.0
    }
    st.session_state.api_call_count = {
        "conversation": 0,
        "schema_generation": 0,
        "advisor": 0,
        "verification": 0,
        "structure_optimization": 0
    }
    
    # Set a flag to display a notification that the conversation was reset
    st.session_state.just_reset = True

# --- LLM Client Setup ---
def get_llm_client():
    """Get the appropriate LLM client based on the selected provider"""
    if st.session_state.llm_provider == "OpenAI":
        return OpenAI(api_key=st.session_state.openai_api_key)
    elif st.session_state.llm_provider == "Azure OpenAI":
        return AzureOpenAI(
            api_key=st.session_state.azure_api_key,
            azure_endpoint=st.session_state.azure_endpoint,
            api_version=st.session_state.azure_api_version
        )
    else:  # OpenAI-Compatible
        return OpenAI(
            api_key=st.session_state.openai_api_key,
            base_url=st.session_state.base_url
        )

# --- LLM API Calls ---
def conversation_agent_call(message):
    """Handle the conversation with the user"""
    try:
        client = get_llm_client()
        messages = [
            {
                "role": "system",
                "content": """You are an expert assistant helping the user create a Pydantic data model through conversation. 
                Your goal is to gather detailed information about the data structure they want to model.
                
                Ask specific questions to understand:
                1. The domain of the data (e.g., healthcare, finance, e-commerce)
                2. The main entities and their relationships
                3. Required fields and their data types
                4. Validation requirements
                5. Nested structures and hierarchies
                
                Be conversational but also guide the user toward providing complete information needed for a comprehensive data model.
                Don't worry about defining the Pydantic model yourself - another agent will do that by analyzing this conversation.
                
                Keep your responses focused on gathering necessary information and clarifying the user's requirements. Ask follow-up
                questions for ambiguous points. Remember, you're gathering requirements, not building the schema directly.
                """
            }
        ]
        
        # Add chat history
        for msg in st.session_state.chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        model = st.session_state.llm_model_conversation
        if st.session_state.llm_provider == "Azure OpenAI":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
        
        # Track token usage and cost
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            # Calculate cost
            cost_info = calculate_api_cost(model, prompt_tokens, completion_tokens)
            current_cost = cost_info["total_cost"]
            
            # Update cost tracking
            st.session_state.total_cost += current_cost
            st.session_state.cost_breakdown["conversation"] += current_cost
            st.session_state.api_call_count["conversation"] += 1
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error in conversation agent: {str(e)}")
        return "I encountered an error processing your request. Please check your API settings or try again later."

def schema_generator_call():
    """Generate the Pydantic schema from the conversation"""
    try:
        client = get_llm_client()
        messages = [
            {
                "role": "system",
                "content": """You are an expert Pydantic schema generator. Your task is to analyze a conversation about data modeling
                and generate a structured definition of Pydantic classes and fields.
                
                Review the entire conversation carefully and identify:
                1. The main entities that should be modeled as Pydantic classes
                2. Fields for each class with appropriate types
                3. Relationships between classes (nested models, etc.)
                4. Validation requirements
                5. Enum classes where appropriate
                
                Produce a structured output following these guidelines:
                - Use clear, descriptive class and field names
                - Follow Python naming conventions (snake_case for fields, PascalCase for classes)
                - Make all fields Optional with None as default unless explicitly required
                - Add helpful Field descriptions for all fields
                - Create Enum classes for fields with fixed sets of values
                - Structure nested data appropriately with sub-models
                
                Your output must be valid JSON that conforms to the SchemaDefinition model.
                """
            }
        ]
        
        # Add chat history
        for msg in st.session_state.chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        model = st.session_state.llm_model_schema
        
        # Add the reference example
        reference_example = r"""
        Here's a reference example of a Pydantic schema for medical echo reports similar to what you should generate:
        
        ```python
        # ENUMS FOR QUALITATIVE DATA
        class DilationSeverity(str, Enum):
            APLASTIC = "Aplastic"
            HYPOPLASTIC = "Hypoplastic"
            SMALL = "Small"
            NORMAL = "Normal"
            MILD = "Mild"
            MODERATE = "Moderate"
            SEVERE = "Severe"
            
        # SUBMODELS FOR NUMERIC FIELDS
        class NumericValue(BaseModel):
            numeric: Optional[float] = Field(None, description="Numeric measurement value.")
            unit: Optional[str] = Field(None, description="Unit of measurement (e.g., mm, cm, m/s, mmHg, etc.).")
            
        # DOMAIN-SPECIFIC SUBMODELS
        class RA_info(BaseModel):
            \"""Right Atrium\"""
            RA_dilation: Optional[DilationSeverity] = Field(None, description="Right atrial (RA) dilation/size severity.")
            
        class LA_info(BaseModel):
            \"""Left Atrium\"""
            LA_dilation: Optional[DilationSeverity] = Field(None, description="Left atrial (LA) dilation/size severity.")
            LA_volume_indexed: Optional[NumericValue] = Field(None, description="Left atrial (LA) volume measurement.")
            
        class Atria(BaseModel):
            \"""Combined submodel for Atria\"""
            RA: Optional[RA_info] = None
            LA: Optional[LA_info] = None
            
        # TOP-LEVEL MODEL
        class EchoReport(BaseModel):
            \"""Main Pydantic model capturing the full echo report.\"""
            atria: Optional[Atria] = None
            # ... other fields
        ```
        
        Analyze the conversation and generate a similar schema definition, but structure your response as a valid JSON SchemaDefinition.
        """
        
        messages.append({"role": "user", "content": reference_example})
        
        # Add explicit request for structured output
        structured_output_request = """
        Generate the Pydantic schema based on the conversation above as a valid JSON object that follows this exact structure:
        
        {
          "main_class": "Name of the main class",
          "classes": [
            {
              "name": "ClassName",
              "description": "Description of the class",
              "is_base_model": true,
              "parent_class": null,
              "is_enum": false
            },
            ...
          ],
          "fields": [
            {
              "name": "field_name",
              "field_type": "str",
              "parent_class": "ClassName",
              "description": "Description of the field",
              "is_optional": true,
              "enum_values": null,
              "default_value": null
            },
            ...
          ],
          "enums": [
            {
              "name": "EnumName",
              "base_type": "str",
              "description": "Description of the enum",
              "values": [
                { "name": "VALUE1", "description": "Description of VALUE1" },
                { "name": "VALUE2", "description": "Description of VALUE2" }
              ]
            },
            ...
          ]
        }
        
        IMPORTANT: For fields with enum values, make sure that 'enum_values' is either null or an array of objects, where each object has both 'name' and 'description' keys. For example:
        
        "enum_values": [
          {"name": "EXCELLENT", "description": "Excellent quality"},
          {"name": "GOOD", "description": "Good quality"}, 
          {"name": "FAIR", "description": "Fair quality"},
          {"name": "POOR", "description": "Poor quality"}
        ]
        
        DO NOT provide simple strings like ["EXCELLENT", "GOOD", "FAIR", "POOR"] for enum_values.
        
        Similarly, for enums, ensure that each value in the 'values' array is an object with 'name' and 'description' properties.
        
        Ensure your response is a validly formatted JSON object and nothing else.
        """
        
        messages.append({"role": "user", "content": structured_output_request})
        
        if st.session_state.llm_provider == "Azure OpenAI":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
        
        # Track token usage and cost
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            # Calculate cost
            cost_info = calculate_api_cost(model, prompt_tokens, completion_tokens)
            current_cost = cost_info["total_cost"]
            
            # Update cost tracking
            st.session_state.total_cost += current_cost
            st.session_state.cost_breakdown["schema_generation"] += current_cost
            st.session_state.api_call_count["schema_generation"] += 1
        
        # Parse the response
        response_text = response.choices[0].message.content
        schema_json = json.loads(response_text)
        
        # Log schema structure for debugging
        debug_info = debug_schema_structure(schema_json)
        print(f"Schema structure debug info:\n{debug_info}")
        
        # Normalize the schema JSON to ensure it's compatible with our models
        normalized_schema = normalize_schema_json(schema_json)
        
        # Create SchemaDefinition
        try:
            return SchemaDefinition(**normalized_schema)
        except Exception as validation_error:
            st.error(f"Schema validation error: {str(validation_error)}")
            st.error("The LLM provided a schema that doesn't match our expected format. Try regenerating the schema.")
            
            # Include debug info in the UI when there's a validation error
            with st.expander("Debug Information (for developers)"):
                st.code(debug_info, language="text")
                st.code(json.dumps(normalized_schema, indent=2), language="json")
            
            return None
        
    except Exception as e:
        st.error(f"Error in schema generator: {str(e)}")
        # Print more detailed error information for debugging
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

def advisor_agent_call(schema):
    """Get recommendations from the advisor agent"""
    try:
        client = get_llm_client()
        
        # Convert schema to JSON string
        schema_json = schema.model_dump_json(indent=2)
        
        # Create message to include the schema and chat history
        messages = [
            {
                "role": "system",
                "content": """You are an expert Pydantic schema advisor. Your task is to analyze a Pydantic schema definition and the conversation that led to it, 
                then provide thoughtful recommendations to improve the schema.
                
                Consider these aspects:
                1. Completeness - Are there missing fields or classes that were mentioned in the conversation?
                2. Structure - Is the hierarchy of classes appropriate? Should any nested models be created or modified?
                3. Types - Are the field types appropriate for the data described?
                4. Validation - Are there validation requirements mentioned that should be implemented?
                5. Best practices - Does the schema follow Pydantic best practices?
                
                Provide actionable recommendations with examples where appropriate. Be specific about what should change and why.
                
                Your output must be valid JSON conforming to the AdvisorAnalysis model, with recommendations categorized by type and priority.
                """
            }
        ]
        
        # Add chat history
        chat_history_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_history])
        messages.append({"role": "user", "content": f"CONVERSATION HISTORY:\n\n{chat_history_text}"})
        
        # Add the schema
        messages.append({"role": "user", "content": f"CURRENT SCHEMA DEFINITION:\n\n{schema_json}"})
        
        # Add request for structured output
        structured_output_request = """
        Analyze the schema and conversation above, then provide your recommendations in the following JSON structure:
        
        {
          "recommendations": [
            {
              "type": "add_field",
              "description": "Add a field 'xyz' to model 'ABC'",
              "priority": "high",
              "reason": "This field was mentioned repeatedly in the conversation but is missing from the schema",
              "example": "xyz: Optional[str] = Field(None, description='Description of xyz')"
            },
            ...
          ],
          "missing_information": [
            "Information about validation requirements for field X",
            ...
          ],
          "general_feedback": "Overall assessment of the schema quality and suggested next steps."
        }
        
        IMPORTANT: For each recommendation, the "type" field MUST be one of these exact values:
        - "add_field" - for suggesting a new field to add to an existing class
        - "modify_field" - for suggesting changes to an existing field
        - "remove_field" - for suggesting removal of an unnecessary field
        - "add_class" - for suggesting a new class to create
        - "modify_class" - for suggesting changes to an existing class structure 
        - "restructure" - for suggesting overall schema restructuring
        - "general" - for general schema advice
        - "good_practice" - for suggesting Pydantic best practices
        
        The "priority" field MUST be one of: "high", "medium", or "low"
        
        DO NOT use any other values for the "type" field, or the validation will fail.
        
        Ensure your response is a validly formatted JSON object and nothing else.
        """
        
        messages.append({"role": "user", "content": structured_output_request})
        
        model = st.session_state.llm_model_advisor
        
        if st.session_state.llm_provider == "Azure OpenAI":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2500,
                response_format={"type": "json_object"}
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2500,
                response_format={"type": "json_object"}
            )
        
        # Track token usage and cost
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            # Calculate cost
            cost_info = calculate_api_cost(model, prompt_tokens, completion_tokens)
            current_cost = cost_info["total_cost"]
            
            # Update cost tracking
            st.session_state.total_cost += current_cost
            st.session_state.cost_breakdown["advisor"] += current_cost
            st.session_state.api_call_count["advisor"] += 1
        
        # Parse the response
        response_text = response.choices[0].message.content
        advisor_json = json.loads(response_text)
        
        # Normalize advisor recommendations
        normalized_advisor_json = normalize_advisor_recommendations(advisor_json)
        
        return AdvisorAnalysis(**normalized_advisor_json)
        
    except Exception as e:
        st.error(f"Error in advisor agent: {str(e)}")
        return None

def structure_optimization_agent_call(schema):
    """Analyze and optimize the structure of the Pydantic schema"""
    try:
        client = get_llm_client()
        
        # Convert schema to JSON string
        schema_json = schema.model_dump_json(indent=2)
        
        # Generate the Python code for the current schema
        current_python_code = generate_pydantic_code(schema)
        
        # Create message to include the schema and code
        messages = [
            {
                "role": "system",
                "content": """You are an expert Pydantic schema structure optimizer. Your task is to analyze a Pydantic schema and suggest 
                structural improvements to make it more elegant, maintainable, and aligned with best practices.
                
                Analyze the schema for:
                1. Hierarchy - Is the class hierarchy logical? Should classes be reorganized?
                2. Composition - Are there opportunities to use composition more effectively?
                3. Common patterns - Are there repeated patterns that could be abstracted?
                4. Naming conventions - Do the classes and fields follow consistent naming conventions?
                5. Code duplication - Is there duplication that could be eliminated with better structure?
                6. Proper type usage - Are the types appropriate and could they be improved?
                
                If the schema structure is already optimal, indicate that no changes are needed.
                
                Your output must be a valid JSON with:
                1. A completely restructured schema (if improvements are possible) or the original (if no improvements needed)
                2. A list of changes made
                3. Reasoning for each change
                4. A flag indicating whether optimization was performed
                """
            }
        ]
        
        # Add the current schema and code
        messages.append({
            "role": "user", 
            "content": f"""
            CURRENT SCHEMA DEFINITION:
            {schema_json}
            
            CURRENT PYTHON CODE:
            ```python
            {current_python_code}
            ```
            
            Please analyze the schema structure and suggest optimizations if needed. Return your response in the following JSON format:
            
            {{
              "is_optimized": true/false,
              "reasoning": "Detailed explanation of the structural analysis and reasoning",
              "changes_made": ["Change 1", "Change 2", ...],
              "optimized_schema": {{
                // Complete schema definition object (same structure as input schema)
                // Only include this if is_optimized is true, otherwise set to null
              }}
            }}
            
            If the schema is already well-structured, set "is_optimized" to false, include your reasoning for why no changes are needed,
            and set "changes_made" to an empty array and "optimized_schema" to null.
            
            If you suggest optimizations, ensure the "optimized_schema" follows the exact same JSON structure as the input schema
            and is a complete representation (not just the changed parts). Fields and classes should maintain their original IDs and
            references where appropriate.
            """
        })
        
        model = st.session_state.llm_model_optimizer
        
        if st.session_state.llm_provider == "Azure OpenAI":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
        
        # Track token usage and cost
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            # Calculate cost
            cost_info = calculate_api_cost(model, prompt_tokens, completion_tokens)
            current_cost = cost_info["total_cost"]
            
            # Update cost tracking
            st.session_state.total_cost += current_cost
            st.session_state.cost_breakdown["structure_optimization"] += current_cost
            st.session_state.api_call_count["structure_optimization"] += 1
        
        # Parse the response
        response_text = response.choices[0].message.content
        optimization_json = json.loads(response_text)
        
        # Create the structure optimization object
        result = {
            "is_optimized": optimization_json.get("is_optimized", False),
            "reasoning": optimization_json.get("reasoning", "No structural optimization analysis provided."),
            "changes_made": optimization_json.get("changes_made", [])
        }
        
        # Handle the optimized schema if present
        if result["is_optimized"] and optimization_json.get("optimized_schema"):
            # Process the optimized schema to ensure it matches our expected structure
            normalized_schema = normalize_schema_json(optimization_json["optimized_schema"])
            try:
                # Create a SchemaDefinition object from the normalized schema
                result["optimized_schema"] = SchemaDefinition(**normalized_schema)
            except Exception as schema_error:
                st.error(f"Error creating optimized schema: {str(schema_error)}")
                result["is_optimized"] = False
                result["reasoning"] += "\n\nThe proposed optimized schema had validation errors and couldn't be applied."
                result["changes_made"].append("Optimization failed due to schema validation errors.")
                result["optimized_schema"] = None
        else:
            result["optimized_schema"] = None
        
        return StructureOptimization(**result)
        
    except Exception as e:
        st.error(f"Error in structure optimization agent: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

def verification_agent_call(schema, is_satisfied=False):
    """Get final verification from the verification agent"""
    try:
        client = get_llm_client()
        
        # Convert schema to JSON string
        schema_json = schema.model_dump_json(indent=2)
        
        # Generate the Python code
        python_code = generate_pydantic_code(schema)
        
        # Create message to include the schema and code
        messages = [
            {
                "role": "system",
                "content": """You are an expert Pydantic schema validator. Your task is to perform a final verification of a Pydantic schema
                before it is delivered to the user.
                
                Analyze the schema carefully for:
                1. Technical correctness - Does it follow Pydantic syntax and conventions properly?
                2. Completeness - Does it include all necessary components for the described domain?
                3. Best practices - Does it follow Pydantic best practices?
                4. Usability - Will the generated code be practical to use?
                
                Your output must be valid JSON that conforms to the FinalVerification model.
                """
            }
        ]
        
        satisfaction_status = "The user has indicated they are SATISFIED with the schema." if is_satisfied else "The user has NOT YET indicated satisfaction with the schema."
        
        # Add the schema and code
        messages.append({
            "role": "user", 
            "content": f"""
            {satisfaction_status}
            
            SCHEMA DEFINITION:
            {schema_json}
            
            GENERATED PYTHON CODE:
            ```python
            {python_code}
            ```
            
            Please provide your final verification in the following JSON structure:
            
            {{
              "approved": true/false,
              "reasons": ["Reason for approval/rejection 1", "Reason 2", ...],
              "suggestions": ["Suggestion for improvement 1", "Suggestion 2", ...]
            }}
            
            If the user is satisfied and you find the schema technically correct and following best practices, you should approve it.
            If there are serious issues that must be fixed, you should not approve it regardless of user satisfaction.
            If there are minor issues, you should approve it if the user is satisfied, but include suggestions for improvement.
            
            Ensure your response is a validly formatted JSON object and nothing else.
            """
        })
        
        model = st.session_state.llm_model_verifier
        
        if st.session_state.llm_provider == "Azure OpenAI":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
        
        # Track token usage and cost
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            # Calculate cost
            cost_info = calculate_api_cost(model, prompt_tokens, completion_tokens)
            current_cost = cost_info["total_cost"]
            
            # Update cost tracking
            st.session_state.total_cost += current_cost
            st.session_state.cost_breakdown["verification"] += current_cost
            st.session_state.api_call_count["verification"] += 1
        
        # Parse the response
        response_text = response.choices[0].message.content
        verification_json = json.loads(response_text)
        
        return FinalVerification(**verification_json)
        
    except Exception as e:
        st.error(f"Error in verification agent: {str(e)}")
        return None

# --- Pydantic Code Generation ---
def generate_pydantic_code(schema):
    """Generate Python code from the schema definition"""
    code_lines = []
    
    # Import statements
    code_lines.append("from pydantic import BaseModel, Field")
    code_lines.append("from typing import Optional, List, Dict, Any, Union")
    code_lines.append("from enum import Enum")
    code_lines.append("")
    code_lines.append("")
    
    # Enums
    if schema.enums:
        code_lines.append("# --- Enum Definitions ---")
        for enum_def in schema.enums:
            code_lines.append(f"class {enum_def.name}({enum_def.base_type}, Enum):")
            code_lines.append(f"    \"\"\"{enum_def.description}\"\"\"")
            for value in enum_def.values:
                code_lines.append(f"    {value['name']} = \"{value['description']}\"")
            code_lines.append("")
        code_lines.append("")
    
    # Sort classes to ensure parent classes are defined before their children
    # First, collect parent-child relationships
    parent_child_map = {}
    for cls in schema.classes:
        if cls.parent_class:
            if cls.parent_class not in parent_child_map:
                parent_child_map[cls.parent_class] = []
            parent_child_map[cls.parent_class].append(cls.name)
    
    # Helper function to determine class order
    def get_class_order(classes):
        result = []
        processed = set()
        
        def process_class(class_name):
            if class_name in processed:
                return
            processed.add(class_name)
            result.append(class_name)
            if class_name in parent_child_map:
                for child in parent_child_map[class_name]:
                    process_class(child)
        
        # Start with classes that have no parent or whose parent is "BaseModel"
        for cls in classes:
            if not cls.parent_class or cls.parent_class == "BaseModel":
                process_class(cls.name)
        
        # Handle any remaining classes
        for cls in classes:
            if cls.name not in processed:
                process_class(cls.name)
        
        return result
    
    # Get the ordered list of class names
    ordered_class_names = get_class_order([cls for cls in schema.classes if not cls.is_enum])
    
    # Filter and order the actual class objects
    ordered_classes = []
    for name in ordered_class_names:
        for cls in schema.classes:
            if cls.name == name and not cls.is_enum:
                ordered_classes.append(cls)
                break
    
    # Generate class definitions
    code_lines.append("# --- Class Definitions ---")
    for cls in ordered_classes:
        parent = cls.parent_class if cls.parent_class else "BaseModel"
        code_lines.append(f"class {cls.name}({parent}):")
        code_lines.append(f"    \"\"\"{cls.description}\"\"\"")
        
        # Add fields for this class
        class_fields = [f for f in schema.fields if f.parent_class == cls.name]
        if not class_fields:
            code_lines.append("    pass")
        else:
            for field in class_fields:
                field_type = field.field_type
                if field.is_optional and not field_type.startswith("Optional["):
                    field_type = f"Optional[{field_type}]"
                
                default_value = "None" if field.default_value is None and field.is_optional else field.default_value
                if default_value is not None and not isinstance(default_value, (int, float, bool)):
                    default_value = f'"{default_value}"'
                
                if field.description:
                    field_line = f"    {field.name}: {field_type} = Field({default_value}, description=\"{field.description}\")"
                else:
                    field_line = f"    {field.name}: {field_type} = {default_value}"
                
                code_lines.append(field_line)
        
        code_lines.append("")
    
    return "\n".join(code_lines)

# --- Session State Management ---
def save_app_state():
    """Save current conversation and schema state to a downloadable file"""
    try:
        # Create a dict with only the essential data we want to save
        state_to_save = {
            "timestamp": datetime.now().isoformat(),
            "chat_history": st.session_state.chat_history,
            "schema": st.session_state.schema.model_dump() if st.session_state.schema else None,
            "advisor_analysis": st.session_state.advisor_analysis.model_dump() if st.session_state.advisor_analysis else None,
            "verification_result": st.session_state.verification_result.model_dump() if st.session_state.verification_result else None,
            "structure_optimization": st.session_state.structure_optimization.model_dump() if st.session_state.structure_optimization else None,
            "final_code": st.session_state.final_code,
            "cost_data": {
                "total_cost": st.session_state.total_cost,
                "cost_breakdown": st.session_state.cost_breakdown,
                "api_call_count": st.session_state.api_call_count
            }
        }
        
        # Convert to JSON string and encode to base64 for safer transport
        state_json = json.dumps(state_to_save)
        state_b64 = base64.b64encode(state_json.encode()).decode()
        
        return state_b64
    except Exception as e:
        st.error(f"Error saving app state: {str(e)}")
        return None
        
def load_app_state(state_b64):
    """Load conversation and schema state from an uploaded file"""
    try:
        # Decode base64 and parse JSON
        state_json = base64.b64decode(state_b64).decode()
        saved_state = json.loads(state_json)
        
        # Update session state with loaded values
        st.session_state.chat_history = saved_state.get("chat_history", [])
        
        # Handle schema if it exists
        if saved_state.get("schema"):
            st.session_state.schema = SchemaDefinition(**saved_state["schema"])
        else:
            st.session_state.schema = None
            
        # Handle advisor analysis if it exists
        if saved_state.get("advisor_analysis"):
            st.session_state.advisor_analysis = AdvisorAnalysis(**saved_state["advisor_analysis"])
        else:
            st.session_state.advisor_analysis = None
            
        # Handle verification result if it exists
        if saved_state.get("verification_result"):
            st.session_state.verification_result = FinalVerification(**saved_state["verification_result"])
        else:
            st.session_state.verification_result = None
        
        # Handle structure optimization if it exists
        if saved_state.get("structure_optimization"):
            # If the optimized_schema is a dict, convert it to a SchemaDefinition object
            if (saved_state["structure_optimization"].get("optimized_schema") and 
                isinstance(saved_state["structure_optimization"]["optimized_schema"], dict)):
                saved_state["structure_optimization"]["optimized_schema"] = SchemaDefinition(
                    **saved_state["structure_optimization"]["optimized_schema"]
                )
            st.session_state.structure_optimization = StructureOptimization(**saved_state["structure_optimization"])
        else:
            st.session_state.structure_optimization = None
            
        # Handle final code
        st.session_state.final_code = saved_state.get("final_code")
        
        # Handle cost data if it exists
        if saved_state.get("cost_data"):
            cost_data = saved_state["cost_data"]
            st.session_state.total_cost = cost_data.get("total_cost", 0.0)
            st.session_state.cost_breakdown = cost_data.get("cost_breakdown", {
                "conversation": 0.0,
                "schema_generation": 0.0,
                "advisor": 0.0,
                "verification": 0.0
            })
            st.session_state.api_call_count = cost_data.get("api_call_count", {
                "conversation": 0,
                "schema_generation": 0,
                "advisor": 0,
                "verification": 0
            })
        
        return True
    except Exception as e:
        st.error(f"Error loading app state: {str(e)}")
        return False

# --- UI Components ---
def setup_sidebar():
    """Setup the sidebar with LLM provider settings"""
    with st.sidebar:
        st.title("Pydantic Generator Settings")
        
        # LLM Provider selection
        st.session_state.llm_provider = st.radio(
            "Select LLM Provider",
            ["OpenAI", "Azure OpenAI", "OpenAI-Compatible (Ollama, Fireworks, etc.)"]
        )
        
        # Provider-specific settings
        if st.session_state.llm_provider == "OpenAI":
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
        elif st.session_state.llm_provider == "Azure OpenAI":
            st.session_state.azure_api_key = st.text_input("Azure OpenAI API Key", type="password", value=st.session_state.azure_api_key)
            st.session_state.azure_endpoint = st.text_input("Azure Endpoint", value=st.session_state.azure_endpoint)
            st.session_state.azure_api_version = st.text_input("API Version", value=st.session_state.azure_api_version)
        else:  # OpenAI-Compatible
            st.session_state.openai_api_key = st.text_input("API Key", type="password", value=st.session_state.openai_api_key)
            st.session_state.base_url = st.text_input("Base URL", value=st.session_state.base_url)
        
        # Model selection
        st.subheader("Model Selection")
        
        # Default model suggestions based on provider
        default_models = {
            "OpenAI": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o", "o3", "o3-mini"],
            "Azure OpenAI": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o", "o3", "o3-mini"],
            "OpenAI-Compatible (Ollama, Fireworks, etc.)": ["hermes3:8b-llama3.1-q8_0", "llama3:8b", "mistral:7b", "mixtral:8x7b"]
        }
        
        provider_key = st.session_state.llm_provider
        default_model_options = default_models.get(provider_key, ["gpt-4.1-mini"])
        
        # Conversation agent model
        st.session_state.llm_model_conversation = st.selectbox(
            "Conversation Agent Model",
            options=default_model_options,
            index=0,
            key="conv_model"
        )
        
        # Schema generator model
        st.session_state.llm_model_schema = st.selectbox(
            "Schema Generator Model",
            options=default_model_options,
            index=0,
            key="schema_model"
        )
        
        # Advisor agent model
        st.session_state.llm_model_advisor = st.selectbox(
            "Advisor Agent Model",
            options=default_model_options,
            index=0,
            key="advisor_model"
        )
        
        # Verification agent model
        st.session_state.llm_model_verifier = st.selectbox(
            "Verification Agent Model",
            options=default_model_options,
            index=0,
            key="verifier_model"
        )
        
        # Structure optimization agent model
        st.session_state.llm_model_optimizer = st.selectbox(
            "Structure Optimization Model",
            options=default_model_options,
            index=0,
            key="optimizer_model"
        )
        
        # Add a divider
        st.divider()
        
        # Add API connection check
        if st.button("Check API Connection"):
            try:
                client = get_llm_client()
                # Simple API call to test connection
                response = client.chat.completions.create(
                    model=st.session_state.llm_model_conversation,
                    messages=[{"role": "user", "content": "Hello, this is a test message."}],
                    max_tokens=20
                )
                st.success("✅ API connection successful!")
            except Exception as e:
                st.error(f"❌ API connection failed: {str(e)}")
        
        # Add a divider
        st.divider()
        
        # Save/Load UI section
        st.subheader("Conversation Backup")
        
        # Download button for saving state
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            state_data = save_app_state()
            if state_data:
                # Generate a filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pydantic_conversation_{timestamp}.b64"
                
                st.download_button(
                    label="Save Conversation",
                    data=state_data,
                    file_name=filename,
                    mime="text/plain"
                )
        
        # Upload button for loading state
        uploaded_file = st.file_uploader("Restore from backup:", type=['b64', 'txt'])
        if uploaded_file is not None:
            # Read and load the file
            state_b64 = uploaded_file.read().decode()
            if load_app_state(state_b64):
                st.success("Conversation restored successfully!")
                # Add a button to refresh the UI
                if st.button("Refresh UI"):
                    st.rerun()
        
        # Reset button
        if st.button("Reset Conversation"):
            # Reset all conversation and schema related state
            reset_app_state()
            st.rerun()
            
        # Add a divider
        st.divider()
        
        # Cost tracking section
        st.subheader("Cost Tracking")
        
        # Pricing settings
        with st.expander("Pricing Settings"):
            st.caption("Cost per 1 million tokens in USD")
            
            # Initialize pricing settings if not present
            if "pricing_settings" not in st.session_state:
                # Default pricing per 1 million tokens as of May 2025
                st.session_state.pricing_settings = {
                    "gpt-4o": {"prompt": 5000.0, "completion": 15000.0},
                    "gpt-4.1-mini": {"prompt": 1500.0, "completion": 5000.0},
                    "gpt-4": {"prompt": 10000.0, "completion": 30000.0},
                    "gpt-3.5-turbo": {"prompt": 500.0, "completion": 1500.0},
                    "default": {"prompt": 1000.0, "completion": 2000.0}
                }
            
            # Create a copy to avoid modifying the dictionary during iteration
            models = list(st.session_state.pricing_settings.keys())
            
            # Option to add custom model
            custom_model = st.text_input("Add custom model (optional)")
            if custom_model and custom_model not in st.session_state.pricing_settings:
                st.session_state.pricing_settings[custom_model] = {"prompt": 1.0, "completion": 2.0}
                models.append(custom_model)
            
            # Display pricing settings for each model
            selected_model = st.selectbox("Select model to edit pricing", models)
            
            col1, col2 = st.columns(2)
            with col1:
                new_prompt_price = st.number_input(
                    "Input token price",
                    min_value=0.0,
                    value=st.session_state.pricing_settings[selected_model]["prompt"],
                    step=0.1,
                    format="%.2f"
                )
            with col2:
                new_completion_price = st.number_input(
                    "Output token price",
                    min_value=0.0,
                    value=st.session_state.pricing_settings[selected_model]["completion"],
                    step=0.1,
                    format="%.2f"
                )
            
            # Update pricing settings
            st.session_state.pricing_settings[selected_model]["prompt"] = new_prompt_price
            st.session_state.pricing_settings[selected_model]["completion"] = new_completion_price
            
            # Reset cost button
            if st.button("Reset Cost Counter"):
                st.session_state.total_cost = 0.0
                st.session_state.cost_breakdown = {
                    "conversation": 0.0,
                    "schema_generation": 0.0,
                    "advisor": 0.0,
                    "verification": 0.0,
                    "structure_optimization": 0.0
                }
                st.session_state.api_call_count = {
                    "conversation": 0,
                    "schema_generation": 0,
                    "advisor": 0,
                    "verification": 0,
                    "structure_optimization": 0
                }
                st.success("Cost counters have been reset!")
                st.rerun()
        
        st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
        
        # Display cost breakdown
        with st.expander("Cost Breakdown"):
            cost_data = {
                "Agent": ["Conversation", "Schema Generation", "Advisor", "Verification", "Structure Optimization", "Total"],
                "Cost ($)": [
                    f"${st.session_state.cost_breakdown['conversation']:.4f}",
                    f"${st.session_state.cost_breakdown['schema_generation']:.4f}",
                    f"${st.session_state.cost_breakdown['advisor']:.4f}",
                    f"${st.session_state.cost_breakdown['verification']:.4f}",
                    f"${st.session_state.cost_breakdown['structure_optimization']:.4f}",
                    f"${st.session_state.total_cost:.4f}"
                ],
                "API Calls": [
                    st.session_state.api_call_count['conversation'],
                    st.session_state.api_call_count['schema_generation'],
                    st.session_state.api_call_count['advisor'],
                    st.session_state.api_call_count['verification'],
                    st.session_state.api_call_count['structure_optimization'],
                    sum(st.session_state.api_call_count.values())
                ]
            }
            
            st.dataframe(cost_data, use_container_width=True)
            
            # Display current pricing for the models in use
            st.write("**Current Model Pricing (per 1 million tokens)**")
            active_models = {
                "Conversation": st.session_state.llm_model_conversation,
                "Schema Generation": st.session_state.llm_model_schema, 
                "Advisor": st.session_state.llm_model_advisor,
                "Verification": st.session_state.llm_model_verifier,
                "Structure Optimization": st.session_state.llm_model_optimizer
            }
            
            pricing_data = {
                "Agent": [],
                "Model": [],
                "Input Price ($)": [],
                "Output Price ($)": []
            }
            
            for agent, model in active_models.items():
                # Find the appropriate pricing
                base_model = model.split(':')[0].split('@')[0].lower()
                model_pricing = None
                for key in st.session_state.pricing_settings:
                    if key.lower() in base_model:
                        model_pricing = st.session_state.pricing_settings[key]
                        break
                
                if not model_pricing:
                    model_pricing = st.session_state.pricing_settings["default"]
                
                pricing_data["Agent"].append(agent)
                pricing_data["Model"].append(model)
                pricing_data["Input Price ($)"].append(f"${model_pricing['prompt']:.2f}")
                pricing_data["Output Price ($)"].append(f"${model_pricing['completion']:.2f}")
            
            st.dataframe(pricing_data, use_container_width=True)
            
            # Add a button to export cost report
            if st.button("Export Cost Report"):
                cost_report = {
                    "timestamp": datetime.now().isoformat(),
                    "total_cost": st.session_state.total_cost,
                    "breakdown": st.session_state.cost_breakdown,
                    "api_calls": st.session_state.api_call_count,
                    "models_used": {
                        "conversation": st.session_state.llm_model_conversation,
                        "schema_generation": st.session_state.llm_model_schema,
                        "advisor": st.session_state.llm_model_advisor,
                        "verification": st.session_state.llm_model_verifier,
                        "structure_optimization": st.session_state.llm_model_optimizer
                    }
                }
                
                # Convert to JSON string and encode for download
                cost_report_json = json.dumps(cost_report, indent=2)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download Cost Report",
                    data=cost_report_json,
                    file_name=f"pydantic_cost_report_{timestamp}.json",
                    mime="application/json"
                )

def render_conversation():
    """Render the conversation interface"""
    st.title("Pydantic Model Generator")
    
    # Display notification if conversation was reset
    if "just_reset" in st.session_state and st.session_state.just_reset:
        st.success("Conversation and schema have been reset. Start a new conversation!")
        st.session_state.just_reset = False
    
    # Chat interface
    for message in st.session_state.chat_history:
        role_color = "blue" if message["role"] == "user" else "green"
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
    
    # Input for new messages
    if user_input := st.chat_input("Describe your data model..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Show user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        
        # Get assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = conversation_agent_call(user_input)
                st.markdown(response)
        
        # Add assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Update schema and analysis after each exchange
        with st.spinner("Updating Pydantic schema..."):
            update_schema_and_analysis()

def render_schema():
    """Render the current Pydantic schema"""
    st.title("Generated Pydantic Schema")
    
    # Display reset notification if applicable
    if "just_reset" in st.session_state and st.session_state.just_reset:
        st.info("Schema has been reset. Start a new conversation to generate a schema.")
    
    if st.session_state.schema:
        # Add a button to regenerate the schema
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Regenerate Schema"):
                with st.spinner("Regenerating schema..."):
                    update_schema_and_analysis(force_update=True)
                    st.rerun()
        
        with col2:
            if st.button("Improve Structure"):
                with st.spinner("Analyzing and optimizing schema structure..."):
                    st.session_state.structure_optimization = structure_optimization_agent_call(st.session_state.schema)
                    # If optimizations were made, update the schema
                    if (st.session_state.structure_optimization and 
                        st.session_state.structure_optimization.is_optimized and 
                        st.session_state.structure_optimization.optimized_schema):
                        st.session_state.schema = st.session_state.structure_optimization.optimized_schema
                st.rerun()
        
        # Add a button to finalize
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("I'm Satisfied", type="primary"):
                with st.spinner("Performing final verification..."):
                    st.session_state.verification_result = verification_agent_call(st.session_state.schema, is_satisfied=True)
                    if st.session_state.verification_result and st.session_state.verification_result.approved:
                        st.session_state.final_code = generate_pydantic_code(st.session_state.schema)
                st.rerun()
        
        with col2:
            if st.session_state.verification_result and st.session_state.verification_result.approved:
                if st.download_button(
                    label="Download Pydantic Code",
                    data=st.session_state.final_code,
                    file_name="pydantic_model.py",
                    mime="text/plain"
                ):
                    st.success("File downloaded successfully!")
        
        # Display the generated code
        python_code = generate_pydantic_code(st.session_state.schema)
        st.code(python_code, language="python")
        
        # Display structure optimization results if available
        if st.session_state.structure_optimization:
            with st.expander("Structure Optimization Results", expanded=True):
                if st.session_state.structure_optimization.is_optimized:
                    st.success("✅ Schema structure has been optimized!")
                    st.markdown("### Changes Made:")
                    for change in st.session_state.structure_optimization.changes_made:
                        st.markdown(f"- {change}")
                else:
                    st.info("ℹ️ No structural optimizations were needed.")
                
                st.markdown("### Reasoning:")
                st.markdown(st.session_state.structure_optimization.reasoning)
        
        # Display verification result if available
        if st.session_state.verification_result:
            if st.session_state.verification_result.approved:
                st.success("✅ Schema approved! The verification agent is satisfied with your schema.")
                if st.session_state.verification_result.suggestions:
                    with st.expander("Optional Suggestions for Future Improvement"):
                        for suggestion in st.session_state.verification_result.suggestions:
                            st.markdown(f"- {suggestion}")
            else:
                st.error("❌ Schema needs improvement before it can be finalized.")
                if st.session_state.verification_result.reasons:
                    with st.expander("Issues to Address"):
                        for reason in st.session_state.verification_result.reasons:
                            st.markdown(f"- {reason}")
                if st.session_state.verification_result.suggestions:
                    with st.expander("Suggestions"):
                        for suggestion in st.session_state.verification_result.suggestions:
                            st.markdown(f"- {suggestion}")
    else:
        st.info("Start a conversation to generate your Pydantic schema.")

def render_advisor():
    """Render the advisor's recommendations"""
    st.title("Advisor Recommendations")
    
    # Display reset notification if applicable
    if "just_reset" in st.session_state and st.session_state.just_reset:
        st.info("Advisor recommendations have been reset.")
    
    if st.session_state.advisor_analysis:
        # Show general feedback
        st.markdown("### General Feedback")
        st.markdown(st.session_state.advisor_analysis.general_feedback)
        
        # Show missing information
        if st.session_state.advisor_analysis.missing_information:
            st.markdown("### Missing Information")
            for missing in st.session_state.advisor_analysis.missing_information:
                st.markdown(f"- {missing}")
        
        # Show recommendations
        if st.session_state.advisor_analysis.recommendations:
            st.markdown("### Recommendations")
            
            # Group recommendations by priority
            priorities = ["high", "medium", "low"]
            for priority in priorities:
                priority_recs = [r for r in st.session_state.advisor_analysis.recommendations if r.priority == priority]
                if priority_recs:
                    with st.expander(f"{priority.upper()} Priority ({len(priority_recs)})"):
                        for rec in priority_recs:
                            st.markdown(f"**{rec.type.replace('_', ' ').title()}**: {rec.description}")
                            st.markdown(f"*Reason*: {rec.reason}")
                            if rec.example:
                                st.code(rec.example, language="python")
                            st.divider()
    else:
        st.info("Start a conversation to get advisor recommendations.")

def update_schema_and_analysis(force_update=False):
    """Update the schema and advisor analysis if enough conversation has occurred"""
    try:
        # Only update if we have at least 4 messages (2 exchanges) or if forced
        if force_update or (len(st.session_state.chat_history) >= 4):
            with st.spinner("Generating schema from conversation..."):
                st.session_state.schema = schema_generator_call()
                
            if st.session_state.schema:
                with st.spinner("Analyzing schema for recommendations..."):
                    st.session_state.advisor_analysis = advisor_agent_call(st.session_state.schema)
                    
                # Clear any previous verification results when schema changes
                st.session_state.verification_result = None
                st.session_state.structure_optimization = None
                st.session_state.final_code = None
    except Exception as e:
        st.error(f"Error updating schema and analysis: {str(e)}")

def normalize_schema_json(schema_json):
    """Normalize the schema JSON to ensure it's compatible with our Pydantic models"""
    try:
        # Make a copy to avoid modifying the original
        normalized = schema_json.copy()
        
        # Fix enum values in fields - this is where the validation error occurs
        if "fields" in normalized:
            for i, field in enumerate(normalized["fields"]):
                if field.get("enum_values") and isinstance(field["enum_values"], list):
                    # Check if any of the values are simple strings (not dictionaries)
                    has_simple_strings = any(isinstance(val, str) for val in field["enum_values"])
                    
                    if has_simple_strings:
                        # Convert simple string values to the required dictionary format
                        field["enum_values"] = [
                            {"name": val.upper(), "description": val.replace("_", " ").title()} 
                            if isinstance(val, str) else val 
                            for val in field["enum_values"]
                        ]
        
        # Fix enum values in the enums section
        if "enums" in normalized:
            for i, enum in enumerate(normalized["enums"]):
                if "values" in enum and isinstance(enum["values"], list):
                    has_simple_strings = any(isinstance(val, str) for val in enum["values"])
                    
                    if has_simple_strings:
                        enum["values"] = [
                            {"name": val.upper(), "description": val.replace("_", " ").title()} 
                            if isinstance(val, str) else val 
                            for val in enum["values"]
                        ]
        
        return normalized
        
    except Exception as e:
        st.error(f"Error in schema normalization: {str(e)}")
        st.error(f"Detailed normalization error: {traceback.format_exc()}")
        # Return the original to avoid breaking the flow
        return schema_json

def debug_schema_structure(schema_json):
    """Debug helper to log the structure of the schema JSON to help diagnose issues"""
    try:
        debug_info = []
        
        # Print main class
        debug_info.append(f"Main class: {schema_json.get('main_class', 'NOT SET')}")
        
        # Classes count
        classes = schema_json.get("classes", [])
        debug_info.append(f"Number of classes: {len(classes)}")
        
        # Fields count and enum fields
        fields = schema_json.get("fields", [])
        debug_info.append(f"Number of fields: {len(fields)}")
        
        # Count fields with enum values
        enum_fields = [f for f in fields if f.get("enum_values")]
        debug_info.append(f"Fields with enum_values: {len(enum_fields)}")
        
        # Examine the structure of enum values in fields
        for i, field in enumerate(enum_fields):
            if field.get("enum_values"):
                enum_vals = field["enum_values"]
                if enum_vals and len(enum_vals) > 0:
                    first_val = enum_vals[0]
                    debug_info.append(f"Field {i+1} ({field.get('name', 'unnamed')}): enum_values[0] type = {type(first_val).__name__}")
                    if isinstance(first_val, str):
                        debug_info.append(f"  - Value: {first_val}")
                    elif isinstance(first_val, dict):
                        debug_info.append(f"  - Keys: {list(first_val.keys())}")
                        debug_info.append(f"  - Value: {first_val}")
        
        # Count enums
        enums = schema_json.get("enums", [])
        debug_info.append(f"Number of enums: {len(enums)}")
        
        # Return debug info as string
        return "\n".join(debug_info)
        
    except Exception as e:
        return f"Error in debug function: {str(e)}"

def normalize_advisor_recommendations(advisor_json):
    """
    Normalize the advisor recommendations to ensure the type field
    matches one of the allowed literal values.
    """
    if not advisor_json or "recommendations" not in advisor_json:
        return advisor_json
    
    # Map of common incorrect types to their correct versions
    type_mapping = {
        # Common incorrect types and their corrections
        "add_validation": "modify_field",
        "add_enum": "modify_field",
        "validation": "modify_field",
        "enum": "modify_field",
        "structure": "restructure",
        "best_practice": "good_practice",
        "rename_field": "modify_field",
        "refactor": "restructure",
        "documentation": "general",
        "optimize": "general"
    }
    
    # Allowed types in AdvisorRecommendation class
    allowed_types = [
        "add_field", "modify_field", "remove_field", 
        "add_class", "modify_class", "restructure", 
        "general", "good_practice"
    ]
    
    # Normalize recommendation types
    for rec in advisor_json["recommendations"]:
        if "type" in rec:
            # Convert to lowercase for case-insensitive matching
            rec_type = rec["type"].lower()
            
            # Direct match to allowed types (case insensitive)
            for allowed_type in allowed_types:
                if rec_type == allowed_type.lower():
                    rec["type"] = allowed_type
                    break
            else:
                # No direct match found, try mapping
                if rec_type in type_mapping:
                    rec["type"] = type_mapping[rec_type]
                else:
                    # Default to "general" if no match found
                    rec["type"] = "general"
                    
            # Ensure priority is one of the allowed values
            if "priority" in rec:
                priority = rec["priority"].lower()
                if priority not in ["high", "medium", "low"]:
                    # Default to medium
                    rec["priority"] = "medium"
            else:
                rec["priority"] = "medium"
    
    # Ensure general_feedback is present
    if "general_feedback" not in advisor_json:
        advisor_json["general_feedback"] = "The schema appears to be well-structured overall, but please review the specific recommendations for improvements."
    
    return advisor_json

def calculate_api_cost(model_name, prompt_tokens, completion_tokens):
    """Calculate the cost of an API call based on model and tokens used"""
    # Get pricing from session state or use defaults
    if "pricing_settings" not in st.session_state:
        # Default pricing per 1 million tokens as of May 2025
        st.session_state.pricing_settings = {
            "gpt-4o": {"prompt": 5000.0, "completion": 15000.0},
            "gpt-4.1-mini": {"prompt": 1500.0, "completion": 5000.0},
            "gpt-4": {"prompt": 10000.0, "completion": 30000.0},
            "gpt-3.5-turbo": {"prompt": 500.0, "completion": 1500.0},
            "default": {"prompt": 1000.0, "completion": 2000.0}
        }
    
    pricing = st.session_state.pricing_settings
    
    # Extract base model name (remove any deployment-specific suffixes)
    base_model = model_name.split(':')[0].split('@')[0].lower()
    
    # Determine which pricing to use
    model_pricing = None
    for key in pricing:
        if key.lower() in base_model:
            model_pricing = pricing[key]
            break
    
    if not model_pricing:
        model_pricing = pricing["default"]
    
    # Calculate costs
    prompt_cost = (prompt_tokens / 1000000) * model_pricing["prompt"]
    completion_cost = (completion_tokens / 1000000) * model_pricing["completion"]
    total_cost = prompt_cost + completion_cost
    
    return {
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost
    }
    
    return {
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost
    }

# --- Main App ---
def main():
    # Initialize session state
    init_session_state()
    
    # Setup sidebar
    setup_sidebar()
    
    # Create 3-column layout
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    # Render components in each column
    with col1:
        render_conversation()
    
    with col2:
        render_schema()
    
    with col3:
        render_advisor()

if __name__ == "__main__":
    main()

def structure_optimization_agent_call(schema):
    """Optimize the structure of the current schema"""
    try:
        client = get_llm_client()
        
        # Convert schema to JSON string
        schema_json = schema.model_dump_json(indent=2)
        
        # Generate the Python code for the current schema
        current_code = generate_pydantic_code(schema)
        
        # Create message to include the schema
        messages = [
            {
                "role": "system",
                "content": """You are an expert Pydantic schema structure optimizer. Your task is to analyze a Pydantic schema definition
                and suggest optimizations to its structure to make it more:
                
                1. Maintainable - Is the hierarchy of classes logical and easy to understand?
                2. Reusable - Are there opportunities to create common base classes or shared components?
                3. Efficient - Could the structure be simplified without losing information?
                4. Well-organized - Are related fields grouped together appropriately?
                5. Following best practices - Does the schema follow Pydantic best practices for structure?
                
                Analyze the current schema and decide if structural improvements can be made.
                If improvements are possible, provide a complete optimized schema that maintains all the same data fields 
                but with a better structure. If the current structure is already optimal, indicate that no changes are needed.
                
                Your output must be valid JSON conforming to the StructureOptimization model.
                """
            }
        ]
        
        # Add the current schema
        messages.append({"role": "user", "content": f"CURRENT SCHEMA DEFINITION:\n\n{schema_json}"})
        messages.append({"role": "user", "content": f"CURRENT PYTHON CODE:\n\n```python\n{current_code}\n```"})
        
        # Add request for structured output
        structured_output_request = """
        Analyze the schema above, then provide your optimization in the following JSON structure:
        
        {
          "optimized_schema": null,  // Set to null if no optimization needed, otherwise provide complete SchemaDefinition
          "changes_made": [],        // List of strings describing each structural change made
          "reasoning": "Detailed explanation of why these changes improve the structure or why no changes were needed",
          "is_optimized": false      // Set to true if you made optimizations, false if current structure is already optimal
        }
        
        If you decide to optimize the schema, the "optimized_schema" should be a complete SchemaDefinition following this structure:
        
        {
          "main_class": "Name of the main class",
          "classes": [
            {
              "name": "ClassName",
              "description": "Description of the class",
              "is_base_model": true,
              "parent_class": null,
              "is_enum": false
            },
            ...
          ],
          "fields": [
            {
              "name": "field_name",
              "field_type": "str",
              "parent_class": "ClassName",
              "description": "Description of the field",
              "is_optional": true,
              "enum_values": null,
              "default_value": null
            },
            ...
          ],
          "enums": [
            {
              "name": "EnumName",
              "base_type": "str",
              "description": "Description of the enum",
              "values": [
                { "name": "VALUE1", "description": "Description of VALUE1" },
                { "name": "VALUE2", "description": "Description of VALUE2" }
              ]
            },
            ...
          ]
        }
        
        If the current structure is already optimal, set "optimized_schema" to null and explain your reasoning.
        
        Ensure your response is a validly formatted JSON object and nothing else.
        """
        
        messages.append({"role": "user", "content": structured_output_request})
        
        model = st.session_state.llm_model_optimizer
        
        if st.session_state.llm_provider == "Azure OpenAI":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
        
        # Track token usage and cost
        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            # Calculate cost
            cost_info = calculate_api_cost(model, prompt_tokens, completion_tokens)
            current_cost = cost_info["total_cost"]
            
            # Update cost tracking
            st.session_state.total_cost += current_cost
            st.session_state.cost_breakdown["structure_optimization"] += current_cost
            st.session_state.api_call_count["structure_optimization"] += 1
        
        # Parse the response
        response_text = response.choices[0].message.content
        optimization_json = json.loads(response_text)
        
        # Process the optimized schema if present
        if optimization_json.get("is_optimized", False) and optimization_json.get("optimized_schema"):
            # Normalize the schema JSON to ensure it's compatible with our models
            normalized_schema = normalize_schema_json(optimization_json["optimized_schema"])
            try:
                optimized_schema = SchemaDefinition(**normalized_schema)
                optimization_json["optimized_schema"] = optimized_schema
            except Exception as validation_error:
                st.error(f"Schema validation error: {str(validation_error)}")
                st.error("The LLM provided an optimized schema that doesn't match our expected format.")
                optimization_json["optimized_schema"] = None
                optimization_json["is_optimized"] = False
                optimization_json["changes_made"] = []
                optimization_json["reasoning"] = "Error in schema validation. Keeping the original schema."
        
        return StructureOptimization(**optimization_json)
        
    except Exception as e:
        st.error(f"Error in structure optimization agent: {str(e)}")
        return None