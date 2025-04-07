"""
Example Pydantic model for testing extraction and validation workflows.

Served by: Direct import

Path to venv, if required: "the_venvs/venv_example_schema"

Libraries to import:
- pydantic
"""

###################
####  imports  ####
###################
from pydantic import BaseModel, Field
from typing import List, Optional

###############################
####  Example Pydantic Model  ####
###############################
class ExampleModel(BaseModel):
    field1: str = Field(description="A string field")
    field2: int = Field(description="An integer field")
    field3: Optional[List[str]] = Field(default=None, description="An optional list of strings")
    
    class Config:
        schema_extra = {
            "example": {
                "field1": "example_value",
                "field2": 42,
                "field3": ["item1", "item2"]
            }
        }
