#!/usr/bin/env python3
"""
Test the improved generate_default_values function
"""

from typing import Optional, List, Literal
from enum import Enum
from pydantic import BaseModel

# Mock simple types to simulate the real model structure
class TestEnum(Enum):
    VALUE1 = "value1"
    VALUE2 = "value2"

class NestedModel(BaseModel):
    field1: Optional[str] = None
    field2: Optional[int] = None
    
class DeepNestedModel(BaseModel):
    numeric: Optional[float] = None
    unit: Optional[str] = None

class MiddleModel(BaseModel):
    simple_field: Optional[str] = None
    deep_nested: Optional[DeepNestedModel] = None

class TestModel(BaseModel):
    atria: Optional[MiddleModel] = None
    ventricles: Optional[NestedModel] = None
    enum_field: Optional[TestEnum] = None
    list_field: Optional[List[str]] = None

def simulate_generate_default_values(model_class):
    """Simulate the improved generate_default_values function"""
    import inspect
    from typing import get_origin, get_args
    
    def is_optional_type(field_type):
        """Check if a field type is Optional (Union with None)"""
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            return type(None) in field_type.__args__
        return False
    
    def get_base_type(field_type):
        """Extract the base type from Optional or other generic types"""
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            # For Optional types, get the non-None type
            non_none_args = [arg for arg in field_type.__args__ if arg is not type(None)]
            return non_none_args[0] if non_none_args else type(None)
        return field_type
    
    if not model_class or not hasattr(model_class, 'model_fields'):
        return {}
    
    default_values = {}
    
    def get_default_value(field_info):
        """Get the default value for a field based on its type."""
        try:
            field_annotation = field_info.annotation
            is_opt = is_optional_type(field_annotation)
            base_type = get_base_type(field_annotation)
            
            # For nested pydantic models, always create the full structure 
            # even if optional, to ensure consistent flattening
            if inspect.isclass(base_type) and issubclass(base_type, BaseModel):
                return simulate_generate_default_values(base_type)
            
            # For lists, always create empty list structure
            if get_origin(base_type) is list:
                return []
            
            # For optional primitive types, still return None for most, but "" for strings
            if is_opt:
                # Enum
                if inspect.isclass(base_type) and issubclass(base_type, Enum):
                    return None
                # Literal  
                if get_origin(base_type) is Literal:
                    return None
                # Boolean
                if base_type == bool:
                    return None
                # Numbers
                if base_type == int:
                    return None
                if base_type == float:
                    return None
                # String
                return ""
            
            # Non-optional defaults
            # Enum
            if inspect.isclass(base_type) and issubclass(base_type, Enum):
                return None
            
            # Literal
            if get_origin(base_type) is Literal:
                return None
            
            # Boolean
            if base_type == bool:
                return None
            
            # Numbers
            if base_type == int:
                return 0
            if base_type == float:
                return 0.0
            
            # String (default)
            return ""
            
        except Exception as e:
            print(f"Error processing field: {e}")
            return None
    
    # Generate defaults for all fields
    for field_name, field_info in model_class.model_fields.items():
        default_values[field_name] = get_default_value(field_info)
    
    return default_values

def test_improved_defaults():
    """Test the improved default values generation"""
    print("Testing improved default values generation...")
    
    # Test the improved function
    defaults = simulate_generate_default_values(TestModel)
    
    print("\n=== IMPROVED DEFAULT VALUES ===")
    import json
    print(json.dumps(defaults, indent=2))
    
    # Simulate flatten_for_export
    def flatten_for_export(obj, prefix="", separator="::"):
        """Simulate the flatten function"""
        result = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}{separator}{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    result.update(flatten_for_export(v, new_key, separator))
                else:
                    result[new_key] = v
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_key = f"{prefix}{separator}{i}"
                if isinstance(item, (dict, list)):
                    result.update(flatten_for_export(item, new_key, separator))
                else:
                    result[new_key] = item
        return result
    
    flattened = flatten_for_export(defaults)
    print(f"\n=== FLATTENED IMPROVED DEFAULTS ===")
    print(json.dumps(flattened, indent=2))
    
    print(f"\n=== ANALYSIS ===")
    print(f"Number of flattened fields: {len(flattened)}")
    print(f"Sample keys: {list(flattened.keys())[:5]}")
    
    # Check if this creates the same structure as manual save
    expected_keys = [
        "atria::simple_field",
        "atria::deep_nested::numeric", 
        "atria::deep_nested::unit",
        "ventricles::field1",
        "ventricles::field2"
    ]
    
    all_present = all(key in flattened for key in expected_keys)
    print(f"All expected nested keys present: {all_present}")
    
    return all_present

if __name__ == "__main__":
    from typing import Union
    
    success = test_improved_defaults()
    if success:
        print("\n✅ The improved function should fix the format consistency issue!")
    else:
        print("\n❌ There are still issues with the default values generation.")
