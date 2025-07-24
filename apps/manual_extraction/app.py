"""
Pydantic Extraction App

# Description
    [A manual extraction application that allows users to upload documents and extract information 
    with assistance from LLMs. Provides a user interface for document processing and extraction operations.]

    - Arguments:
        - Data (Excel, CSV, JSON, TXT, DOCX, PDF): The file having the unstructured text for review  .
        - Pydantic Model (.py) Or Code (string): The file having the pydantic model code for the extraction.
        - Previous Session (JSON): The file having the previous extraction data for review.

    - Enviroment Arguments:
        - COLOR_PALETTE (list): A list of hex color codes for the pydantic top level fields.
        - BRIGHTER_COLOR_RATE (float):  The rate of brightness increase for each nested field.
        - LONG_TEXT_FIELD_LIST (list): A list of field names that are considered long text fields.
        - flatten_for_export_SEPARATOR (str): The separator used for flattening nested structures for export.
 
    - Returns
        - Session (JSON): The file having the app memory (including Data and Pydantic Model and previous extractions). This is usable for saving and loading to continue the extraction.
        - Extractions (JSON)
        - Extractions (CSV)

# Engine:
    - Serve (utils/data/main-function/sub-function): main-function
    - Served by (API/Direct/Subprocess): Subprocess
    - Path to venv, if require separate venv: the_venvs/venv_streamlit
    - libraries to import: [pydantic,PyPDF2,docx2txt,pandas,openpyxl] 

# Identity
    - Last Status (future/in-progress/complete/published): published
    - Publish Date: 2025-04-07
    - Version: 0.1
    - License: MIT
    - Author: Seyed Amir Ahmad Safavi-Naini Safavi-Naini, sdamirsa@gmail.com (the nominee for the longest name ever)
    - Source: https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps

# Changelog
    - 2025-04-07: version 0.1

# To-do: 
    - [] Add the functionality to add LLM output to the memory (based on the id column). It should check the ...
    - [] Fix the json session load
    - [X] Fox minimum of text input from 60 to 68
    - [] resolve the incorrectly saving previous data for the new patient 
    - [] The issue with rendering the text (exit code)
"""



import streamlit as st
import pandas as pd
import json
import traceback
import inspect
import types
from datetime import datetime
from enum import Enum
from typing import get_type_hints, get_origin, get_args, Dict, List, Optional, Literal, Union, Any
from pydantic import BaseModel, Field
import docx2txt
import PyPDF2
import copy

# =====================================================
# 0) Enviroment Arguments
# =====================================================

COLOR_PALETTE = [
    "#FFCDD2", "#C8E6C9", "#BBDEFB", "#FFE0B2",
    "#D1C4E9", "#B2DFDB", "#F8BBD0", "#FFF9C4",
    "#DCEDC8", "#FFCCBC"
]

BRIGHTER_COLOR_RATE = 0.22

LONG_TEXT_FIELD_LIST = ["description", "comment", "notes", "information", "text"]

flatten_for_export_SEPARATOR = "::"

# =====================================================
# 1) UTILITY FUNCTIONS
# =====================================================

def load_model_code(code_str: str):
    """
    Dynamically load and return all pydantic model classes defined in the given string.
    Also return a list of valid BaseModel class names for the dropdown.
    """
    try:
        import types
        module = types.ModuleType('dynamic_models')
        # Add the required imports to the module namespace
        module.__dict__.update({
            'BaseModel': BaseModel,
            'Field': Field,
            'Enum': Enum,
            'Optional': Optional,
            'List': List,
            'Dict': Dict,
            'Union': Union,
            'Literal': Literal,
            'Any': Any,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
        })
        exec(code_str, module.__dict__)

        all_classes = {}
        for k, v in module.__dict__.items():
            if (
                inspect.isclass(v) 
                and issubclass(v, BaseModel) 
                and v is not BaseModel
            ):
                all_classes[k] = v
        return all_classes
    except Exception as e:
        st.error(f"Error loading model code: {e}")
        st.error(traceback.format_exc())
        return {}

def parse_uploaded_file(file) -> Dict[str, Any]:
    """
    Parse an uploaded file into a dictionary with:
      'data': a pandas DataFrame (if structured),
      'text': a string (if unstructured),
      'type': a short string for the file type,
      'filename': original filename.
    """
    result = {'type': None, 'data': None, 'text': None, 'filename': file.name}

    if file.name.lower().endswith(".xlsx"):
        result['type'] = 'excel'
        result['data'] = pd.read_excel(file)
    elif file.name.lower().endswith(".csv"):
        result['type'] = 'csv'
        result['data'] = pd.read_csv(file)
    elif file.name.lower().endswith(".json"):
        result['type'] = 'json'
        try:
            file.seek(0)
            json_data = json.load(file)
            # Check if it's a previous export
            if isinstance(json_data, dict) and "__pydantic_extraction_data__" in json_data:
                return {"type": "previous_export", "data": json_data, "filename": file.name}
            
            if isinstance(json_data, list):
                result['data'] = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                result['data'] = pd.DataFrame([json_data])
            else:
                # Not a typical structure
                result['text'] = json.dumps(json_data, indent=2)
        except:
            st.warning("Failed to parse JSON as structured data. Treating as text.")
            file.seek(0)
            result['text'] = file.read().decode('utf-8')
    elif file.name.lower().endswith(".txt"):
        result['type'] = 'text'
        result['text'] = file.read().decode('utf-8')
    elif file.name.lower().endswith(".docx"):
        result['type'] = 'docx'
        result['text'] = docx2txt.process(file)
    elif file.name.lower().endswith(".pdf"):
        result['type'] = 'pdf'
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
            result['text'] = text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
    else:
        st.warning("Unsupported file format. Please upload Excel, CSV, JSON, TXT, DOCX, or PDF.")
    
    return result

def create_or_update_extraction(index, field_values, source_data=None, unique_id=None):
    """
    Stores or updates extraction data in st.session_state["extractions"] at the given index.
    Also keeps the entire row's raw data in `source_data`.
    """
    while len(st.session_state["extractions"]) <= index:
        st.session_state["extractions"].append({})
    
    extraction = {
        "values": field_values,
        "row_index": index,
        "source_data": source_data,  # Entire raw data
        "review_status": "manually_reviewed",  # Status indicator for manual review
        "review_timestamp": datetime.now().isoformat(),  # When it was reviewed
    }
    if unique_id:
        extraction["id"] = str(unique_id)  # Convert to string for consistency
    else:
        extraction["id"] = f"row_{index+1}"
    
    st.session_state["extractions"][index] = extraction

def generate_default_values(model_class):
    """
    Generate default values for a Pydantic model to ensure consistent blank/None values.
    """
    if not model_class or not hasattr(model_class, 'model_fields'):
        return {}
    
    default_values = {}
    
    def get_default_value(field_info):
        """Get the default value for a field based on its type."""
        try:
            field_annotation = field_info.annotation
            is_opt = is_optional_type(field_annotation)
            base_type = get_base_type(field_annotation)
            
            # If optional, default to None
            if is_opt:
                return None
            
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
            
            # Nested pydantic model
            if inspect.isclass(base_type) and issubclass(base_type, BaseModel):
                return generate_default_values(base_type)
            
            # List
            if get_origin(base_type) is list:
                return []
            
            # String (default)
            return ""
            
        except Exception:
            return None
    
    # Generate defaults for all fields
    for field_name, field_info in model_class.model_fields.items():
        default_values[field_name] = get_default_value(field_info)
    
    return default_values

def initialize_all_rows_in_memory():
    """
    Initialize all rows from the loaded data in session state with 'not_reviewed' status.
    This ensures all rows are included in exports, even if not manually reviewed.
    Also generates default values for all fields.
    """
    file_data = st.session_state.get("loaded_file")
    model_class = st.session_state.get("model_class")
    
    if not file_data or file_data.get("data") is None or not model_class:
        return
    
    df = file_data["data"]
    total_rows = len(df)
    
    # Generate default values for the model
    default_values = generate_default_values(model_class)
    
    # Extend extractions list to match all rows
    while len(st.session_state["extractions"]) < total_rows:
        st.session_state["extractions"].append({})
    
    # Initialize each row with source data and not_reviewed status
    for i in range(total_rows):
        if not st.session_state["extractions"][i]:  # Only initialize if empty
            row_data = df.iloc[i].to_dict()
            
            # Generate ID from ID column value, not row number
            id_col = st.session_state.get("id_column")
            if id_col and id_col in row_data:
                # Use actual ID column value (convert to string), even if it's falsy (0, empty string, etc.)
                unique_id = str(row_data[id_col])
            else:
                unique_id = f"row_{i+1}"  # Fallback to row number only if ID column doesn't exist
            
            # Initialize with default values in the same nested structure as the model
            # This will be flattened during export, maintaining consistency
            st.session_state["extractions"][i] = {
                "values": default_values.copy(),  # Use nested default values (will be flattened on export)
                "row_index": i,
                "source_data": row_data,
                "review_status": "not_reviewed",  # Status indicator
                "review_timestamp": None,  # No review timestamp yet
                "id": unique_id  # Use proper ID from data
            }

def init_session_states():
    """Initialize session state variables for the app."""
    if "extractions" not in st.session_state:
        st.session_state["extractions"] = [] 
    if "current_row_index" not in st.session_state:
        st.session_state["current_row_index"] = 0
    if "model_class" not in st.session_state:
        st.session_state["model_class"] = None
    if "loaded_file" not in st.session_state:
        st.session_state["loaded_file"] = None
    if "extracted_count" not in st.session_state:
        st.session_state["extracted_count"] = 0
    if "model_code_str" not in st.session_state:
        st.session_state["model_code_str"] = ""
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = ""
    if "id_column" not in st.session_state:
        st.session_state["id_column"] = ""
    if "text_column" not in st.session_state:
        st.session_state["text_column"] = ""
    if "extraction_dashboard_columns_height" not in st.session_state:
        st.session_state["extraction_dashboard_columns_height"] = 600
    if "row_selection_temp" not in st.session_state:
        st.session_state["row_selection_temp"] = 1
    if "row_selection_input" not in st.session_state:
        st.session_state["row_selection_input"] = 1
    if "color_index" not in st.session_state:
        st.session_state["color_index"] = 0
    if "id_column_warning" not in st.session_state:
        st.session_state["id_column_warning"] = None
    if "extraction_initialized" not in st.session_state:
        st.session_state["extraction_initialized"] = False

    # For available model names
    if "available_model_names" not in st.session_state:
        st.session_state["available_model_names"] = []

def serialize_model_code():
    """Serialize the user’s pydantic model code and name for embedding in export JSON."""
    return {
        "model_code": st.session_state.get("model_code_str", ""),
        "model_name": st.session_state.get("model_name", "")
    }

def is_optional_type(field_type):
    """Check if a field type is Optional[...]"""
    origin = get_origin(field_type)
    if origin is Union:
        args = get_args(field_type)
        return type(None) in args
    return False

def get_base_type(field_type):
    """Return the underlying type of an Optional[...] type, else the given type."""
    if is_optional_type(field_type):
        args = get_args(field_type)
        for arg in args:
            if arg is not type(None):
                return arg
    return field_type

def format_source_data_as_markdown(source_data):
    """Convert source_data dictionary to a formatted markdown string."""
    if not source_data:
        return "*No source data available*"
    
    markdown = "### Source Data\n\n"
    
    for key, value in source_data.items():
        # Handle different data types appropriately
        if isinstance(value, str) and len(value) > 100:
            # For long text fields, format as a block
            markdown += f"**{key}**:\n```\n{value}\n```\n\n"
        else:
            # For shorter values, show inline
            markdown += f"**{key}**: {value}\n\n"
    
    return markdown

def flatten_for_export(obj, prefix="", separator=flatten_for_export_SEPARATOR):
    """Recursively flatten a nested structure into a dictionary with concatenated keys."""
    result = {}
    # Safe separator that can be parsed later
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}{separator}{k}" if prefix else k
            if isinstance(v, (dict, list)):
                result.update(flatten_for_export(v, new_key))
            else:
                result[new_key] = v
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_key = f"{prefix}{separator}{i}"
            if isinstance(item, (dict, list)):
                result.update(flatten_for_export(item, new_key))
            else:
                result[new_key] = item
    return result

def unflatten_from_export(flattened_dict, separator=flatten_for_export_SEPARATOR):
    """
    Reconstruct a nested structure from a flattened dictionary.
    Reverse operation of flatten_for_export.
    """
    result = {}
    
    for key, value in flattened_dict.items():
        if separator in key:
            # Split the key into parts
            parts = key.split(separator)
            current = result
            
            # Navigate through the nested structure
            for i, part in enumerate(parts[:-1]):
                if part.isdigit():
                    # This is a list index
                    index = int(part)
                    # Ensure current is a list
                    if not isinstance(current, list):
                        current = []
                    # Extend list if needed
                    while len(current) <= index:
                        current.append({})
                    current = current[index]
                else:
                    # This is a dictionary key
                    if part not in current:
                        # Look ahead to see if next part is a digit (indicating a list)
                        next_part = parts[i + 1] if i + 1 < len(parts) else None
                        if next_part and next_part.isdigit():
                            current[part] = []
                        else:
                            current[part] = {}
                    current = current[part]
            
            # Set the final value
            final_key = parts[-1]
            if final_key.isdigit():
                index = int(final_key)
                if not isinstance(current, list):
                    current = []
                while len(current) <= index:
                    current.append(None)
                current[index] = value
            else:
                current[final_key] = value
        else:
            # Simple key without separator
            result[key] = value
    
    return result
    
# =====================================================
# 2) COLOR & STYLED CONTAINER HELPERS
# =====================================================

def lighten_color(color_hex: str, percentage: float) -> str:
    """
    Lighten the given hex color by the specified percentage (0.0 to 1.0).
    E.g., lighten_color("#CCDDFF", 0.15) => 15% lighter color.
    """
    color_hex = color_hex.lstrip('#')
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)

    # Increase each channel by the given percentage, up to 255
    r = int(r + (255 - r) * percentage)
    g = int(g + (255 - g) * percentage)
    b = int(b + (255 - b) * percentage)

    return f"#{r:02x}{g:02x}{b:02x}"

def get_next_base_color():
    """
    Return the next color in the palette sequentially (instead of random),
    cycling through if needed.
    """
    idx = st.session_state["color_index"] % len(COLOR_PALETTE)
    color = COLOR_PALETTE[idx]
    st.session_state["color_index"] += 1
    return color

def styled_container(unique_key: str, bg_color: str = "#FFFFFF"):
    """
    Creates a container with a unique key so it picks up a class name `.st-key-<unique_key>`.
    Then we inject a <style> block that targets the `.st-key-<unique_key>` class to apply a
    background color, border, etc.
    """
    container = st.container(key=unique_key)
    style_block = f"""
    <style>
    .st-key-{unique_key} {{
        background-color: {bg_color} !important;
        border: 1px solid #888 !important;
        border-radius: 4px !important;
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }}
    </style>
    """
    container.markdown(style_block, unsafe_allow_html=True)
    return container

# =====================================================
# 3) LIST CALLBACK FOR SIMPLE ITEMS
# =====================================================

def add_simple_item_callback(prefix: str, field_name: str, item_type: Any):
    """
    Callback to add a simple (non-nested) item to the list in session state,
    then clear the text input.
    """
    new_simple_key = f"{prefix}{field_name}_new_item"
    list_key = f"{prefix}{field_name}_items"

    new_val_str = st.session_state.get(new_simple_key, "").strip()
    if not new_val_str:
        st.warning("No item was entered.")
        return

    try:
        if item_type == int:
            new_val = int(new_val_str)
        elif item_type == float:
            new_val = float(new_val_str)
        elif item_type == bool:
            lv = new_val_str.lower()
            if lv in ["true", "t", "1", "yes", "y"]:
                new_val = True
            elif lv in ["false", "f", "0", "no", "n"]:
                new_val = False
            else:
                new_val = None
        else:
            new_val = new_val_str
    except Exception:
        st.error(f"Invalid input for type {item_type}.")
        return
    
    if list_key not in st.session_state:
        st.session_state[list_key] = []
    st.session_state[list_key].append(new_val)
    st.session_state[new_simple_key] = ""

# =====================================================
# 4) NESTED FIELDS RENDERING
# =====================================================

def render_top_level_field(field_name, field_info, current_value, prefix=""):
    """
    Renders a field from the MAIN Pydantic model, wrapped or standard.
    """
    field_annotation = field_info.annotation
    field_description = field_info.description or ""
    is_opt = is_optional_type(field_annotation)
    base_type = get_base_type(field_annotation)

    # Mark fields that have a non-None value with ✅
    label_core = f"{field_name} (Optional)" if is_opt else field_name
    if current_value is not None:
        label_core += " ✅"

    # If it's a nested object => use an expander
    if inspect.isclass(base_type) and issubclass(base_type, BaseModel):
        if not current_value or not isinstance(current_value, dict):
            current_value = {}
        model_name = base_type.__name__
        with st.expander(f"{label_core} ({model_name})", expanded=False):
            st.caption(field_description)
            base_color = get_next_base_color()
            nested_vals = render_nested_object(
                pyd_model_class=base_type,
                current_values=current_value,
                prefix=f"{prefix}{field_name}_",
                depth=1,
                base_color=base_color
            )
        return nested_vals
    else:
        return render_nested_field(
            field_name=field_name,
            field_info=field_info,
            current_value=current_value,
            prefix=prefix,
            depth=0,
            base_color="#FFFFFF"
        )

def render_nested_object(pyd_model_class, current_values, prefix, depth, base_color):
    """
    Renders fields of a nested pydantic object in a styled_container.
    """
    color_for_level = lighten_color(base_color, depth * BRIGHTER_COLOR_RATE)
    container_key = f"nested-{prefix}-depth{depth}"
    model_name = pyd_model_class.__name__


    with styled_container(container_key, color_for_level):
        depth_marker = "#" * (depth + 1)  # Create '#' characters based on depth
        st.markdown(f"----- {depth_marker} **{model_name}** -----")
        new_vals = {}
        for f_name, f_info in pyd_model_class.model_fields.items():
            c_val = current_values.get(f_name)
            new_vals[f_name] = render_nested_field(
                field_name=f_name,
                field_info=f_info,
                current_value=c_val,
                prefix=prefix,
                depth=depth,
                base_color=base_color
            )
    return new_vals

def render_list_advance(label_core, field_description, current_value, key_base, depth, base_color, item_type):
    """
    Advanced renderer for list fields.
    - Initializes the list in session state if not already present.
    - Shows a button for adding a new empty object.
    - Renders each list item inside a container with a 22% lighter background per item.
    """
    list_key = f"{key_base}_list"
    # Initialize list in session state
    if list_key not in st.session_state:
        if isinstance(current_value, list):
            st.session_state[list_key] = current_value
        else:
            st.session_state[list_key] = []
    items = st.session_state[list_key]

    st.markdown(f"**{label_core}**")
    if field_description:
        st.caption(field_description)

    # Add New Item button (shows at the top)
    if st.button(f"Add {item_type.__name__} Object", key=f"{list_key}_add_btn"):
        # For nested Pydantic models, start with an empty dict; for simple types, use an empty string.
        if inspect.isclass(item_type) and issubclass(item_type, BaseModel):
            items.append({})
        else:
            items.append("")
        st.session_state[list_key] = items

    # Render each item in the list
    for idx, item in enumerate(items):
        item_key = f"{list_key}_obj{idx}"
        # Each successive item's container is 22% brighter (adjusted by depth and index)
        container_color = lighten_color(base_color, (depth + idx) * 0.22)
        with styled_container(item_key, container_color):
            st.markdown(f"**Item {idx + 1} ({item_type.__name__})**")
            if inspect.isclass(item_type) and issubclass(item_type, BaseModel):
                # Render a nested object; if not already a dict, initialize as empty dict.
                if not isinstance(item, dict):
                    item = {}
                updated_item = render_nested_object(
                    pyd_model_class=item_type,
                    current_values=item,
                    prefix=f"{list_key}_{idx}_",
                    depth=depth + 1,
                    base_color=container_color
                )
                items[idx] = updated_item
            else:
                # For simple types, render a text input.
                new_val = st.text_input("Item", value=str(item) if item is not None else "", key=f"{item_key}_val")
                items[idx] = new_val
    return items

def render_nested_field(field_name, field_info, current_value, prefix, depth, base_color):
    """
    Renders a single field (bool, int, float, string, nested object, or list).
    Adds a "✅" suffix to the label if current_value is not None.
    """
    field_annotation = field_info.annotation
    field_description = field_info.description or ""
    is_opt = is_optional_type(field_annotation)
    base_type = get_base_type(field_annotation)

    # Build label, add check mark if not None
    label_core = f"{field_name} (Optional)" if is_opt else field_name
    if current_value is not None:
        label_core += " ✅"
    key_base = f"{prefix}{field_name}"
    
    # Enums
    if inspect.isclass(base_type) and issubclass(base_type, Enum):
        enum_values = [e.value for e in base_type]
        options = ["(None)"] + enum_values
        # Check session state first, then fallback to current_value
        session_value = st.session_state.get(key_base, current_value)
        if session_value not in enum_values:
            session_value = None
        index = options.index(session_value) if session_value in options else 0
        if len(enum_values) <= 5:
            val = st.radio(label_core, options, index=index, help=field_description, key=key_base)
        else:
            val = st.selectbox(label_core, options, index=index, help=field_description, key=key_base)
        return None if val == "(None)" else val

    # Literal
    if get_origin(base_type) is Literal:
        literal_values = get_args(base_type)
        options = ["(None)"] + list(literal_values)
        # Check session state first, then fallback to current_value
        session_value = st.session_state.get(key_base, current_value)
        if session_value not in literal_values:
            session_value = None
        index = options.index(session_value) if session_value in options else 0
        if len(literal_values) <= 5:
            val = st.radio(label_core, options, index=index, help=field_description, key=key_base)
        else:
            val = st.selectbox(label_core, options, index=index, help=field_description, key=key_base)
        return None if val == "(None)" else val

    # Booleans
    if base_type == bool:
        bool_options = ["(None)", "True", "False"]
        # Check session state first, then fallback to current_value
        session_value = st.session_state.get(key_base, None)
        if session_value is None:
            if current_value is True:
                session_value = "True"
            elif current_value is False:
                session_value = "False"
            else:
                session_value = "(None)"
        
        if session_value == "True":
            selected_idx = 1
        elif session_value == "False":
            selected_idx = 2
        else:
            selected_idx = 0
        chosen = st.radio(label_core, bool_options, index=selected_idx, help=field_description, key=key_base)
        if chosen == "(None)":
            return None
        elif chosen == "True":
            return True
        else:
            return False

    # Int
    if base_type == int:
        if is_opt:
            # Let user pick None or a number
            modes = ["(None)", "Number"]
            # Check session state first for mode - prioritize user selection
            session_mode = st.session_state.get(key_base + "_mode", None)
            if session_mode is not None:
                # User has made a selection, use it
                mode_index = 1 if session_mode == "Number" else 0
            else:
                # No user selection yet, use current_value to determine default
                mode_index = 1 if (current_value is not None) else 0
            
            choice = st.radio(label_core, modes, index=mode_index, help=field_description, key=key_base + "_mode")
            
            # Use session state directly instead of relying on radio return value to avoid timing issues
            actual_choice = st.session_state.get(key_base + "_mode", choice)
            
            if actual_choice == "(None)":
                return None
            else:
                # Check session state first for value
                session_value = st.session_state.get(key_base, None)
                if session_value is None:
                    default_val = 0 if current_value is None else int(current_value)
                else:
                    default_val = int(session_value)
                val = st.number_input(
                    label_core + " (int)",
                    value=default_val,
                    step=1,
                    help=field_description,
                    key=key_base
                )
                return val
        else:
            # Check session state first for value
            session_value = st.session_state.get(key_base, None)
            if session_value is None:
                default_val = 0 if current_value is None else int(current_value)
            else:
                default_val = int(session_value)
            val = st.number_input(
                label_core,
                value=default_val,
                step=1,
                help=field_description,
                key=key_base
            )
            return val

    # Float
    if base_type == float:
        if is_opt:
            modes = ["(None)", "Number"]
            # Check session state first for mode - prioritize user selection
            session_mode = st.session_state.get(key_base + "_mode", None)
            if session_mode is not None:
                # User has made a selection, use it
                mode_index = 1 if session_mode == "Number" else 0
            else:
                # No user selection yet, use current_value to determine default
                mode_index = 1 if (current_value is not None) else 0
            
            choice = st.radio(label_core, modes, index=mode_index, help=field_description, key=key_base + "_mode")
            
            # Use session state directly instead of relying on radio return value to avoid timing issues
            actual_choice = st.session_state.get(key_base + "_mode", choice)
            
            if actual_choice == "(None)":
                return None
            else:
                # Check session state first for value
                session_value = st.session_state.get(key_base, None)
                if session_value is None:
                    default_val = 0.0 if current_value is None else float(current_value)
                else:
                    default_val = float(session_value)
                val = st.number_input(
                    label_core + " (float)",
                    value=default_val,
                    step=1.0,
                    help=field_description,
                    key=key_base
                )
                return val
        else:
            # Check session state first for value
            session_value = st.session_state.get(key_base, None)
            if session_value is None:
                default_val = 0.0 if current_value is None else float(current_value)
            else:
                default_val = float(session_value)
            val = st.number_input(
                label_core,
                value=default_val,
                step=1.0,
                help=field_description,
                key=key_base
            )
            return val

    # Nested pydantic
    if inspect.isclass(base_type) and issubclass(base_type, BaseModel):
        model_name = base_type.__name__
        st.markdown(f"**{label_core} ({model_name})**")
        if field_description:
            st.caption(field_description)
        if not current_value or not isinstance(current_value, dict):
            current_value = {}
        return render_nested_object(
            pyd_model_class=base_type,
            current_values=current_value,
            prefix=f"{key_base}_",
            depth=depth+1,
            base_color=base_color
        )

    # List 
    if get_origin(base_type) is list:
        item_type = get_args(base_type)[0]
        return render_list_advance(label_core, field_description, current_value, key_base, depth, base_color, item_type) 

    # string
    # Check session state first for value
    session_value = st.session_state.get(key_base, None)
    if session_value is None:
        default_val = str(current_value) if current_value is not None else ""
    else:
        default_val = str(session_value)
    
    # Decide whether to use text_area or text_input based on field name
    if field_name.lower() in LONG_TEXT_FIELD_LIST:
        val = st.text_area(label_core, value=default_val, height=68, help=field_description, key=key_base)
        return val
    else:
        val = st.text_input(label_core, value=default_val, help=field_description, key=key_base)
        return val

def process_main_model_fields(model_class, current_values, prefix=""):
    """
    Renders all fields in the main Pydantic model.
    """
    result_vals = {}
    st.session_state["color_index"] = 0  # reset color index
    for fn, fi in model_class.model_fields.items():
        cur_val = current_values.get(fn)
        result_vals[fn] = render_top_level_field(fn, fi, cur_val, prefix)
    return result_vals

def gather_values_from_state(model_class, prefix=""):
    """
    Recursively gather values from st.session_state for the given model_class fields.
    """
    def get_value(field_name, field_info, prefix):
        base_t = get_base_type(field_info.annotation)
        k_mode = f"{prefix}{field_name}_mode"   # For optional numeric radio
        k_val = f"{prefix}{field_name}"         # For actual input

        # Enum or literal
        if inspect.isclass(base_t) and issubclass(base_t, Enum):
            val = st.session_state.get(k_val, None)
            return None if val == "(None)" else val
        if get_origin(base_t) is Literal:
            val = st.session_state.get(k_val, None)
            return None if val == "(None)" else val

        # Bool
        if base_t == bool:
            chosen = st.session_state.get(k_val, None)
            if chosen == "(None)":
                return None
            elif chosen == "True":
                return True
            elif chosen == "False":
                return False
            return None

        # int / float
        if base_t == int or base_t == float:
            # if optional => check if user picked (None)
            mode_val = st.session_state.get(k_mode, None)
            if mode_val == "(None)":
                return None
            return st.session_state.get(k_val, None)

        # Nested pydantic
        if inspect.isclass(base_t) and issubclass(base_t, BaseModel):
            sub_obj = {}
            for nf, nf_info in base_t.model_fields.items():
                sub_obj[nf] = get_value(nf, nf_info, f"{prefix}{field_name}_")
            return sub_obj

        # List
        if get_origin(base_t) is list:
            list_key = f"{prefix}{field_name}_items"
            return st.session_state.get(list_key, [])

        # String
        return st.session_state.get(k_val, "")

    out = {}
    for f_name, f_info in model_class.model_fields.items():
        out[f_name] = get_value(f_name, f_info, prefix)
    return out

def append_llm_output_to_memory(llm_data, separator=flatten_for_export_SEPARATOR):
    """
    Append LLM output to memory, linking it via an 'id' column.
    The input can be JSON or Excel (flattened using the separator).
    """
    try:
        # Flatten the LLM data if it's a nested structure
        flattened_llm_data = flatten_for_export(llm_data, separator=separator)

        # Ensure the 'id' column exists in the LLM data
        if 'id' not in flattened_llm_data:
            st.error("LLM data must contain an 'id' column.")
            return

        llm_id = flattened_llm_data['id']

        # Check if the 'id' matches any existing row in session_state
        for extraction in st.session_state.get("extractions", []):
            if extraction.get("id") == llm_id:
                # Append LLM data as defaults without overwriting manual extractions
                extraction.setdefault("llm_values", {}).update(flattened_llm_data)
                st.success(f"LLM output appended to memory for ID: {llm_id}")
                return

        # If no match found, add a new entry for the LLM data
        st.session_state["extractions"].append({
            "id": llm_id,
            "llm_values": flattened_llm_data,
            "values": {},  # Empty manual extraction values
        })
        st.success(f"New LLM output added to memory for ID: {llm_id}")
    except Exception as e:
        st.error(f"Error appending LLM output to memory: {e}")
        st.error(traceback.format_exc())

def generate_export_data():
    """
    Generate the complete export data including all rows (reviewed and not reviewed)
    with status indicators and timestamps.
    """
    extractions = st.session_state["extractions"]
    extracted_data = []
    
    for i, extraction in enumerate(extractions):
        # Include ALL rows, whether reviewed or not
        if extraction:  # Only if extraction exists (not empty dict)
            vals = extraction.get("values", {})
            raw_data = extraction.get("source_data", {})
            review_status = extraction.get("review_status", "not_reviewed")
            review_timestamp = extraction.get("review_timestamp")
            
            # Flatten extracted fields recursively
            flat = flatten_for_export(vals)
            
            # Add index and ID
            # Ensure ID is first key in dict and convert to string to avoid type conflicts
            id_val = extraction.get("id", f"row_{i+1}")
            new_flat = {"id": str(id_val), "row_index": extraction.get("row_index", i)}
            
            # Add review status information
            new_flat["review_status"] = review_status
            new_flat["review_timestamp"] = review_timestamp
            
            # Add flattened extraction values
            for k, v in flat.items():
                if k != "id":
                    new_flat[k] = v
            
            # Add raw data with raw_ prefix
            for rk, rv in raw_data.items():
                new_flat[f"raw_{rk}"] = rv
            
            extracted_data.append(new_flat)
    
    return extracted_data

# =====================================================
# 5) CALLBACKS
# =====================================================

def load_pydantic_code():
    """
    Parse pydantic code from text_area or uploaded .py,
    store all discovered model names in st.session_state["available_model_names"].
    """
    code_str = st.session_state.get("pydantic_model_code", "").strip()
    if not code_str and st.session_state.get("uploaded_py", None):
        code_str = st.session_state["uploaded_py"].read().decode("utf-8")

    if not code_str:
        st.warning("No code provided to parse.")
        return

    st.session_state["model_code_str"] = code_str
    found_classes = load_model_code(code_str)
    if not found_classes:
        st.warning("No valid Pydantic models found in code.")
        st.session_state["available_model_names"] = []
        st.session_state["model_class"] = None
        return

    # Store the discovered classes in session for the dropdown
    st.session_state["available_model_names"] = list(found_classes.keys())
    # If user had an existing chosen name, keep it if valid
    if st.session_state["model_name"] in found_classes:
        st.session_state["model_class"] = found_classes[st.session_state["model_name"]]
    else:
        # default to the last model in code
        last_name = st.session_state["available_model_names"][-1]
        st.session_state["model_name"] = last_name
        st.session_state["model_class"] = found_classes[last_name]

def on_model_select_change():
    """
    Called when user picks a new model name from the dropdown.
    """
    code_str = st.session_state.get("model_code_str", "")
    if not code_str:
        return
    found_classes = load_model_code(code_str)
    chosen = st.session_state.get("model_name_select", "")
    if chosen in found_classes:
        st.session_state["model_name"] = chosen
        st.session_state["model_class"] = found_classes[chosen]
        st.session_state["extraction_initialized"] = False  # Reset extraction initialization flag

def upload_data_source():
    """
    Parse the newly uploaded data file. (Excel, CSV, text, etc.)
    """
    uploaded_file = st.session_state.get("data_file", None)
    if not uploaded_file:
        return

    fdata = parse_uploaded_file(uploaded_file)
    st.session_state["loaded_file"] = fdata
    if fdata["type"] in ("excel", "csv") and fdata["data"] is not None:
        st.success(f"✅ Loaded structured data from {fdata['filename']}")
        # Reset extractions for new file
        st.session_state["extractions"] = []
        st.session_state["current_row_index"] = 0
        st.session_state["extracted_count"] = 0
        st.session_state["form_state_initialized"] = False  # Reset form state initialization flag
        st.session_state["extraction_initialized"] = False  # Reset extraction initialization flag
        # Set form state to defaults when loading new data
        if st.session_state.get("model_class"):
            default_values = generate_default_values(st.session_state["model_class"])
            set_form_session_state_values(st.session_state["model_class"], default_values)
    elif fdata["text"] is not None:
        st.success(f"✅ Loaded text from {fdata['filename']}")
    else:
        st.warning("No structured data or text could be parsed.")

def validate_model_values(model_class, values):
    """
    Validate user-provided values against the model_class.
    """
    try:
        model_instance = model_class(**values)
        return True, None, model_instance
    except Exception as e:
        return False, str(e), None

def save_extraction_callback():
    """
    Gather the current row's field values and save them to st.session_state,
    then move to next row if possible.
    """
    if not st.session_state["model_class"]:
        st.warning("No model loaded.")
        return
    
    row_index = st.session_state["current_row_index"]
    model_class = st.session_state["model_class"]
    extracted_values = gather_values_from_state(model_class, prefix="")

    file_data = st.session_state["loaded_file"]
    source_data = {}
    if file_data and file_data["data"] is not None:
        df = file_data["data"]
        if row_index < len(df):
            source_data = df.iloc[row_index].to_dict()
    else:
        if file_data and file_data.get("text"):
            source_data = {"text": file_data["text"]}

    unique_id = None
    id_col = st.session_state.get("id_column")
    if id_col and source_data and id_col in source_data:
        unique_id = source_data[id_col]
    
    create_or_update_extraction(row_index, extracted_values, source_data, unique_id)
    st.session_state["extracted_count"] += 1
    st.success(f"Row {row_index+1}: Extraction Saved.")

    if file_data and file_data["data"] is not None:
        if row_index + 1 < len(file_data["data"]):
            st.session_state["current_row_index"] = row_index + 1
            # Mark that form state needs to be reset for the new row
            st.session_state["form_state_initialized"] = False
            # Set form state to match the target row's values
            if st.session_state.get("model_class") and st.session_state.get("extractions"):
                new_idx = row_index + 1
                target_vals = {}
                if len(st.session_state["extractions"]) > new_idx and st.session_state["extractions"][new_idx]:
                    target_vals = st.session_state["extractions"][new_idx].get("values", {})
                set_form_session_state_values(st.session_state["model_class"], target_vals)

def restore_previous_export(export_data):
    """
    Restore session states from a previously exported JSON structure
    and rebuild data so user can continue extraction.
    """
    try:
        # Check if it's a complete session state export
        if "extractions" in export_data or "extractions" in export_data.get("data", {}):
            # Support both direct and nested under 'data'
            data_source = export_data if "extractions" in export_data else export_data["data"]

            # Process key session state variables
            for key in [
                "extractions", "model_code_str", "model_name", "id_column", 
                "text_column", "current_row_index", "extracted_count", 
                "color_index", "available_model_names"
            ]:
                if key in data_source:
                    # Avoid ambiguous truth value for pandas Series
                    val = data_source[key]
                    if not (isinstance(val, pd.Series) or isinstance(val, pd.DataFrame)):
                        st.session_state[key] = val
                    elif isinstance(val, pd.Series) and not val.empty:
                        st.session_state[key] = val
                    elif isinstance(val, pd.DataFrame) and not val.empty:
                        st.session_state[key] = val

            # Re-load model code, if present
            if "model_code_str" in data_source and isinstance(data_source["model_code_str"], str) and data_source["model_code_str"]:
                # Parse it for classes
                found = load_model_code(data_source["model_code_str"])
                st.session_state["available_model_names"] = list(found.keys())
                # Pick the model_name from JSON, if valid
                if data_source.get("model_name", "") in found:
                    st.session_state["model_name"] = data_source["model_name"]
                    st.session_state["model_class"] = found[data_source["model_name"]]
                elif st.session_state["available_model_names"]:
                    # Default to last
                    last_name = st.session_state["available_model_names"][-1]
                    st.session_state["model_name"] = last_name
                    st.session_state["model_class"] = found[last_name]

            # Rebuild a DataFrame from source_data
            all_source_data = []
            for ex in st.session_state["extractions"]:
                if isinstance(ex, dict) and "source_data" in ex:
                    sd = ex.get("source_data", {})
                    if sd:
                        all_source_data.append(sd)

            new_df = None
            if all_source_data:
                new_df = pd.DataFrame(all_source_data)

            # Load file info
            filename = data_source.get("loaded_file", {}).get("filename", "previous_session.json")
            file_type = data_source.get("loaded_file", {}).get("type", "unknown")
            
            st.session_state["loaded_file"] = {
                "filename": filename,
                "type": file_type,
                "data": new_df,
                "text": None
            }
            
            # Handle unstructured text case
            if (new_df is None or new_df.empty) and len(all_source_data) > 0:
                if "text" in all_source_data[0]:
                    st.session_state["loaded_file"]["text"] = all_source_data[0]["text"]

            # Ensure backward compatibility: add review_status to old extractions
            if "extractions" in st.session_state:
                for extraction in st.session_state["extractions"]:
                    if isinstance(extraction, dict) and "review_status" not in extraction:
                        if extraction.get("values"):  # If has values, assume it was manually reviewed
                            extraction["review_status"] = "manually_reviewed"
                            extraction["review_timestamp"] = None  # Unknown timestamp for old data
                        else:
                            extraction["review_status"] = "not_reviewed"
                            extraction["review_timestamp"] = None

            return True
        else:
            st.error("Invalid export format (extractions missing).")
            return False
    except Exception as e:
        st.error(f"Error restoring previous export: {e}")
        st.error(traceback.format_exc())
        return False

# Modify session state export to exclude unserializable objects
def safe_serialize_session_state():
    state_copy = copy.deepcopy(dict(st.session_state))
    # Remove unserializable objects
    for key in list(state_copy.keys()):
        val = state_copy[key]
        if callable(val) or isinstance(val, (st.runtime.scriptrunner.ScriptRunContext,)):
            del state_copy[key]
        elif isinstance(val, (pd.DataFrame, pd.Series)):
            # Convert DataFrame/Series to JSON string
            state_copy[key] = val.to_json()
        elif isinstance(val, bytes):
            # Convert bytes to string
            state_copy[key] = val.decode('utf-8', errors='ignore')
    return json.dumps(state_copy, default=str)

# =====================================================
# 6) SESSION STATE MANAGEMENT
# =====================================================

def reset_session_state_for_new_patient():
    """
    Resolve the issue where previous patient data incorrectly persists across new patient sessions.
    """
    keys_to_reset = [
        "extractions", "current_row_index", "extracted_count", "loaded_file",
        "model_code_str", "model_name", "id_column", "text_column"
    ]
    for key in keys_to_reset:
        st.session_state[key] = None
    st.success("Session state reset for new patient.")

# =====================================================
# 7) TEXT RENDERING FIX
# =====================================================

def render_long_text_field(field_name, text):
    """
    Investigate and correct the text rendering problem for long unstructured text inputs.
    """
    try:
        if len(text) > 1000:  # Arbitrary threshold for long text
            st.text_area(field_name, value=text, height=300)
        else:
            st.text_area(field_name, value=text)
    except Exception as e:
        st.error(f"Error rendering text field '{field_name}': {e}")
        st.error(traceback.format_exc())

# =====================================================
# 8) MAIN APP
# =====================================================

def check_setup_requirements():
    """
    Check if all required components are loaded and configured.
    Returns True if ready to start extraction, False otherwise.
    """
    # Check if Pydantic model is loaded
    if not st.session_state.get("model_class"):
        return False
    
    # Check if data file is loaded
    if not st.session_state.get("loaded_file"):
        return False
    
    # Check if structured data requires column selection
    file_data = st.session_state["loaded_file"]
    if file_data.get("data") is not None and not file_data["data"].empty:
        # For structured data, require ID and text column selection
        if not st.session_state.get("id_column"):
            return False
        if not st.session_state.get("text_column"):
            return False
    
    return True

def show_setup_requirements():
    """
    Display the setup requirements and current status.
    """
    st.warning("⚠️ Please complete the setup requirements before starting extraction:")
    
    # Check Pydantic model
    if st.session_state.get("model_class"):
        st.success("✅ Pydantic model loaded: " + st.session_state.get("model_name", "Unknown"))
    else:
        st.error("❌ Please load a Pydantic model in the sidebar:")
        st.info("   1. Paste your Pydantic model code in the text area OR upload a .py file")
        st.info("   2. Click 'Parse Pydantic Code' button")
        st.info("   3. Select the model from the dropdown")
    
    # Check data file
    if st.session_state.get("loaded_file"):
        file_data = st.session_state["loaded_file"]
        st.success(f"✅ Data file loaded: {file_data.get('filename', 'Unknown')}")
        
        # For structured data, check column selection
        if file_data.get("data") is not None and not file_data["data"].empty:
            df = file_data["data"]
            st.info(f"   📊 Structured data with {len(df)} rows and {len(df.columns)} columns")
            
            # Check ID column
            if st.session_state.get("id_column"):
                st.success(f"✅ ID column selected: {st.session_state['id_column']}")
            else:
                st.error("❌ Please select an ID column in the sidebar:")
                st.info("   1. Go to 'Column Selection' section")
                st.info("   2. Choose the 'Unique ID Column' from the dropdown")
                st.info(f"   3. Available columns: {list(df.columns)}")
            
            # Check text column
            if st.session_state.get("text_column"):
                st.success(f"✅ Text column selected: {st.session_state['text_column']}")
            else:
                st.error("❌ Please select a text column in the sidebar:")
                st.info("   1. Go to 'Column Selection' section")
                st.info("   2. Choose the 'Text/Content Column' from the dropdown")
                st.info(f"   3. Available columns: {list(df.columns)}")
    else:
        st.error("❌ Please upload a data file in the sidebar:")
        st.info("   1. Go to 'Data Source' section")
        st.info("   2. Upload a CSV, Excel, JSON, TXT, DOCX, or PDF file")

def initialize_extraction_session():
    """
    Initialize the extraction session with proper ID column configuration.
    """
    try:
        # Initialize all rows with the selected ID column
        initialize_all_rows_in_memory()
        
        # Mark extraction as initialized
        st.session_state["extraction_initialized"] = True
        
        # Initialize form state for the current row
        st.session_state["form_state_initialized"] = False  # Reset to trigger re-initialization
        
        # Show success message
        file_data = st.session_state["loaded_file"]
        total_rows = len(file_data["data"]) if file_data.get("data") is not None else 1
        id_col = st.session_state.get("id_column", "row numbers")
        
        st.success(f"🎉 Extraction session initialized!")
        st.info(f"📊 {total_rows} rows initialized with ID column: {id_col}")
        
        # Auto-rerun to show the extraction interface
        st.rerun()
        
    except Exception as e:
        st.error(f"Error initializing extraction session: {e}")
        st.error("Please check your setup and try again.")

def main():
    st.set_page_config(
        page_title="Pydantic Extraction App",
        page_icon="📋",
        layout="wide",
    )
    init_session_states()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SIDEBAR: SESSION SETUP
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with st.sidebar:
        st.title("Session Setup")

        # Pydantic Model Setup
        st.subheader("Pydantic Model Setup")
        st.text_area(
            "Paste your Pydantic model code here:",
            height=150,
            value=st.session_state.get("model_code_str", ""),
            key="pydantic_model_code"
        )
        st.file_uploader(
            "Or upload a Python file (.py) with your model:",
            type=["py"],
            key="uploaded_py"
        )
        if st.button("Parse Pydantic Code"):
            load_pydantic_code()

        # If we found classes, let user pick from them
        if st.session_state["available_model_names"]:
            default_idx = len(st.session_state["available_model_names"]) - 1
            if st.session_state["model_name"] in st.session_state["available_model_names"]:
                default_idx = st.session_state["available_model_names"].index(st.session_state["model_name"])
            
            picked = st.selectbox(
                "Select Pydantic Model",
                st.session_state["available_model_names"],
                index=default_idx,
                key="model_name_select",
                on_change=on_model_select_change
            )

        st.markdown("---")
        st.subheader("Data Source")
        st.file_uploader(
            "Upload data file (Excel, CSV, JSON, TXT, DOCX, PDF):",
            type=["xlsx", "csv", "json", "txt", "docx", "pdf"],
            key="data_file",
            on_change=upload_data_source
        )

        # If structured data loaded, column selection
        loaded_info = st.session_state.get("loaded_file", {})
        if loaded_info and loaded_info.get("data") is not None:
            df = loaded_info["data"]
            if not df.empty:
                available_cols = [""] + list(df.columns)
                st.markdown("**Column Selection**")
                # Ensure 'Unique ID Column' is displayed first in the UI
                idx_id = available_cols.index("id") if "id" in available_cols else 0
                st.selectbox(
                    "Unique ID Column (optional):",
                    available_cols,
                    index=idx_id,
                    help="Optional unique ID column",
                    key="id_column_select",
                    on_change=update_id_column
                )

                cur_txt_col = st.session_state.get("text_column", "")
                idx_txt = available_cols.index(cur_txt_col) if cur_txt_col in available_cols else 0
                st.selectbox(
                    "Text/Content Column:",
                    available_cols,
                    index=idx_txt,
                    help="Column containing the text for extraction",
                    key="text_column_select",
                    on_change=update_text_column
                )

        st.markdown("---")
        st.subheader("Previous Extractions")
        
        # Upload previous extractions JSON
        prev_extractions_file = st.file_uploader(
            "Upload previous extractions JSON:",
            type=["json"],
            key="previous_extractions_file",
            help="Upload a JSON file exported from a previous manual extraction session"
        )
        
        if st.button("🔄 Inject Previous Extractions", type="secondary"):
            inject_previous_extractions(prev_extractions_file)

        st.markdown("---")
        st.title("🔧 UI Configuration")
        st.subheader("Column Heights")
        st.session_state["extraction_dashboard_columns_height"] = st.number_input(
            "Evaluation Column Height (px)",
            min_value=300,
            max_value=2000,
            value=st.session_state["extraction_dashboard_columns_height"],
            step=50
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MAIN AREA: EXTRACTION DASHBOARD + REVIEW/EXPORT
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.title("📋 Pydantic Extraction Dashboard")

    # 1) Check if all required components are loaded and configured
    setup_complete = check_setup_requirements()
    
    if not setup_complete:
        show_setup_requirements()
        return

    # 2) Show "Start Extraction" button if not already initialized
    if not st.session_state.get("extraction_initialized", False):
        st.success("✅ All requirements met! Ready to start extraction.")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Start Extraction", type="primary"):
                initialize_extraction_session()
        
        with col2:
            st.info("This will initialize all rows with the selected ID column and prepare the extraction interface.")
        
        return

    # 3) Data Nav + Extraction (only shown after initialization)
    file_data = st.session_state["loaded_file"]
    model_class = st.session_state["model_class"]
    
    # Show ID column warning if exists
    if st.session_state.get("id_column_warning"):
        st.warning(st.session_state["id_column_warning"])

    # Initialize form state for the current row if not already done
    # This prevents overwriting user input during the same session
    if not st.session_state.get("form_state_initialized", False):
        st.session_state["form_state_initialized"] = True
        row_index = st.session_state["current_row_index"]
        current_vals = {}
        if len(st.session_state["extractions"]) > row_index:
            ex = st.session_state["extractions"][row_index]
            if isinstance(ex, dict) and "values" in ex:
                current_vals = ex["values"]
        # Clear previous form state before initializing new row
        clear_form_session_state(model_class, prefix="")
        set_form_session_state_values(model_class, current_vals, prefix="")

    # If there's structured data, show row nav
    if file_data["data"] is not None and not file_data["data"].empty:
        df = file_data["data"]
        total_rows = len(df)
        st.subheader("Data Navigation")

        # Calculate progress based on manual review status
        reviewed_count = len([e for e in st.session_state["extractions"] if e and e.get("review_status") == "manually_reviewed"])
        st.progress(reviewed_count / total_rows, f"Manually Reviewed {reviewed_count} of {total_rows} rows")

        nav_cols = st.columns([1,1,1,1,2])
        with nav_cols[0]:
            st.button("\u276e Previous", on_click=previous_extraction, disabled=(st.session_state["current_row_index"] == 0))
        with nav_cols[1]:
            st.button("Next \u276f", on_click=next_extraction, disabled=(st.session_state["current_row_index"] >= total_rows - 1))
        with nav_cols[2]:
            st.markdown(f"**Row {st.session_state['current_row_index']+1} of {total_rows}**")
        with nav_cols[3]:
            st.button("Jump to \u279c", on_click=set_row_selection)
        with nav_cols[4]:
            st.number_input(
                "Row index to jump (1{})".format(total_rows),
                min_value=1,
                max_value=total_rows,
                value=st.session_state["row_selection_input"],
                step=1,
                key="row_selection_input",
                label_visibility="collapsed",
                on_change=show_jump_warning
            )
            if st.session_state.get("show_jump_warning", False):
                st.warning("Please press the Jump To button to navigate to the selected row.")
    else:
        st.session_state["current_row_index"] = 0

    row_index = st.session_state["current_row_index"]

    # Layout for Extraction + Source Data
    col_extraction, col_source = st.columns([3,2])
    with col_extraction.container(height=st.session_state["extraction_dashboard_columns_height"]):
        st.markdown(f"#### Extract Data into *{model_class.__name__}*")
        
        # If we already have something saved for this row, load it
        current_vals = {}
        if len(st.session_state["extractions"]) > row_index:
            ex = st.session_state["extractions"][row_index]
            if isinstance(ex, dict) and "values" in ex:
                current_vals = ex["values"]

        # Render the model (widgets will read from session state)
        process_main_model_fields(model_class, current_vals, prefix="")

        # Buttons
        bc1, bc2 = st.columns([1,1])
        with bc1:
            st.button("Save Extraction", on_click=save_extraction_callback, type="primary")
        with bc2:
            # Show current row status
            if len(st.session_state["extractions"]) > row_index and st.session_state["extractions"][row_index]:
                status = st.session_state["extractions"][row_index].get("review_status", "not_reviewed")
                if status == "manually_reviewed":
                    st.success("✅ Reviewed")
                else:
                    st.warning("⏳ Not Reviewed")

    with col_source.container(height=st.session_state["extraction_dashboard_columns_height"]):
        if file_data["data"] is not None and not file_data["data"].empty:
            if row_index < len(file_data["data"]):
                row_dict = file_data["data"].iloc[row_index].to_dict()
                text_col = st.session_state.get("text_column")
                if text_col and text_col in row_dict:
                    st.markdown("**Content for Extraction**:")
                    st.markdown(
                        f"<div style='background-color:#F9F9F9; padding:0.5rem;'>{row_dict[text_col]}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(format_source_data_as_markdown(row_dict))
                id_col = st.session_state.get("id_column")
                if id_col and id_col in row_dict:
                    st.info(f"**ID**: {row_dict[id_col]}")
        else:
            # Unstructured text scenario
            if file_data.get("text"):
                st.markdown("**Content for Extraction**:")
                st.markdown(
                    f"<div style='background-color:#F9F9F9; padding:0.5rem;'>{file_data['text']}</div>", 
                    unsafe_allow_html=True
                )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # REVIEW & EXPORT (at the bottom)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.subheader("Review & Export")
    extractions = st.session_state["extractions"]
    if not extractions:
        st.info("No extractions yet.")
        return

    tab_data, tab_json = st.tabs(["Data View", "JSON View"])
    with tab_data:
        extracted_data = generate_export_data()

        if extracted_data:
            df_extractions = pd.DataFrame(extracted_data)
            
            # Ensure DataFrame is compatible with PyArrow for Streamlit display
            df_extractions = ensure_dataframe_arrow_compatibility(df_extractions)
            
            # Add legend for color coding
            st.markdown("**Legend:** 🟢 Green = Manually Reviewed | 🔴 Red = Not Reviewed")
            
            # Add color coding for review status in the display
            def highlight_review_status(row):
                if row['review_status'] == 'manually_reviewed':
                    return ['background-color: #d4edda'] * len(row)  # Light green for reviewed
                elif row['review_status'] == 'not_reviewed':
                    return ['background-color: #f8d7da'] * len(row)  # Light red for not reviewed
                else:
                    return [''] * len(row)
            
            st.dataframe(
                df_extractions.style.apply(highlight_review_status, axis=1),
                use_container_width=True
            )
            
            # Show summary statistics
            total_rows = len(df_extractions)
            reviewed_rows = len(df_extractions[df_extractions['review_status'] == 'manually_reviewed'])
            not_reviewed_rows = total_rows - reviewed_rows
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", total_rows)
            with col2:
                st.metric("Reviewed", reviewed_rows, delta=f"{reviewed_rows/total_rows*100:.1f}%")
            with col3:
                st.metric("Not Reviewed", not_reviewed_rows, delta=f"{not_reviewed_rows/total_rows*100:.1f}%")
                
        else:
            st.info("No data to display. Please ensure data is loaded and initialize all rows.")

    with tab_json:
        extracted_data = generate_export_data()
        if extracted_data:
            json_str = json.dumps(extracted_data, indent=2)
            st.code(json_str, language="json")
        else:
            st.info("No data to display. Please ensure data is loaded and initialize all rows")        

    # Export Buttons
    ec1, ec2 = st.columns([1,1])
    export_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with ec1:
        extracted_data = generate_export_data()
        json_str = json.dumps(extracted_data, indent=2)
        st.download_button(
            "Download Extractions (JSON)",
            data=json_str,
            file_name=f"extractions_{export_time_stamp}.json",
            mime="application/json"
        )
        
    with ec2:
        extracted_data = generate_export_data()
        if extracted_data:
            df_for_csv = ensure_dataframe_arrow_compatibility(pd.DataFrame(extracted_data))
            csv_str = df_for_csv.to_csv(index=False)
            st.download_button(
                "Download Extractions (CSV)",
                data=csv_str,
                file_name=f"extractions_{export_time_stamp}.csv",
                mime="text/csv"
            )
            
def export_extractions_to_csv():
    """
    Exports the extractions to a CSV file with the Unique ID Column as the first column.
    """
    file_data = st.session_state.get("loaded_file")
    if file_data and file_data["data"] is not None:
        df = file_data["data"]
        if "id" in df.columns:
            # Ensure 'id' column is the first column
            df = df[["id"] + [col for col in df.columns if col != "id"]]
        df.to_csv("extractions.csv", index=False)
        st.success("Extractions exported successfully.")

def update_id_column():
    """
    Updates the session state for the ID column based on user selection.
    Also checks for uniqueness of values in the selected column.
    """
    st.session_state["id_column"] = st.session_state["id_column_select"]
    st.session_state["extraction_initialized"] = False  # Reset extraction initialization flag
    
    # Check uniqueness of UID column values
    if st.session_state["id_column"] and st.session_state.get("loaded_file"):
        file_data = st.session_state["loaded_file"]
        if file_data.get("data") is not None:
            try:
                df = file_data["data"]
                id_col = st.session_state["id_column"]
                if id_col in df.columns:
                    # Check for duplicates
                    duplicate_values = df[df[id_col].duplicated()][id_col].unique()
                    if len(duplicate_values) > 0:
                        # Store warning in session state to avoid blocking
                        st.session_state["id_column_warning"] = f"Duplicate values found in ID column '{id_col}': {list(duplicate_values)}"
                    else:
                        # Clear warning if no duplicates
                        st.session_state["id_column_warning"] = None
                else:
                    st.session_state["id_column_warning"] = None
            except Exception as e:
                # Handle errors gracefully
                st.session_state["id_column_warning"] = f"Error checking ID column uniqueness: {str(e)}"

def update_text_column():
    """
    Updates the session state for the text/content column based on user selection.
    """
    st.session_state["text_column"] = st.session_state["text_column_select"]
    st.session_state["extraction_initialized"] = False  # Reset extraction initialization flag

def clear_form_session_state(model_class, prefix=""):
    """
    Clear form field session state keys for the given model_class.
    This prevents previous row values from appearing when navigating to a new row.
    """
    if not model_class or not hasattr(model_class, 'model_fields'):
        return
        
    def clear_field_keys(field_name, field_info, prefix):
        try:
            base_t = get_base_type(field_info.annotation)
            k_mode = f"{prefix}{field_name}_mode"   # For optional numeric radio
            k_val = f"{prefix}{field_name}"         # For actual input
            
            # Clear the main field key
            if k_val in st.session_state:
                del st.session_state[k_val]
            
            # Clear the mode key for optional numeric fields
            if k_mode in st.session_state:
                del st.session_state[k_mode]
            
            # Handle nested models
            if inspect.isclass(base_t) and hasattr(base_t, 'model_fields'):
                for nested_fn, nested_fi in base_t.model_fields.items():
                    nested_prefix = f"{prefix}{field_name}."
                    clear_field_keys(nested_fn, nested_fi, nested_prefix)
            
            # Handle lists - clear individual item keys
            if get_origin(base_t) is list:
                # Clear any existing list item keys
                keys_to_delete = [key for key in st.session_state.keys() if key.startswith(f"{prefix}{field_name}_")]
                for key in keys_to_delete:
                    del st.session_state[key]
        except Exception as e:
            # Silent error handling to prevent crashes during form clearing
            pass
    
    # Clear all fields in the model
    try:
        for fn, fi in model_class.model_fields.items():
            clear_field_keys(fn, fi, prefix)
    except Exception as e:
        # Silent error handling to prevent crashes during form clearing
        pass

def set_form_session_state_values(model_class, values, prefix=""):
    """
    Set form field session state keys to match the provided values.
    This ensures widgets display the correct values for the current row.
    MUST be called BEFORE rendering widgets.
    """
    if not model_class or not hasattr(model_class, 'model_fields'):
        return
        
    def set_field_values(field_name, field_info, field_value, prefix):
        try:
            base_t = get_base_type(field_info.annotation)
            is_opt = is_optional_type(field_info.annotation)
            k_mode = f"{prefix}{field_name}_mode"   # For optional numeric radio
            k_val = f"{prefix}{field_name}"         # For actual input
            
            # Handle different field types
            if inspect.isclass(base_t) and issubclass(base_t, Enum):
                # Enum fields
                if field_value in [e.value for e in base_t]:
                    st.session_state[k_val] = field_value
                else:
                    if k_val not in st.session_state:
                        st.session_state[k_val] = "(None)"
                    
            elif get_origin(base_t) is Literal:
                # Literal fields
                literal_values = get_args(base_t)
                if field_value in literal_values:
                    st.session_state[k_val] = field_value
                else:
                    if k_val not in st.session_state:
                        st.session_state[k_val] = "(None)"
                    
            elif base_t == bool:
                # Boolean fields
                if field_value is True:
                    st.session_state[k_val] = "True"
                elif field_value is False:
                    st.session_state[k_val] = "False"
                else:
                    if k_val not in st.session_state:
                        st.session_state[k_val] = "(None)"
                    
            elif base_t == int:
                # Integer fields
                if is_opt:
                    if field_value is not None:
                        st.session_state[k_mode] = "Number"
                        st.session_state[k_val] = int(field_value)
                    else:
                        # Only set if not already set (don't overwrite user selections)
                        if k_mode not in st.session_state:
                            st.session_state[k_mode] = "(None)"
                        if k_val not in st.session_state:
                            st.session_state[k_val] = 0
                else:
                    if k_val not in st.session_state:
                        st.session_state[k_val] = int(field_value) if field_value is not None else 0
                    
                    
            elif base_t == float:
                # Float fields
                if is_opt:
                    if field_value is not None:
                        st.session_state[k_mode] = "Number"
                        st.session_state[k_val] = float(field_value)
                    else:
                        # Only set if not already set (don't overwrite user selections)
                        if k_mode not in st.session_state:
                            st.session_state[k_mode] = "(None)"
                        if k_val not in st.session_state:
                            st.session_state[k_val] = 0.0
                else:
                    if k_val not in st.session_state:
                        st.session_state[k_val] = float(field_value) if field_value is not None else 0.0
                    
            elif inspect.isclass(base_t) and issubclass(base_t, BaseModel):
                # Nested pydantic model
                nested_values = field_value if isinstance(field_value, dict) else {}
                set_form_session_state_values(base_t, nested_values, f"{prefix}{field_name}_")
                
            elif get_origin(base_t) is list:
                # List fields
                list_key = f"{prefix}{field_name}_items"
                if isinstance(field_value, list):
                    st.session_state[list_key] = field_value.copy()
                else:
                    if list_key not in st.session_state:
                        st.session_state[list_key] = []
                    
            else:
                # String and other fields
                if k_val not in st.session_state:
                    st.session_state[k_val] = str(field_value) if field_value is not None else ""
                
        except Exception as e:
            # Silent error handling to prevent crashes
            pass
    
    # Set values for all fields in the model
    try:
        for fn, fi in model_class.model_fields.items():
            field_value = values.get(fn)
            set_field_values(fn, fi, field_value, prefix)
    except Exception as e:
        # Silent error handling to prevent crashes
        pass

def previous_extraction():
    """
    Navigate to the previous row for extraction.
    """
    current_idx = st.session_state["current_row_index"]
    if current_idx > 0:
        st.session_state["current_row_index"] = current_idx - 1
        # Mark that form state needs to be reset for the new row
        st.session_state["form_state_initialized"] = False
        # Clear previous form state before initializing new row
        if st.session_state.get("model_class") and st.session_state.get("extractions"):
            clear_form_session_state(st.session_state["model_class"], prefix="")
            new_idx = current_idx - 1
            target_vals = {}
            if len(st.session_state["extractions"]) > new_idx and st.session_state["extractions"][new_idx]:
                target_vals = st.session_state["extractions"][new_idx].get("values", {})
            set_form_session_state_values(st.session_state["model_class"], target_vals)

def next_extraction():
    """
    Navigate to the next row for extraction.
    """
    current_idx = st.session_state["current_row_index"]
    file_data = st.session_state["loaded_file"]
    if file_data and file_data.get("data") is not None:
        df = file_data["data"]
        if current_idx < len(df) - 1:
            st.session_state["current_row_index"] = current_idx + 1
            # Mark that form state needs to be reset for the new row
            st.session_state["form_state_initialized"] = False
            # Clear previous form state before initializing new row
            if st.session_state.get("model_class") and st.session_state.get("extractions"):
                clear_form_session_state(st.session_state["model_class"], prefix="")
                new_idx = current_idx + 1
                target_vals = {}
                if len(st.session_state["extractions"]) > new_idx and st.session_state["extractions"][new_idx]:
                    target_vals = st.session_state["extractions"][new_idx].get("values", {})
                set_form_session_state_values(st.session_state["model_class"], target_vals)

def set_row_selection():
    """
    Reads the user numeric input for row selection and sets current_row_index.
    Displays a warning box until the Jump To button is pressed.
    """
    row_sel = st.session_state["row_selection_input"]
    file_data = st.session_state["loaded_file"]
    if file_data and file_data.get("data") is not None:
        df_len = len(file_data["data"])
        if 1 <= row_sel <= df_len:
            st.session_state["current_row_index"] = row_sel - 1
            st.session_state["show_jump_warning"] = False
            # Mark that form state needs to be reset for the new row
            st.session_state["form_state_initialized"] = False
            # Clear previous form state before initializing new row
            if st.session_state.get("model_class") and st.session_state.get("extractions"):
                clear_form_session_state(st.session_state["model_class"], prefix="")
                new_idx = row_sel - 1
                target_vals = {}
                if len(st.session_state["extractions"]) > new_idx and st.session_state["extractions"][new_idx]:
                    target_vals = st.session_state["extractions"][new_idx].get("values", {})
                set_form_session_state_values(st.session_state["model_class"], target_vals)
        else:
            st.warning("Invalid row selection.")

def show_jump_warning():
    """
    Displays a warning box prompting the user to press the Jump To button.
    """
    st.session_state["show_jump_warning"] = True


def inject_previous_extractions(uploaded_file):
    """
    Inject previous extractions from a JSON file into the current session.
    Matches UIDs and replaces extracted values and review status for matching rows.
    """
    if not uploaded_file:
        st.warning("Please upload a JSON file containing previous extractions.")
        return
    
    if not st.session_state.get("loaded_file") or not st.session_state.get("model_class"):
        st.warning("Please load data and select a Pydantic model first before injecting previous extractions.")
        return
    
    try:
        # Parse the uploaded JSON file
        file_data = parse_uploaded_file(uploaded_file)
        
        # Check if it's a previous export or regular JSON file
        if file_data["type"] == "previous_export":
            # This is a previous export file with the correct structure
            json_data = file_data["data"]
        elif file_data["type"] == "json":
            # This is a regular JSON file, check if data exists
            if file_data.get("data") is None:
                st.error("Invalid JSON file. The file appears to be empty or corrupted.")
                return
            # Convert DataFrame back to JSON-like structure if needed
            json_data = file_data["data"]
        else:
            st.error("Invalid file format. Please upload a valid JSON file.")
            return
        
        # Handle different JSON structures
        if isinstance(json_data, pd.DataFrame):
            # DataFrame from regular JSON file - convert to list of dicts
            previous_extractions = json_data.to_dict('records')
        elif isinstance(json_data, list):
            # Direct list of extraction records
            previous_extractions = json_data
        elif isinstance(json_data, dict) and "extractions" in json_data:
            # Session state export format
            previous_extractions = json_data["extractions"]
        elif isinstance(json_data, dict) and "__pydantic_extraction_data__" in json_data:
            # Previous export format - extract extractions list
            previous_extractions = json_data.get("extractions", [])
        else:
            st.error("Unsupported JSON structure. Please upload a valid extraction export file.")
            return
        
        if not previous_extractions:
            st.warning("No previous extractions found in the uploaded file.")
            return
        
        # Get current session info
        current_extractions = st.session_state.get("extractions", [])
        id_column = st.session_state.get("id_column", "")
        injected_count = 0
        matched_count = 0
        
        # Create a mapping of previous extractions by ID
        previous_by_id = {}
        for prev_extraction in previous_extractions:
            if isinstance(prev_extraction, dict):
                # Handle different ID field formats
                extraction_id = None
                if "id" in prev_extraction:
                    extraction_id = prev_extraction["id"]
                elif id_column and f"raw_{id_column}" in prev_extraction:
                    extraction_id = prev_extraction[f"raw_{id_column}"]
                elif "row_index" in prev_extraction:
                    extraction_id = f"row_{prev_extraction['row_index'] + 1}"
                
                if extraction_id:
                    previous_by_id[str(extraction_id)] = prev_extraction  # Convert to string for consistency
        
        # Match and inject previous extractions
        for i, current_extraction in enumerate(current_extractions):
            if not isinstance(current_extraction, dict):
                continue
                
            current_id = str(current_extraction.get("id", f"row_{i+1}"))
            
            if current_id in previous_by_id:
                matched_count += 1
                prev_data = previous_by_id[current_id]
                
                # Separate flattened extraction data from metadata
                flattened_extraction_data = {}
                metadata = {}
                
                for key, value in prev_data.items():
                    # Skip null values and empty strings for extraction data
                    if key.startswith(("id", "row_index", "review_status", "review_timestamp", "raw_")):
                        metadata[key] = value
                    else:
                        # Only include non-null, non-empty values in extraction data
                        if value is not None and value != "":
                            flattened_extraction_data[key] = value
                
                # Reconstruct nested structure from flattened data
                if flattened_extraction_data:
                    try:
                        reconstructed_values = unflatten_from_export(flattened_extraction_data)
                        
                        # Update the current extraction with reconstructed data
                        current_extraction["values"] = reconstructed_values
                        current_extraction["review_status"] = metadata.get("review_status", "manually_reviewed")
                        current_extraction["review_timestamp"] = metadata.get("review_timestamp")
                        injected_count += 1
                        
                    except Exception as e:
                        st.warning(f"Failed to reconstruct data for ID {current_id}: {e}")
                        # Fallback: try to use the data as-is if it's already in the right format
                        if any(key not in ["id", "row_index", "review_status", "review_timestamp"] and not key.startswith("raw_") for key in prev_data.keys()):
                            # Filter out metadata and use remaining data
                            filtered_data = {k: v for k, v in prev_data.items() 
                                           if not k.startswith(("id", "row_index", "review_status", "review_timestamp", "raw_")) 
                                           and v is not None and v != ""}
                            if filtered_data:
                                current_extraction["values"] = filtered_data
                                current_extraction["review_status"] = metadata.get("review_status", "manually_reviewed")
                                current_extraction["review_timestamp"] = metadata.get("review_timestamp")
                                injected_count += 1
        
        # Update session state
        st.session_state["extractions"] = current_extractions
        
        # Reset form state to reflect the changes
        st.session_state["form_state_initialized"] = False
        
        # Show results
        if injected_count > 0:
            st.success(f"✅ Successfully injected {injected_count} previous extractions (matched {matched_count} rows by ID)")
            # Update extracted count
            manually_reviewed = len([e for e in current_extractions if e and e.get("review_status") == "manually_reviewed"])
            st.session_state["extracted_count"] = manually_reviewed
        else:
            if matched_count > 0:
                st.warning(f"⚠️ Found {matched_count} matching IDs but no valid extraction data to inject.")
            else:
                st.warning("⚠️ No matching IDs found between current data and previous extractions.")
        
    except Exception as e:
        st.error(f"Error injecting previous extractions: {e}")
        st.error(traceback.format_exc())

def ensure_dataframe_arrow_compatibility(df):
    """
    Ensure DataFrame is compatible with PyArrow for Streamlit display.
    Fixes mixed types and other compatibility issues.
    """
    if df is None or df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Convert mixed type columns to string
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            # Check if column has mixed types
            try:
                # Try to convert to string to ensure consistency
                df_copy[col] = df_copy[col].astype(str)
            except Exception:
                # If that fails, handle null values
                df_copy[col] = df_copy[col].fillna('').astype(str)
    
    return df_copy

if __name__ == "__main__":
    # Initialize session state variables
    init_session_states()

    # Run the main app
    main()

    # If user has not pressed Jump To, show warning
    if st.session_state.get("show_jump_warning", False):
        st.warning("Please press the Jump To button to navigate to the selected row.")
