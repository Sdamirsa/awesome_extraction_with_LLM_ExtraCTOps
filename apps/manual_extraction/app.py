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
    - []
"""



import streamlit as st
import pandas as pd
import json
import traceback
import inspect
from datetime import datetime
from enum import Enum
from typing import get_type_hints, get_origin, get_args, Dict, List, Optional, Literal, Union, Any
from pydantic import BaseModel, Field
import docx2txt
import PyPDF2

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
    }
    if unique_id:
        extraction["id"] = unique_id
    else:
        extraction["id"] = f"row_{index+1}"
    
    st.session_state["extractions"][index] = extraction

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

    # For controlling session type: initiate new or continue previous
    if "session_type" not in st.session_state:
        st.session_state["session_type"] = "Initiate New"
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
        if current_value not in enum_values:
            current_value = None
        index = options.index(current_value) if current_value in options else 0
        if len(enum_values) <= 5:
            val = st.radio(label_core, options, index=index, help=field_description, key=key_base)
        else:
            val = st.selectbox(label_core, options, index=index, help=field_description, key=key_base)
        return None if val == "(None)" else val

    # Literal
    if get_origin(base_type) is Literal:
        literal_values = get_args(base_type)
        options = ["(None)"] + list(literal_values)
        if current_value not in literal_values:
            current_value = None
        index = options.index(current_value) if current_value in options else 0
        if len(literal_values) <= 5:
            val = st.radio(label_core, options, index=index, help=field_description, key=key_base)
        else:
            val = st.selectbox(label_core, options, index=index, help=field_description, key=key_base)
        return None if val == "(None)" else val

    # Booleans
    if base_type == bool:
        bool_options = ["(None)", "True", "False"]
        if current_value is True:
            selected_idx = 1
        elif current_value is False:
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
            mode_index = 1 if (current_value is not None) else 0
            choice = st.radio(label_core, modes, index=mode_index, help=field_description, key=key_base + "_mode")
            if choice == "(None)":
                return None
            else:
                default_val = 0 if current_value is None else int(current_value)
                val = st.number_input(
                    label_core + " (int)",
                    value=default_val,
                    step=1,
                    help=field_description,
                    key=key_base
                )
                return val
        else:
            default_val = 0 if current_value is None else int(current_value)
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
            mode_index = 1 if (current_value is not None) else 0
            choice = st.radio(label_core, modes, index=mode_index, help=field_description, key=key_base + "_mode")
            if choice == "(None)":
                return None
            else:
                default_val = 0.0 if current_value is None else float(current_value)
                val = st.number_input(
                    label_core + " (float)",
                    value=default_val,
                    step=1.0,
                    help=field_description,
                    key=key_base
                )
                return val
        else:
            default_val = 0.0 if current_value is None else float(current_value)
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
    default_val = str(current_value) if current_value is not None else ""
    # Decide whether to use text_area or text_input based on field name
    if field_name.lower() in LONG_TEXT_FIELD_LIST:
        val = st.text_area(label_core, value=default_val, height=60, help=field_description, key=key_base)
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

def validate_extraction():
    """
    Validate the current row's fields against the Pydantic model.
    """
    if not st.session_state["model_class"]:
        st.warning("No model loaded.")
        return
    model_class = st.session_state["model_class"]
    extracted_values = gather_values_from_state(model_class, prefix="")
    valid, errs, _ = validate_model_values(model_class, extracted_values)
    if valid:
        st.success("Validation successful!")
    else:
        st.error(f"Validation failed: {errs}")

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

def restore_previous_export(export_data):
    """
    Restore session states from a previously exported JSON structure
    and rebuild data so user can continue extraction.
    """
    try:
        # Check if it's a complete session state export
        if "extractions" in export_data:
            # Process key session state variables
            for key in [
                "extractions", "model_code_str", "model_name", "id_column", 
                "text_column", "current_row_index", "extracted_count", 
                "color_index", "available_model_names"
            ]:
                if key in export_data:
                    st.session_state[key] = export_data[key]
            
            # Re-load model code, if present
            if "model_code_str" in export_data and export_data["model_code_str"]:
                # Parse it for classes
                found = load_model_code(export_data["model_code_str"])
                st.session_state["available_model_names"] = list(found.keys())
                # Pick the model_name from JSON, if valid
                if export_data.get("model_name", "") in found:
                    st.session_state["model_name"] = export_data["model_name"]
                    st.session_state["model_class"] = found[export_data["model_name"]]
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
            filename = export_data.get("loaded_file", {}).get("filename", "previous_session.json")
            file_type = export_data.get("loaded_file", {}).get("type", "unknown")
            
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

            return True
        else:
            st.error("Invalid export format (extractions missing).")
            return False
    except Exception as e:
        st.error(f"Error restoring previous export: {e}")
        st.error(traceback.format_exc())
        return False

# =====================================================
# 6) NAVIGATION CALLBACKS
# =====================================================

def previous_extraction():
    """
    Goes to the previous row (if possible).
    """
    current_idx = st.session_state["current_row_index"]
    if current_idx > 0:
        st.session_state["current_row_index"] = current_idx - 1

def next_extraction():
    """
    Goes to the next row (if possible).
    """
    current_idx = st.session_state["current_row_index"]
    file_data = st.session_state["loaded_file"]
    if file_data and file_data["data"] is not None:
        df = file_data["data"]
        if current_idx < len(df) - 1:
            st.session_state["current_row_index"] = current_idx + 1

def set_row_selection():
    """
    Reads the user numeric input for row selection, sets current_row_index.
    """
    row_sel = st.session_state["row_selection_input"]
    file_data = st.session_state["loaded_file"]
    if file_data and file_data["data"] is not None:
        df_len = len(file_data["data"])
        if 1 <= row_sel <= df_len:
            st.session_state["current_row_index"] = row_sel - 1

def update_id_column():
    st.session_state["id_column"] = st.session_state["id_column_select"]

def update_text_column():
    st.session_state["text_column"] = st.session_state["text_column_select"]

# =====================================================
# 7) MAIN APP
# =====================================================

def main():
    st.set_page_config(
        page_title="Pydantic Extraction App",
        page_icon="📋",
        layout="wide",
    )
    init_session_states()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SIDEBAR: CHOOSE SESSION TYPE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with st.sidebar:
        st.title("Session Setup")

        # Radio for new vs previous
        st.radio(
            "Session Type",
            ["Initiate New", "Continue Previous"],
            key="session_type"
        )

        st.markdown("---")
        if st.session_state["session_type"] == "Continue Previous":
            # Show a JSON loader
            st.subheader("Load Previous JSON")
            prev_json = st.file_uploader(
                "Load previous extraction session (JSON):",
                type=["json"],
                key="previous_export"
            )
            if prev_json is not None:
                file_data = parse_uploaded_file(prev_json)
                if file_data['type'] == 'previous_export':
                    if restore_previous_export(file_data['data']):
                        st.success("Previous session restored.")
                else:
                    st.error("Not a valid previous session JSON.")
        else:
            # "Initiate New" => show pydantic code + data input
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
                    cur_id_col = st.session_state.get("id_column", "")
                    idx_id = available_cols.index(cur_id_col) if cur_id_col in available_cols else 0
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

    # 1) If no model or data loaded, instruct user
    if not st.session_state["model_class"] or not st.session_state["loaded_file"]:
        st.info("Please select 'Initiate New' or 'Continue Previous' in the sidebar to load data and a model.")
        return

    # 2) Data Nav + Extraction
    file_data = st.session_state["loaded_file"]
    model_class = st.session_state["model_class"]

    # If there's structured data, show row nav
    if file_data["data"] is not None and not file_data["data"].empty:
        df = file_data["data"]
        total_rows = len(df)
        st.subheader("Data Navigation")

        completed = len([e for e in st.session_state["extractions"] if e])
        st.progress(completed / total_rows, f"Extracted {completed} of {total_rows} rows")

        nav_cols = st.columns([1,1,1,1,2])
        with nav_cols[0]:
            st.button("❮ Previous", on_click=previous_extraction, disabled=(st.session_state["current_row_index"] == 0))
        with nav_cols[1]:
            st.button("Next ❯", on_click=next_extraction, disabled=(st.session_state["current_row_index"] >= total_rows - 1))
        with nav_cols[2]:
            st.markdown(f"**Row {st.session_state['current_row_index']+1} of {total_rows}**")
        with nav_cols[3]:
            st.button("Jump to ➜", on_click=set_row_selection)
        with nav_cols[4]:
            st.number_input(
                "Row index to jump (1–{})".format(total_rows),
                min_value=1,
                max_value=total_rows,
                value=st.session_state["current_row_index"] + 1,
                step=1,
                key="row_selection_input",
                label_visibility="collapsed"
            )
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

        # Render the model
        process_main_model_fields(model_class, current_vals, prefix="")

        # Buttons
        bc1, bc2, bc3 = st.columns([1,1,1])
        with bc1:
            st.button("Save Extraction", on_click=save_extraction_callback, type="primary")
        with bc2:
            st.button("Validate", on_click=validate_extraction)
        with bc3:
            st.write("")

    with col_source.container(height=st.session_state["extraction_dashboard_columns_height"]):
        st.markdown("#### Source Data")
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
        extracted_data = []
        for i, extraction in enumerate(extractions):
            if extraction:
                vals = extraction.get("values", {})
                raw_data = extraction.get("source_data", {})
                
                # Flatten extracted fields recursively
                flat = flatten_for_export(vals)
                
                # Add index and ID
                flat["row_index"] = extraction.get("row_index", i)
                flat["id"] = extraction.get("id", f"row_{i+1}")
                
                # Add raw data with raw_ prefix
                for rk, rv in raw_data.items():
                    flat[f"raw_{rk}"] = rv
                
                extracted_data.append(flat)

        if extracted_data:
            df_extractions = pd.DataFrame(extracted_data)
            st.dataframe(df_extractions, use_container_width=True)
        else:
            st.info("No valid extractions to display.")

    with tab_json:
        if extracted_data:
            json_str = json.dumps(extracted_data, indent=2)
            st.code(json_str, language="json")
        else:
            st.info("No valid extractions to display.")        

    # Export Buttons
    ec1, ec2, ec3 = st.columns([2,1,1])
    export_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with ec1:
        session_state_export = json.dumps(dict(st.session_state), default=str)
        st.download_button(
            "Download Complete Session State (JSON)",
            data=session_state_export,
            file_name=f"extraction_state_{export_time_stamp}.json",
            mime="application/json",
            help="Download the entire session state (including all extractions). You can use this to save your incomplete extraction and load it in the future in the app."
        )

    with ec2:
        json_str = json.dumps(extracted_data, indent=2)
        st.download_button(
            "Download Extractions (JSON)",
            data=json_str,
            file_name=f"extractions_{export_time_stamp}.json",
            mime="application/json"
        )
        
    with ec3:
        if extracted_data:
            csv_str = pd.DataFrame(extracted_data).to_csv(index=False)
            st.download_button(
                "Download Extractions (CSV)",
                data=csv_str,
                file_name=f"extractions_{export_time_stamp}.csv",
                mime="text/csv"
            )
            
if __name__ == "__main__":
    main()