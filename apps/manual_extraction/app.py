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
    - 2025-07-24: version 0.2
        1. Bullet-proof state management with proper initialization and cleanup
        2. Enhanced session save/load functionality
        3. Better form state synchronization
        4. Improved error handling and data validation
        5. More consistent data flow and separation of concerns

# To-do: 
    - [ ] Add the functionality to add LLM output to the memory (based on the id column). It should check the ...
    - [X] Fix the json session load
    - [X] Fox minimum of text input from 60 to 68
    - [X] resolve the incorrectly saving previous data for the new patient 
    - [X] The issue with rendering the text (exit code)

"""

import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="Enhanced Pydantic Extraction App",
    page_icon="📋",
    layout="wide",
)

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
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import time

# =====================================================
# 0) CONFIGURATION
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
# 1) STATE MANAGEMENT CLASSES
# =====================================================

@dataclass
class SessionState:
    """Centralized session state management"""
    extractions: List[dict]
    current_row_index: int
    model_class: type
    loaded_file: dict
    extracted_count: int
    model_code_str: str
    model_name: str
    id_column: str
    text_column: str
    extraction_dashboard_columns_height: int
    row_selection_input: int
    color_index: int
    id_column_warning: str
    extraction_initialized: bool
    available_model_names: List[str]
    form_state_initialized: bool
    session_locked: bool
    last_update_timestamp: float

class StateManager:
    """Thread-safe state manager for the application"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._state_version = 0
    
    @contextmanager
    def state_transaction(self):
        """Context manager for atomic state operations"""
        with self._lock:
            try:
                self._state_version += 1
                yield
            except Exception as e:
                st.error(f"State transaction failed: {e}")
                raise
    
    def init_session_states(self):
        """Initialize all session state variables with proper defaults"""
        defaults = {
            "extractions": [],
            "current_row_index": 0,
            "model_class": None,
            "loaded_file": None,
            "extracted_count": 0,
            "model_code_str": "",
            "model_name": "",
            "id_column": "",
            "text_column": "",
            "extraction_dashboard_columns_height": 600,
            "row_selection_input": 1,
            "color_index": 0,
            "id_column_warning": None,
            "extraction_initialized": False,
            "available_model_names": [],
            "form_state_initialized": False,
            "session_locked": False,
            "last_update_timestamp": time.time(),
            "state_version": 0
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def validate_state_consistency(self):
        """Validate that session state is in a consistent state"""
        try:
            # Check critical dependencies
            if st.session_state.get("extraction_initialized", False):
                if not st.session_state.get("model_class"):
                    st.warning("Inconsistent state detected: extraction initialized without model. Resetting...")
                    st.session_state["extraction_initialized"] = False
                    return False
                
                if not st.session_state.get("loaded_file"):
                    st.warning("Inconsistent state detected: extraction initialized without data. Resetting...")
                    st.session_state["extraction_initialized"] = False
                    return False
            
            return True
        except Exception as e:
            st.error(f"State validation failed: {e}")
            return False
    
    def get_current_state_snapshot(self):
        """Get a snapshot of current session state for export"""
        with self._lock:
            try:
                state_data = {}
                for key in st.session_state.keys():
                    val = st.session_state[key]
                    if self._is_serializable(val):
                        state_data[key] = copy.deepcopy(val)
                
                # Add metadata
                state_data["__session_metadata__"] = {
                    "export_timestamp": datetime.now().isoformat(),
                    "state_version": self._state_version,
                    "app_version": "0.2"
                }
                
                return state_data
            except Exception as e:
                st.error(f"Failed to create state snapshot: {e}")
                return None
    
    def restore_state_from_snapshot(self, state_data):
        """Restore session state from a snapshot"""
        with self.state_transaction():
            try:
                # Validate snapshot
                if not isinstance(state_data, dict):
                    raise ValueError("Invalid state data format")
                
                # Clear current form state to prevent conflicts
                self._clear_all_form_state()
                
                # Restore core state variables
                core_keys = [
                    "extractions", "model_code_str", "model_name", "id_column", 
                    "text_column", "current_row_index", "extracted_count", 
                    "color_index", "available_model_names"
                ]
                
                for key in core_keys:
                    if key in state_data:
                        st.session_state[key] = state_data[key]
                
                # Rebuild model class if code exists
                if state_data.get("model_code_str"):
                    self._rebuild_model_from_code(state_data["model_code_str"], state_data.get("model_name"))
                
                # Rebuild data file from extractions
                self._rebuild_data_file_from_extractions()
                
                # Reset initialization flags
                st.session_state["extraction_initialized"] = bool(state_data.get("extractions"))
                st.session_state["form_state_initialized"] = False
                
                return True
                
            except Exception as e:
                st.error(f"Failed to restore state: {e}")
                st.error(traceback.format_exc())
                return False
    
    def _is_serializable(self, obj):
        """Check if an object is JSON serializable"""
        try:
            json.dumps(obj, default=str)
            return True
        except (TypeError, ValueError):
            return False
    
    def _clear_all_form_state(self):
        """Clear all form-related session state keys"""
        keys_to_remove = []
        for key in st.session_state.keys():
            if any(suffix in key for suffix in ["_mode", "_items", "_new_item", "_list", "_val"]):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
    
    def _rebuild_model_from_code(self, code_str, model_name):
        """Rebuild pydantic model from code string"""
        try:
            found_classes = load_model_code(code_str)
            st.session_state["available_model_names"] = list(found_classes.keys())
            
            if model_name and model_name in found_classes:
                st.session_state["model_class"] = found_classes[model_name]
            elif found_classes:
                # Default to first model
                first_name = list(found_classes.keys())[0]
                st.session_state["model_name"] = first_name
                st.session_state["model_class"] = found_classes[first_name]
        except Exception as e:
            st.error(f"Failed to rebuild model: {e}")
    
    def _rebuild_data_file_from_extractions(self):
        """Rebuild data file structure from extractions"""
        try:
            extractions = st.session_state.get("extractions", [])
            if not extractions:
                return
            
            # Extract source data from extractions
            source_data_list = []
            for extraction in extractions:
                if isinstance(extraction, dict) and "source_data" in extraction:
                    source_data = extraction["source_data"]
                    if source_data:
                        source_data_list.append(source_data)
            
            if source_data_list:
                df = pd.DataFrame(source_data_list)
                st.session_state["loaded_file"] = {
                    "filename": "restored_session.csv",
                    "type": "csv",
                    "data": df,
                    "text": None
                }
        except Exception as e:
            st.error(f"Failed to rebuild data file: {e}")

# =====================================================
# 2) UTILITY FUNCTIONS
# =====================================================

def load_model_code(code_str: str):
    """Enhanced model loading with better error handling"""
    try:
        module = types.ModuleType('dynamic_models')
        # Add required imports
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
            if (inspect.isclass(v) and 
                issubclass(v, BaseModel) and 
                v is not BaseModel):
                all_classes[k] = v
        
        if not all_classes:
            raise ValueError("No valid Pydantic models found in code")
        
        return all_classes
    except Exception as e:
        st.error(f"Error loading model code: {e}")
        raise

def parse_uploaded_file(file) -> Dict[str, Any]:
    """Enhanced file parsing with better error handling"""
    result = {'type': None, 'data': None, 'text': None, 'filename': file.name}
    
    try:
        if file.name.lower().endswith(".xlsx"):
            result['type'] = 'excel'
            result['data'] = pd.read_excel(file)
        elif file.name.lower().endswith(".csv"):
            result['type'] = 'csv'
            result['data'] = pd.read_csv(file)
        elif file.name.lower().endswith(".json"):
            result['type'] = 'json'
            file.seek(0)
            json_data = json.load(file)
            
            # Check for session export
            if isinstance(json_data, dict) and "__session_metadata__" in json_data:
                return {"type": "session_export", "data": json_data, "filename": file.name}
            
            # Handle regular JSON
            if isinstance(json_data, list):
                result['data'] = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                result['data'] = pd.DataFrame([json_data])
            else:
                result['text'] = json.dumps(json_data, indent=2)
                
        elif file.name.lower().endswith(".txt"):
            result['type'] = 'text'
            result['text'] = file.read().decode('utf-8')
        elif file.name.lower().endswith(".docx"):
            result['type'] = 'docx'
            result['text'] = docx2txt.process(file)
        elif file.name.lower().endswith(".pdf"):
            result['type'] = 'pdf'
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
            result['text'] = text
        else:
            raise ValueError("Unsupported file format")
            
    except Exception as e:
        st.error(f"Error parsing file {file.name}: {e}")
        raise
    
    return result

def create_or_update_extraction(index, field_values, source_data=None, unique_id=None):
    """Enhanced extraction creation with validation"""
    try:
        # Ensure extractions list is long enough
        while len(st.session_state["extractions"]) <= index:
            st.session_state["extractions"].append({})
        
        # Validate field values
        if not isinstance(field_values, dict):
            raise ValueError("Field values must be a dictionary")
        
        extraction = {
            "values": copy.deepcopy(field_values),
            "row_index": index,
            "source_data": copy.deepcopy(source_data) if source_data else {},
            "review_status": "manually_reviewed",
            "review_timestamp": datetime.now().isoformat(),
            "id": str(unique_id) if unique_id is not None else f"row_{index+1}"
        }
        
        st.session_state["extractions"][index] = extraction
        st.session_state["last_update_timestamp"] = time.time()
        
    except Exception as e:
        st.error(f"Failed to create extraction: {e}")
        raise

def initialize_all_rows_in_memory():
    """Enhanced row initialization with better data consistency"""
    try:
        file_data = st.session_state.get("loaded_file")
        model_class = st.session_state.get("model_class")
        
        if not file_data or not model_class:
            raise ValueError("Missing required data or model")
        
        if file_data.get("data") is None:
            # Handle single text file
            if file_data.get("text"):
                st.session_state["extractions"] = [{
                    "values": generate_default_values(model_class),
                    "row_index": 0,
                    "source_data": {"text": file_data["text"]},
                    "review_status": "not_reviewed",
                    "review_timestamp": None,
                    "id": "single_text"
                }]
            return
        
        df = file_data["data"]
        total_rows = len(df)
        default_values = generate_default_values(model_class)
        id_col = st.session_state.get("id_column")
        
        # Initialize or extend extractions list
        while len(st.session_state["extractions"]) < total_rows:
            st.session_state["extractions"].append({})
        
        # Initialize each row
        for i in range(total_rows):
            if not st.session_state["extractions"][i]:  # Only initialize empty slots
                row_data = df.iloc[i].to_dict()
                
                # Generate unique ID
                if id_col and id_col in row_data:
                    unique_id = str(row_data[id_col])
                else:
                    unique_id = f"row_{i+1}"
                
                st.session_state["extractions"][i] = {
                    "values": copy.deepcopy(default_values),
                    "row_index": i,
                    "source_data": row_data,
                    "review_status": "not_reviewed",
                    "review_timestamp": None,
                    "id": unique_id
                }
        
        st.session_state["last_update_timestamp"] = time.time()
        
    except Exception as e:
        st.error(f"Failed to initialize rows: {e}")
        raise

def generate_default_values(model_class):
    """Enhanced default value generation"""
    if not model_class or not hasattr(model_class, 'model_fields'):
        return {}
    
    def get_default_for_field(field_info):
        try:
            field_annotation = field_info.annotation
            is_opt = is_optional_type(field_annotation)
            base_type = get_base_type(field_annotation)
            
            if is_opt:
                return None
            
            if inspect.isclass(base_type) and issubclass(base_type, Enum):
                return None
            
            if get_origin(base_type) is Literal:
                return None
            
            if base_type == bool:
                return None
            
            if base_type == int:
                return 0
            
            if base_type == float:
                return 0.0
            
            if inspect.isclass(base_type) and issubclass(base_type, BaseModel):
                return generate_default_values(base_type)
            
            if get_origin(base_type) is list:
                return []
            
            return ""
            
        except Exception:
            return None
    
    defaults = {}
    for field_name, field_info in model_class.model_fields.items():
        defaults[field_name] = get_default_for_field(field_info)
    
    return defaults

def is_optional_type(field_type):
    """Check if a field type is Optional[...]"""
    origin = get_origin(field_type)
    if origin is Union:
        args = get_args(field_type)
        return type(None) in args
    return False

def get_base_type(field_type):
    """Return the underlying type of an Optional[...] type"""
    if is_optional_type(field_type):
        args = get_args(field_type)
        for arg in args:
            if arg is not type(None):
                return arg
    return field_type

def lighten_color(color_hex: str, percentage: float) -> str:
    """Lighten hex color by percentage"""
    color_hex = color_hex.lstrip('#')
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    
    r = int(r + (255 - r) * percentage)
    g = int(g + (255 - g) * percentage)
    b = int(b + (255 - b) * percentage)
    
    return f"#{r:02x}{g:02x}{b:02x}"

def get_next_base_color():
    """Get next color from palette"""
    idx = st.session_state["color_index"] % len(COLOR_PALETTE)
    color = COLOR_PALETTE[idx]
    st.session_state["color_index"] += 1
    return color

def styled_container(unique_key: str, bg_color: str = "#FFFFFF"):
    """Create styled container with background color"""
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

def flatten_for_export(obj, prefix="", separator=flatten_for_export_SEPARATOR):
    """Flatten nested structure for export"""
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

# =====================================================
# 3) FORM STATE MANAGEMENT
# =====================================================

class FormStateManager:
    """Enhanced form state management with proper synchronization"""
    
    @staticmethod
    def clear_form_state(model_class, prefix=""):
        """Clear form state for a model with improved key detection"""
        if not model_class or not hasattr(model_class, 'model_fields'):
            return
        
        def clear_field_state(field_name, field_info, current_prefix):
            try:
                base_type = get_base_type(field_info.annotation)
                field_key = f"{current_prefix}{field_name}"
                mode_key = f"{field_key}_mode"
                
                # Clear main field keys
                keys_to_clear = [field_key, mode_key]
                
                # Add list-related keys
                list_keys = [f"{field_key}_items", f"{field_key}_list", f"{field_key}_new_item"]
                keys_to_clear.extend(list_keys)
                
                # Clear the keys
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Handle nested models recursively
                if inspect.isclass(base_type) and issubclass(base_type, BaseModel):
                    nested_prefix = f"{field_key}_"
                    FormStateManager.clear_form_state(base_type, nested_prefix)
                
                # Handle lists with complex items
                if get_origin(base_type) is list:
                    item_type = get_args(base_type)[0]
                    if inspect.isclass(item_type) and issubclass(item_type, BaseModel):
                        # Clear any existing list item states
                        list_pattern = f"{field_key}_list_obj"
                        keys_to_remove = [k for k in st.session_state.keys() if k.startswith(list_pattern)]
                        for key in keys_to_remove:
                            del st.session_state[key]
                
            except Exception as e:
                # Silent handling to prevent cascading errors
                pass
        
        try:
            for field_name, field_info in model_class.model_fields.items():
                clear_field_state(field_name, field_info, prefix)
        except Exception as e:
            st.error(f"Error clearing form state: {e}")
    
    @staticmethod
    def set_form_state(model_class, values, prefix=""):
        """Set form state values with proper type handling"""
        if not model_class or not hasattr(model_class, 'model_fields'):
            return
        
        def set_field_state(field_name, field_info, field_value, current_prefix):
            try:
                base_type = get_base_type(field_info.annotation)
                is_optional = is_optional_type(field_info.annotation)
                field_key = f"{current_prefix}{field_name}"
                mode_key = f"{field_key}_mode"
                
                # Handle different field types
                if inspect.isclass(base_type) and issubclass(base_type, Enum):
                    enum_values = [e.value for e in base_type]
                    if field_value in enum_values:
                        st.session_state[field_key] = field_value
                    else:
                        st.session_state[field_key] = "(None)"
                
                elif get_origin(base_type) is Literal:
                    literal_values = get_args(base_type)
                    if field_value in literal_values:
                        st.session_state[field_key] = field_value
                    else:
                        st.session_state[field_key] = "(None)"
                
                elif base_type == bool:
                    if field_value is True:
                        st.session_state[field_key] = "True"
                    elif field_value is False:
                        st.session_state[field_key] = "False"
                    else:
                        st.session_state[field_key] = "(None)"
                
                elif base_type in (int, float):
                    if is_optional:
                        if field_value is not None:
                            st.session_state[mode_key] = "Number"
                            st.session_state[field_key] = base_type(field_value)
                        else:
                            st.session_state[mode_key] = "(None)"
                            st.session_state[field_key] = base_type(0)
                    else:
                        st.session_state[field_key] = base_type(field_value) if field_value is not None else base_type(0)
                
                elif inspect.isclass(base_type) and issubclass(base_type, BaseModel):
                    nested_values = field_value if isinstance(field_value, dict) else {}
                    nested_prefix = f"{field_key}_"
                    FormStateManager.set_form_state(base_type, nested_values, nested_prefix)
                
                elif get_origin(base_type) is list:
                    list_key = f"{field_key}_items"
                    if isinstance(field_value, list):
                        st.session_state[list_key] = copy.deepcopy(field_value)
                    else:
                        st.session_state[list_key] = []
                
                else:  # String and others
                    st.session_state[field_key] = str(field_value) if field_value is not None else ""
                
            except Exception as e:
                # Silent handling with fallback
                st.session_state[f"{current_prefix}{field_name}"] = ""
        
        try:
            for field_name, field_info in model_class.model_fields.items():
                field_value = values.get(field_name)
                set_field_state(field_name, field_info, field_value, prefix)
        except Exception as e:
            st.error(f"Error setting form state: {e}")

# =====================================================
# 4) RENDERING FUNCTIONS
# =====================================================

def render_nested_field(field_name, field_info, current_value, prefix, depth, base_color):
    """Render a single field with proper state management"""
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

def render_nested_object(pyd_model_class, current_values, prefix, depth, base_color):
    """Render fields of a nested pydantic object in a styled_container"""
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
    """Advanced renderer for list fields"""
    list_key = f"{key_base}_items"
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

def render_top_level_field(field_name, field_info, current_value, prefix=""):
    """Renders a field from the MAIN Pydantic model, wrapped or standard"""
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

def process_main_model_fields(model_class, current_values, prefix=""):
    """Renders all fields in the main Pydantic model"""
    result_vals = {}
    st.session_state["color_index"] = 0  # reset color index
    for fn, fi in model_class.model_fields.items():
        cur_val = current_values.get(fn)
        result_vals[fn] = render_top_level_field(fn, fi, cur_val, prefix)
    return result_vals

def gather_values_from_state(model_class, prefix=""):
    """Recursively gather values from st.session_state for the given model_class fields"""
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
# 5) SESSION MANAGEMENT
# =====================================================

def export_complete_session():
    """Export complete session including metadata"""
    try:
        state_manager = StateManager()
        session_data = state_manager.get_current_state_snapshot()
        
        if session_data:
            # Add extraction-specific metadata
            session_data["__pydantic_extraction_session__"] = True
            session_data["model_schema"] = {}
            
            if st.session_state.get("model_class"):
                try:
                    session_data["model_schema"] = st.session_state["model_class"].model_json_schema()
                except Exception:
                    pass
            
            return json.dumps(session_data, indent=2, default=str)
        return None
    except Exception as e:
        st.error(f"Failed to export session: {e}")
        return None

def import_complete_session(file_data):
    """Import complete session with validation"""
    try:
        if file_data["type"] != "session_export":
            return False
        
        session_data = file_data["data"]
        
        # Validate session data
        if not isinstance(session_data, dict):
            st.error("Invalid session format")
            return False
        
        if not session_data.get("__session_metadata__"):
            st.error("Missing session metadata")
            return False
        
        # Use state manager to restore
        state_manager = StateManager()
        success = state_manager.restore_state_from_snapshot(session_data)
        
        if success:
            st.success("Session restored successfully!")
            st.rerun()
        
        return success
        
    except Exception as e:
        st.error(f"Failed to import session: {e}")
        st.error(traceback.format_exc())
        return False

# =====================================================
# 6) CALLBACK FUNCTIONS
# =====================================================

def parse_pydantic_code(code_str, uploaded_py):
    """Parse Pydantic code with enhanced error handling"""
    try:
        # Get code from text area or uploaded file
        if uploaded_py:
            code_str = uploaded_py.read().decode("utf-8")
        
        if not code_str.strip():
            st.warning("No code provided")
            return
        
        # Parse the code
        found_classes = load_model_code(code_str)
        
        if found_classes:
            st.session_state["model_code_str"] = code_str
            st.session_state["available_model_names"] = list(found_classes.keys())
            
            # Set default model
            if found_classes:
                first_model = list(found_classes.keys())[0]
                st.session_state["model_name"] = first_model
                st.session_state["model_class"] = found_classes[first_model]
            
            st.success(f"✅ Found {len(found_classes)} model(s): {', '.join(found_classes.keys())}")
        else:
            st.error("No valid Pydantic models found")
            
    except Exception as e:
        st.error(f"Error parsing code: {e}")

def on_model_change():
    """Handle model selection change"""
    try:
        selected_model = st.session_state["model_selector"]
        if selected_model and st.session_state.get("model_code_str"):
            found_classes = load_model_code(st.session_state["model_code_str"])
            if selected_model in found_classes:
                st.session_state["model_name"] = selected_model
                st.session_state["model_class"] = found_classes[selected_model]
                st.session_state["extraction_initialized"] = False  # Reset for new model
                st.success(f"✅ Selected model: {selected_model}")
    except Exception as e:
        st.error(f"Error changing model: {e}")

def process_uploaded_data():
    """Process uploaded data file"""
    try:
        uploaded_file = st.session_state.get("data_file")
        if not uploaded_file:
            return
        
        file_data = parse_uploaded_file(uploaded_file)
        
        # Handle session import
        if file_data.get("type") == "session_export":
            if import_complete_session(file_data):
                return
        
        # Handle regular data files
        st.session_state["loaded_file"] = file_data
        st.session_state["extractions"] = []
        st.session_state["current_row_index"] = 0
        st.session_state["extraction_initialized"] = False
        
        if file_data.get("data") is not None:
            st.success(f"✅ Loaded {len(file_data['data'])} rows from {file_data['filename']}")
        elif file_data.get("text"):
            st.success(f"✅ Loaded text file: {file_data['filename']}")
        
    except Exception as e:
        st.error(f"Error processing file: {e}")

def update_id_column():
    """Update ID column selection"""
    st.session_state["id_column"] = st.session_state["id_col_selector"]
    st.session_state["extraction_initialized"] = False

def update_text_column():
    """Update text column selection"""
    st.session_state["text_column"] = st.session_state["text_col_selector"]
    st.session_state["extraction_initialized"] = False

def check_setup_complete():
    """Check if setup is complete"""
    return (st.session_state.get("model_class") is not None and
            st.session_state.get("loaded_file") is not None and
            (st.session_state["loaded_file"].get("data") is None or 
             st.session_state.get("text_column")))

def show_setup_guide():
    """Show setup requirements guide"""
    st.info("📋 **Setup Required**")
    
    with st.expander("Setup Checklist", expanded=True):
        # Model check
        if st.session_state.get("model_class"):
            st.success("✅ Pydantic model loaded")
        else:
            st.error("❌ Load a Pydantic model in the sidebar")
        
        # Data check
        if st.session_state.get("loaded_file"):
            st.success("✅ Data file loaded")
            
            # Column selection check for structured data
            if st.session_state["loaded_file"].get("data") is not None:
                if st.session_state.get("text_column"):
                    st.success("✅ Text column selected")
                else:
                    st.error("❌ Select a text column in the sidebar")
        else:
            st.error("❌ Upload a data file in the sidebar")

def initialize_extraction():
    """Initialize extraction session"""
    try:
        with StateManager().state_transaction():
            initialize_all_rows_in_memory()
            st.session_state["extraction_initialized"] = True
            st.session_state["form_state_initialized"] = False
            
            # Initialize form state for first row
            if st.session_state["extractions"]:
                first_extraction = st.session_state["extractions"][0]
                FormStateManager.set_form_state(
                    st.session_state["model_class"],
                    first_extraction.get("values", {}),
                    ""
                )
        
        st.success("🎉 Extraction session initialized!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to initialize extraction: {e}")

def save_extraction_callback():
    """Gather current row's field values and save them"""
    try:
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

        # Move to next row if possible
        file_data = st.session_state["loaded_file"]
        if file_data and file_data["data"] is not None:
            if row_index + 1 < len(file_data["data"]):
                navigate_to_row(row_index + 1)
        
    except Exception as e:
        st.error(f"Failed to save extraction: {e}")

def navigate_previous():
    """Navigate to previous row with enhanced state management"""
    try:
        current_idx = st.session_state["current_row_index"]
        if current_idx > 0:
            # Force form state reset
            st.session_state["form_state_initialized"] = False
            # Clear all form-related keys
            FormStateManager.clear_form_state(st.session_state["model_class"], "")
            # Update index
            st.session_state["current_row_index"] = current_idx - 1
    except Exception as e:
        st.error(f"Navigation error: {e}")

def navigate_next():
    """Navigate to next row with enhanced state management"""
    try:
        current_idx = st.session_state["current_row_index"]
        file_data = st.session_state.get("loaded_file")
        if file_data and file_data.get("data") is not None:
            total_rows = len(file_data["data"])
            if current_idx < total_rows - 1:
                # Force form state reset
                st.session_state["form_state_initialized"] = False
                # Clear all form-related keys
                FormStateManager.clear_form_state(st.session_state["model_class"], "")
                # Update index
                st.session_state["current_row_index"] = current_idx + 1
    except Exception as e:
        st.error(f"Navigation error: {e}")

def navigate_to_row():
    """Navigate to specific row from input field"""
    try:
        target_row = st.session_state.get("row_selection_input", 1) - 1
        file_data = st.session_state.get("loaded_file")
        if file_data and file_data.get("data") is not None:
            total_rows = len(file_data["data"])
            if 0 <= target_row < total_rows:
                # Force form state reset
                st.session_state["form_state_initialized"] = False
                # Clear all form-related keys
                FormStateManager.clear_form_state(st.session_state["model_class"], "")
                # Update index
                st.session_state["current_row_index"] = target_row
    except Exception as e:
        st.error(f"Navigation error: {e}")

def show_extraction_interface():
    """Show the main extraction interface with enhanced state management"""
    file_data = st.session_state["loaded_file"]
    model_class = st.session_state["model_class"]
    
# Initialize form state for current row if needed
    if not st.session_state.get("form_state_initialized", False):
        row_index = st.session_state["current_row_index"]
        current_vals = {}
        if len(st.session_state["extractions"]) > row_index:
            ex = st.session_state["extractions"][row_index]
            if isinstance(ex, dict) and "values" in ex:
                current_vals = ex["values"]
        
        # Always clear form state first to prevent conflicts
        FormStateManager.clear_form_state(model_class, prefix="")
        
        # Small delay to ensure state is cleared
        import time
        time.sleep(0.01)
        
        # Set new form state
        FormStateManager.set_form_state(model_class, current_vals, prefix="")
        st.session_state["form_state_initialized"] = True

    # Show navigation for structured data
    if file_data["data"] is not None and not file_data["data"].empty:
        df = file_data["data"]
        total_rows = len(df)
        
        st.subheader("Data Navigation")
        
        # Progress tracking
        reviewed_count = len([e for e in st.session_state["extractions"] if e and e.get("review_status") == "manually_reviewed"])
        st.progress(reviewed_count / total_rows, f"Reviewed {reviewed_count} of {total_rows} rows")

        # Navigation controls
        nav_cols = st.columns([1,1,1,1,2])
        
        with nav_cols[0]:
            st.button(
                "◀ Previous", 
                disabled=(st.session_state["current_row_index"] == 0),
                on_click=navigate_previous,
                key="nav_previous_btn"
            )
        
        with nav_cols[1]:
            st.button(
                "Next ▶", 
                disabled=(st.session_state["current_row_index"] >= total_rows - 1),
                on_click=navigate_next,
                key="nav_next_btn"
            )
        
        with nav_cols[2]:
            st.markdown(f"**Row {st.session_state['current_row_index']+1} of {total_rows}**")
        
        with nav_cols[3]:
            st.button(
                "Jump to →",
                on_click=navigate_to_row,
                key="nav_jump_btn"
            )
        
        with nav_cols[4]:
            st.number_input(
                f"Row (1-{total_rows}):",
                min_value=1,
                max_value=total_rows,
                value=st.session_state["current_row_index"] + 1,
                key="row_selection_input",
                label_visibility="collapsed"
            )

    row_index = st.session_state["current_row_index"]

    # Main extraction interface
    col_extraction, col_source = st.columns([3, 2])
    
    # Extraction column
    with col_extraction.container(height=st.session_state["extraction_dashboard_columns_height"]):
        st.markdown(f"#### Extract Data into *{model_class.__name__}*")
        
        # Get current values for this row
        current_vals = {}
        if len(st.session_state["extractions"]) > row_index:
            ex = st.session_state["extractions"][row_index]
            if isinstance(ex, dict) and "values" in ex:
                current_vals = ex["values"]

        # Render the model fields
        process_main_model_fields(model_class, current_vals, prefix="")

        # Action buttons
        btn_col1, btn_col2 = st.columns([1, 1])
        
        with btn_col1:
            if st.button("💾 Save Extraction", type="primary"):
                save_extraction_callback()
        
        with btn_col2:
            # Show current row status
            if len(st.session_state["extractions"]) > row_index and st.session_state["extractions"][row_index]:
                status = st.session_state["extractions"][row_index].get("review_status", "not_reviewed")
                if status == "manually_reviewed":
                    st.success("✅ Reviewed")
                else:
                    st.warning("⏳ Not Reviewed")

    # Source data column
    with col_source.container(height=st.session_state["extraction_dashboard_columns_height"]):
        st.markdown("#### Source Data")
        
        if file_data["data"] is not None and not file_data["data"].empty:
            if row_index < len(file_data["data"]):
                row_dict = file_data["data"].iloc[row_index].to_dict()
                text_col = st.session_state.get("text_column")
                
                if text_col and text_col in row_dict:
                    st.markdown("**Content for Extraction:**")
                    st.markdown(
                        f"<div style='background-color:#F9F9F9; padding:0.5rem; border-radius:4px;'>{row_dict[text_col]}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    # Show all row data
                    for key, value in row_dict.items():
                        st.markdown(f"**{key}**: {value}")
                
                # Show ID if available
                id_col = st.session_state.get("id_column")
                if id_col and id_col in row_dict:
                    st.info(f"**ID**: {row_dict[id_col]}")
        else:
            # Unstructured text scenario
            if file_data.get("text"):
                st.markdown("**Content for Extraction:**")
                st.markdown(
                    f"<div style='background-color:#F9F9F9; padding:0.5rem; border-radius:4px;'>{file_data['text']}</div>", 
                    unsafe_allow_html=True
                )

def show_export_interface():
    """Show export options"""
    st.subheader("📤 Export Options")
    
    extractions = st.session_state.get("extractions", [])
    if not extractions:
        st.info("No extractions to export yet.")
        return

    # Generate export data
    extracted_data = generate_export_data()
    
    if extracted_data:
        # Show data preview
        tab_data, tab_json = st.tabs(["Data View", "JSON View"])
        
        with tab_data:
            df_extractions = pd.DataFrame(extracted_data)
            
            # Add color coding legend
            st.markdown("**Legend:** 🟢 Green = Manually Reviewed | 🔴 Red = Not Reviewed")
            
            # Display with color coding
            def highlight_review_status(row):
                if row['review_status'] == 'manually_reviewed':
                    return ['background-color: #d4edda'] * len(row)  # Light green
                else:
                    return ['background-color: #f8d7da'] * len(row)  # Light red
            
            st.dataframe(
                df_extractions.style.apply(highlight_review_status, axis=1),
                use_container_width=True
            )
            
            # Show summary stats
            total_rows = len(df_extractions)
            reviewed_rows = len(df_extractions[df_extractions['review_status'] == 'manually_reviewed'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", total_rows)
            with col2:
                st.metric("Reviewed", reviewed_rows, delta=f"{reviewed_rows/total_rows*100:.1f}%")
            with col3:
                st.metric("Not Reviewed", total_rows - reviewed_rows)
        
        with tab_json:
            json_str = json.dumps(extracted_data, indent=2)
            st.code(json_str, language="json")

    # Export buttons
    col1, col2, col3 = st.columns(3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with col1:
        session_json = export_complete_session()
        if session_json:
            st.download_button(
                "💾 Export Session",
                data=session_json,
                file_name=f"extraction_session_{timestamp}.json",
                mime="application/json",
                help="Export complete session for resuming work later"
            )
    
    with col2:
        if extracted_data:
            df_for_csv = pd.DataFrame(extracted_data)
            csv_str = df_for_csv.to_csv(index=False)
            st.download_button(
                "📊 Export CSV",
                data=csv_str,
                file_name=f"extractions_{timestamp}.csv",
                mime="text/csv",
                help="Export extraction data as CSV"
            )
    
    with col3:
        if extracted_data:
            json_str = json.dumps(extracted_data, indent=2)
            st.download_button(
                "📋 Export JSON",
                data=json_str,
                file_name=f"extractions_{timestamp}.json",
                mime="application/json",
                help="Export extraction data as JSON"
            )

def generate_export_data():
    """Generate export data with proper flattening"""
    try:
        extractions = st.session_state["extractions"]
        extracted_data = []
        
        for i, extraction in enumerate(extractions):
            if extraction:  # Only if extraction exists
                vals = extraction.get("values", {})
                raw_data = extraction.get("source_data", {})
                review_status = extraction.get("review_status", "not_reviewed")
                review_timestamp = extraction.get("review_timestamp")
                
                # Flatten extracted fields
                flat = flatten_for_export(vals)
                
                # Build export record
                id_val = extraction.get("id", f"row_{i+1}")
                new_flat = {
                    "id": str(id_val),
                    "row_index": extraction.get("row_index", i),
                    "review_status": review_status,
                    "review_timestamp": review_timestamp
                }
                
                # Add flattened extraction values
                for k, v in flat.items():
                    if k != "id":
                        new_flat[k] = v
                
                # Add raw data with raw_ prefix
                for rk, rv in raw_data.items():
                    new_flat[f"raw_{rk}"] = rv
                
                extracted_data.append(new_flat)
        
        return extracted_data
    
    except Exception as e:
        st.error(f"Failed to generate export data: {e}")
        return []

def import_session(session_file):
    """Import session from file"""
    try:
        file_data = parse_uploaded_file(session_file)
        if import_complete_session(file_data):
            st.success("Session imported successfully!")
    except Exception as e:
        st.error(f"Import failed: {e}")

def reset_session():
    """Reset entire session"""
    try:
        # Clear all session state except UI preferences
        keys_to_keep = ["extraction_dashboard_columns_height"]
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        
        # Reinitialize
        StateManager().init_session_states()
        st.success("Session reset successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Reset failed: {e}")

# =====================================================
# 7) MAIN APPLICATION
# =====================================================

def main():
    # Initialize enhanced state management
    state_manager = StateManager()
    state_manager.init_session_states()
    
    # Validate state consistency
    if not state_manager.validate_state_consistency():
        st.warning("State inconsistency detected and corrected. Please refresh if needed.")
    
    # Sidebar setup
    with st.sidebar:
        st.title("🚀 Enhanced Session Setup")
        
        # Model setup section
        st.subheader("Pydantic Model")
        model_code = st.text_area(
            "Paste your Pydantic model code:",
            height=150,
            value=st.session_state.get("model_code_str", ""),
            key="pydantic_model_code"
        )
        
        uploaded_py = st.file_uploader(
            "Or upload Python file (.py):",
            type=["py"],
            key="uploaded_py"
        )
        
        if st.button("🔄 Parse Pydantic Code", type="primary"):
            parse_pydantic_code(model_code, uploaded_py)
        
        # Model selection
        if st.session_state.get("available_model_names"):
            model_name = st.selectbox(
                "Select Model:",
                st.session_state["available_model_names"],
                index=0 if st.session_state.get("model_name") not in st.session_state["available_model_names"] 
                      else st.session_state["available_model_names"].index(st.session_state["model_name"]),
                key="model_selector",
                on_change=on_model_change
            )
        
        st.divider()
        
        # Data source section
        st.subheader("Data Source")
        uploaded_data = st.file_uploader(
            "Upload data file:",
            type=["xlsx", "csv", "json", "txt", "docx", "pdf"],
            key="data_file",
            on_change=lambda: process_uploaded_data()
        )
        
        # Column selection for structured data
        if st.session_state.get("loaded_file") and st.session_state["loaded_file"].get("data") is not None:
            df = st.session_state["loaded_file"]["data"]
            if not df.empty:
                cols = [""] + list(df.columns)
                
                st.selectbox(
                    "ID Column (optional):",
                    cols,
                    index=cols.index(st.session_state.get("id_column", "")) if st.session_state.get("id_column", "") in cols else 0,
                    key="id_col_selector",
                    on_change=update_id_column
                )
                
                st.selectbox(
                    "Text/Content Column:",
                    cols,
                    index=cols.index(st.session_state.get("text_column", "")) if st.session_state.get("text_column", "") in cols else 0,
                    key="text_col_selector",
                    on_change=update_text_column
                )
        
        st.divider()
        
        # Session management
        st.subheader("📁 Session Management")
        
        # Export session
        col1, col2 = st.columns(2)
        with col1:
            session_json = export_complete_session()
            if session_json:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "💾 Export Session",
                    data=session_json,
                    file_name=f"session_{timestamp}.json",
                    mime="application/json",
                    help="Export complete session"
                )
        
        with col2:
            if st.button("🔄 Reset Session", help="Reset all data", type="secondary"):
                reset_session()
        
        # Import session
        session_file = st.file_uploader(
            "Import Session:",
            type=["json"],
            key="session_import_file"
        )
        
        if session_file and st.button("📂 Import Session"):
            import_session(session_file)
        
        st.divider()
        
        # UI Configuration
        st.subheader("⚙️ UI Configuration")
        st.session_state["extraction_dashboard_columns_height"] = st.slider(
            "Column Height (px):",
            min_value=300,
            max_value=1500,
            value=st.session_state.get("extraction_dashboard_columns_height", 600),
            step=50
        )
    
    # Main content area
    st.title("📋 Enhanced Pydantic Extraction Dashboard")
    
    # Check setup requirements
    if not check_setup_complete():
        show_setup_guide()
        return
    
    # Initialize extraction if not done
    if not st.session_state.get("extraction_initialized", False):
        if st.button("🚀 Initialize Extraction Session", type="primary"):
            initialize_extraction()
        return
    
    # Show extraction interface
    show_extraction_interface()
    
    # Show export section
    show_export_interface()

if __name__ == "__main__":
    main()