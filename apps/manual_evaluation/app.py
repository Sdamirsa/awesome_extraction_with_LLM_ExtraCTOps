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
            if isinstance(json_data, dict) and "__pydantic_evaluation_data__" in json_data:
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

def create_or_update_evaluation(index, field_values, source_data=None, unique_id=None):
    """
    Stores or updates evaluation data in st.session_state["evaluations"] at the given index.
    Also keeps the entire row's raw data in `source_data`.
    """
    while len(st.session_state["evaluations"]) <= index:
        st.session_state["evaluations"].append({})
    
    evaluation = {
        "values": field_values,
        "row_index": index,
        "source_data": source_data,  # Entire raw data
    }
    if unique_id:
        evaluation["id"] = unique_id
    else:
        evaluation["id"] = f"row_{index+1}"
    
    st.session_state["evaluations"][index] = evaluation

def init_session_states():
    """Initialize session state variables for the app."""
    if "evaluations" not in st.session_state:
        st.session_state["evaluations"] = [] 
    if "current_row_index" not in st.session_state:
        st.session_state["current_row_index"] = 0
    if "model_class" not in st.session_state:
        st.session_state["model_class"] = None
    if "loaded_file" not in st.session_state:
        st.session_state["loaded_file"] = None
    if "evaluated_count" not in st.session_state:
        st.session_state["evaluated_count"] = 0
    if "model_code_str" not in st.session_state:
        st.session_state["model_code_str"] = ""
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = ""
    if "id_column" not in st.session_state:
        st.session_state["id_column"] = ""
    if "text_column" not in st.session_state:
        st.session_state["text_column"] = ""
    if "evaluation_dashboard_columns_height" not in st.session_state:
        st.session_state["evaluation_dashboard_columns_height"] = 600
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

def render_top_level_field_for_evaluation(field_name, field_info, current_value, prefix=""):
    """
    Renders a field for evaluation, showing options to approve, reject or add comments.
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
        return render_nested_field_for_evaluation(
            field_name=field_name,
            field_info=field_info,
            current_value=current_value,
            prefix=prefix,
            depth=0,
            base_color="#FFFFFF"
        )

def render_nested_field_for_evaluation(field_name, field_info, current_value, prefix, depth, base_color):
    """
    Renders a single field for evaluation, offering approval or rejection options.
    """
    field_annotation = field_info.annotation
    field_description = field_info.description or ""
    is_opt = is_optional_type(field_annotation)
    base_type = get_base_type(field_annotation)

    # Build label
    label_core = f"{field_name} (Optional)" if is_opt else field_name
    if current_value is not None:
        label_core += " ✅"
    key_base = f"{prefix}{field_name}"
    
    # Approve/Reject Options
    if base_type == bool:
        options = ["Approve", "Reject", "Comment"]
        choice = st.radio(f"Evaluate {label_core}", options, key=key_base)
        if choice == "Approve":
            return "Approved"
        elif choice == "Reject":
            return "Rejected"
        else:
            comment = st.text_area("Add Comment", "", key=f"{key_base}_comment")
            return comment
    
    # Enum/Literal fields
    if inspect.isclass(base_type) and issubclass(base_type, Enum) or get_origin(base_type) is Literal:
        options = ["Approve", "Reject", "Comment"]
        choice = st.radio(f"Evaluate {label_core}", options, key=key_base)
        if choice == "Approve":
            return "Approved"
        elif choice == "Reject":
            return "Rejected"
        else:
            comment = st.text_area("Add Comment", "", key=f"{key_base}_comment")
            return comment

    # For other simple types (int, float, str)
    value_input = st.text_input(f"Enter value for {label_core}", str(current_value) if current_value else "", key=key_base)
    options = ["Approve", "Reject", "Comment"]
    choice = st.radio(f"Evaluate {label_core}", options, key=f"{key_base}_evaluate")
    if choice == "Approve":
        return "Approved"
    elif choice == "Reject":
        return "Rejected"
    else:
        comment = st.text_area("Add Comment", "", key=f"{key_base}_comment")
        return comment

# =====================================================
# 6) NAVIGATION CALLBACKS
# =====================================================

def previous_evaluation():
    """
    Goes to the previous row (if possible).
    """
    current_idx = st.session_state["current_row_index"]
    if current_idx > 0:
        st.session_state["current_row_index"] = current_idx - 1

def next_evaluation():
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
        page_title="Pydantic Evaluation App",
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
                "Load previous evaluation session (JSON):",
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
                        help="Column containing the text for evaluation",
                        key="text_column_select",
                        on_change=update_text_column
                    )

        st.markdown("---")
        st.title("🔧 UI Configuration")
        st.subheader("Column Heights")
        st.session_state["evaluation_dashboard_columns_height"] = st.number_input(
            "Evaluation Column Height (px)",
            min_value=300,
            max_value=2000,
            value=st.session_state["evaluation_dashboard_columns_height"],
            step=50
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MAIN AREA: EVALUATION DASHBOARD + REVIEW/EXPORT
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.title("📋 Pydantic Evaluation Dashboard")

    # 1) If no model or data loaded, instruct user
    if not st.session_state["model_class"] or not st.session_state["loaded_file"]:
        st.info("Please select 'Initiate New' or 'Continue Previous' in the sidebar to load data and a model.")
        return

    # 2) Data Nav + Evaluation
    file_data = st.session_state["loaded_file"]
    model_class = st.session_state["model_class"]

    # If there's structured data, show row nav
    if file_data["data"] is not None and not file_data["data"].empty:
        df = file_data["data"]
        total_rows = len(df)
        st.subheader("Data Navigation")

        completed = len([e for e in st.session_state["evaluations"] if e])
    st.progress(completed / total_rows, f"Evaluated {completed} of {total_rows} rows")

    nav_cols = st.columns([1,1,1,1,2])
    with nav_cols[0]:
        st.button("❮ Previous", on_click=previous_evaluation, disabled=(st.session_state["current_row_index"] == 0))
    with nav_cols[1]:
        st.button("Next ❯", on_click=next_evaluation, disabled=(st.session_state["current_row_index"] >= total_rows - 1))
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

    # Layout for Evaluation + Source Data
    col_evaluation, col_source = st.columns([3,2])
    with col_evaluation.container(height=st.session_state["evaluation_dashboard_columns_height"]):
        st.markdown(f"#### Evaluate Data for *{model_class.__name__}*")
        
        # If we already have something saved for this row, load it
        current_vals = {}
        if len(st.session_state["evaluations"]) > row_index:
            ex = st.session_state["evaluations"][row_index]
            if isinstance(ex, dict) and "values" in ex:
                current_vals = ex["values"]

        # Render the model for evaluation
        process_main_model_fields(model_class, current_vals, prefix="")

        # Buttons
        bc1, bc2, bc3 = st.columns([1,1,1])
        with bc1:
            st.button("Save Evaluation", on_click=save_extraction_callback, type="primary")
        with bc2:
            st.button("Validate", on_click=validate_extraction)
        with bc3:
            st.write("")

    with col_source.container(height=st.session_state["evaluation_dashboard_columns_height"]):
        st.markdown("#### Source Data")
        if file_data["data"] is not None and not file_data["data"].empty:
            if row_index < len(file_data["data"]):
                row_dict = file_data["data"].iloc[row_index].to_dict()
                text_col = st.session_state.get("text_column")
                if text_col and text_col in row_dict:
                    st.markdown("**Content for Evaluation**:")
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
                st.markdown("**Content for Evaluation**:")
                st.markdown(
                    f"<div style='background-color:#F9F9F9; padding:0.5rem;'>{file_data['text']}</div>", 
                    unsafe_allow_html=True
                )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # REVIEW & EXPORT (at the bottom)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.subheader("Review & Export")
    evaluations = st.session_state["evaluations"]
    if not evaluations:
        st.info("No evaluations yet.")
        return

    tab_data, tab_json = st.tabs(["Data View", "JSON View"])
    with tab_data:
        evaluation_data = []
        for i, evaluation in enumerate(evaluations):
            if evaluation:
                vals = evaluation.get("values", {})
                raw_data = evaluation.get("source_data", {})
                
                # Flatten evaluated fields recursively
                flat = flatten_for_export(vals)
                
                # Add index and ID
                flat["row_index"] = evaluation.get("row_index", i)
                flat["id"] = evaluation.get("id", f"row_{i+1}")
                
                # Add raw data with raw_ prefix
                for rk, rv in raw_data.items():
                    flat[f"raw_{rk}"] = rv
                
                evaluation_data.append(flat)

        if evaluation_data:
            df_evaluations = pd.DataFrame(evaluation_data)
            st.dataframe(df_evaluations, use_container_width=True)
        else:
            st.info("No valid evaluations to display.")

    with tab_json:
        if evaluation_data:
            json_str = json.dumps(evaluation_data, indent=2)
            st.code(json_str, language="json")
        else:
            st.info("No valid evaluations to display.")        

    # Export Buttons
    ec1, ec2, ec3 = st.columns([2,1,1])
    export_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with ec1:
        session_state_export = json.dumps(dict(st.session_state), default=str)
        st.download_button(
            "Download Complete Session State (JSON)",
            data=session_state_export,
            file_name=f"evaluation_state_{export_time_stamp}.json",
            mime="application/json",
            help="Download the entire session state (including all evaluations). You can use this to save your incomplete evaluation and load it in the future in the app."
        )

    with ec2:
        json_str = json.dumps(evaluation_data, indent=2)
        st.download_button(
            "Download Evaluations (JSON)",
            data=json_str,
            file_name=f"evaluations_{export_time_stamp}.json",
            mime="application/json"
        )
        
    with ec3:
        if evaluation_data:
            csv_str = pd.DataFrame(evaluation_data).to_csv(index=False)
            st.download_button(
                "Download Evaluations (CSV)",
                data=csv_str,
                file_name=f"evaluations_{export_time_stamp}.csv",
                mime="text/csv"
            )