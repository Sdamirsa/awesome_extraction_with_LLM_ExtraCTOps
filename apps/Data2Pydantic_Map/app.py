"""
Data2Pydantic_Map App

# Description
    [A Streamlit application for mapping and transforming EHR/database data to user-defined Pydantic models. Supports mapping template generation, data transformation, and LLM-assisted auto-mapping.]

    - Arguments:
        - Data (Excel, CSV): The file containing the source data to be mapped (supports both wide and long format).
        - Pydantic Model (.py) Or Code (string): The file or code string containing the Pydantic model definition.
        - Mapping File (Excel, CSV): The file defining the mapping between source data and Pydantic fields.

    - Environment Arguments:
        - None required for basic use. (LLM auto-mapping requires OpenAI/Azure API key.)

    - Returns
        - Mapping Template (Excel): A template for mapping source data to Pydantic fields.
        - Transformed Data (JSON): Data transformed to match the Pydantic model structure.
        - Transformed Data (Excel): Flattened data with nested fields separated by '::'.
        - Mapping Monitor (Excel): Mapping file with validation feedback.

# Engine:
    - Serve (utils/data/main-function/sub-function): main-function
    - Served by (API/Direct/Subprocess): Direct
    - Path to venv, if require separate venv: the_venvs/venv_streamlit
    - libraries to import: [pydantic,streamlit,pandas,openpyxl,xlsxwriter,openai]

# Identity
    - Last Status (future/in-progress/complete/published): published
    - Publish Date: 2025-05-21
    - Version: 0.1
    - License: MIT
    - Author: Seyed Amir Ahmad Safavi-Naini Safavi-Naini, sdamirsa@gmail.com (the nominee for the longest name ever)
    - Source: https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps

# Changelog
    - 2025-05-21: version 0.1 (initial public release)

# To-do:
    - [] Improve error handling and user feedback for all sections
    - [] Add support for additional data formats (e.g., JSON, TXT)
    - [] Enhance LLM prompt engineering for more robust auto-mapping
    - [] Add more sample models and mapping templates
    - [] Add advanced mapping logic (e.g., custom evaluation methods)
"""

import streamlit as st
import pandas as pd
import json
import importlib.util
import sys
import tempfile
from types import ModuleType
from typing import Any, Dict, Tuple, List, get_origin, get_args, Union  # Add get_origin, get_args, Union
from pydantic import BaseModel
import io
from enum import Enum

# --- Utility: Load Pydantic model from .py file or text ---
def load_pydantic_model_from_code(code_str: str) -> Dict[str, Any]:
    import types
    module = types.ModuleType('user_model')
    exec(code_str, module.__dict__)
    # Find all BaseModel subclasses
    models = {k: v for k, v in module.__dict__.items() if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel}
    return models

def load_pydantic_model_from_file(py_file) -> Dict[str, Any]:
    code_str = py_file.read().decode('utf-8')
    return load_pydantic_model_from_code(code_str)

# --- Flatten model (from code.py) ---
def flatten_model(model: type, prefix: str = ""):
    # Use model.model_fields for Pydantic v2
    for name, field in model.model_fields.items():
        typ = field.annotation
        path = f"{prefix}::{name}" if prefix else name
        origin = get_origin(typ)
        if origin in (list, List):
            inner = get_args(typ)[0]
            yield from _flatten_type(inner, path)
        else:
            yield from _flatten_type(typ, path)

def _flatten_type(typ, path):
    origin = get_origin(typ)
    if origin is Union:
        # Optional[...] or Union[...] - pick the first non-None type
        args = [a for a in get_args(typ) if a is not type(None)]
        if args:
            yield from _flatten_type(args[0], path)
        else:
            yield (path, "Any")
    elif isinstance(typ, type) and issubclass(typ, BaseModel):
        yield from flatten_model(typ, path)
    elif isinstance(typ, type) and issubclass(typ, Enum):
        yield (path, typ.__name__)
    else:
        yield (path, getattr(typ, "__name__", str(typ)))

# --- Generate mapping template (NEW LOGIC) ---
def get_enum_options_list(enum_cls):
    if enum_cls is None:
        return []
    try:
        return [e.value for e in enum_cls]
    except Exception:
        return []

def generate_mapping_template(model: BaseModel.__class__, schema_module: ModuleType = None) -> pd.DataFrame:
    rows = []
    for fld, typ_name in flatten_model(model):
        enum_cls = getattr(schema_module, typ_name, None) if schema_module else None
        options = get_enum_options_list(enum_cls)
        # For bool, only two rows: True and False
        if typ_name.lower() == "bool" or (enum_cls and typ_name.lower().startswith("bool")):
            options = ["True", "False"]
        
        # Determine if pydantic_value should be blank
        is_numeric = typ_name.lower() in ("float", "int")

        if options:
            for opt in options:
                rows.append(dict(
                    pydantic_field=fld,
                    pydantic_type=typ_name,
                    evaluation_method="smart_exact_match",
                    multiValue_handling_method="haveBoth",
                    pydantic_value=opt if not is_numeric else "",  # Blank for numerics
                    Observation_ColName="",
                    Observation_Value="",
                    Observation_Value2="",
                    Observation_Value3="",
                    Observation_Value4=""
                ))
        else:
            rows.append(dict(
                pydantic_field=fld,
                pydantic_type=typ_name,
                evaluation_method="smart_exact_match",
                multiValue_handling_method="haveBoth",
                pydantic_value="" if not is_numeric else "", # Blank for numerics, and default for others
                Observation_ColName="",
                Observation_Value="",
                Observation_Value2="",
                Observation_Value3="",
                Observation_Value4=""
            ))
    # Ensure column order
    col_order = [
        "pydantic_field", "pydantic_type", "evaluation_method", "multiValue_handling_method",
        "pydantic_value", "Observation_ColName", "Observation_Value", "Observation_Value2", "Observation_Value3", "Observation_Value4"
    ]
    return pd.DataFrame(rows)[col_order]

def export_mapping_with_guide(mapping_df, out_path):
    import xlsxwriter
    guide_text = (
        "Instructions for filling the mapping template:\n"
        "- For each row, map your data columns to the Pydantic field.\n"
        "- 'pydantic_value': For enums/bools, this is the value to match. For other fields, leave blank.\n"
        "- 'Observation_ColName': Enter the name of the column in your data that corresponds to this mapping.\n"
        "- 'Observation_Value', 'Observation_Value2', ...: Enter up to 4 values from your data that correspond to this mapping (if needed).\n"
        "- For boolean fields, only two rows: one for True, one for False.\n"
        "- For enums, one row per option.\n"
        "- 'evaluation_method' and 'multiValue_handling_method' can be left as default unless you need custom logic.\n"
        "- You may leave unused 'Observation_Value*' columns blank.\n"
    )
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
        mapping_df.to_excel(xw, index=False, sheet_name="mapping")
        guide_df = pd.DataFrame({"Guide": [guide_text]})
        guide_df.to_excel(xw, index=False, sheet_name="guide")
        # Highlight errors if present
        if "mapping_error" in mapping_df.columns:
            wb, ws = xw.book, xw.sheets["mapping"]
            bad_fmt = wb.add_format({"bg_color": "#FFC7CE"})
            for r, err in enumerate(mapping_df["mapping_error"], start=2):
                if err:
                    ws.set_row(r - 1, None, bad_fmt)

def validate_mapping(mapping_df, schema_module):
    from enum import EnumMeta
    mapping_df = mapping_df.copy()
    enum_registry = {
        cls.__name__: cls
        for cls in vars(schema_module).values()
        if isinstance(cls, EnumMeta)
    }
    mapping_df["mapping_error"] = ""
    for i, row in mapping_df.iterrows():
        errors = []
        ptype = row["pydantic_type"]
        if ptype in enum_registry and row["pydantic_value"]:
            allowed = {str(e.value) for e in enum_registry[ptype]}
            if str(row["pydantic_value"]) not in allowed:
                errors.append(f"unknown enum value: {row['pydantic_value']}")
        # At least one Observation_ColName must be set
        if not str(row.get("Observation_ColName", "")).strip():
            errors.append("no Observation_ColName set")
        mapping_df.at[i, "mapping_error"] = "; ".join(errors)
    return mapping_df, enum_registry

# --- Streamlit App ---
st.set_page_config(page_title="Data2Pydantic Mapping Tool", layout="wide")
st.title("Data2Pydantic Mapping Tool")

if "active_section" not in st.session_state:
    st.session_state["active_section"] = None

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("1️⃣ Generate Mapping Template"):
        st.session_state["active_section"] = "template"
with col2:
    if st.button("2️⃣ Transform Data Using Mapping"):
        st.session_state["active_section"] = "transform"
with col3:
    if st.button("3️⃣ Smart (LLM) Mapping"):
        st.session_state["active_section"] = "smart"

# --- Section 1: Generate Mapping Template (update UI for new template) ---
if st.session_state["active_section"] == "template":
    st.header("Step 1: Generate Mapping Template")
    st.write("Upload your Pydantic model as a .py file or paste the code below.")
    py_file = st.file_uploader("Upload .py file with Pydantic model", type=["py"], key="template_py_file")
    code_text = st.text_area("Or paste your Pydantic model code here", height=200, key="template_code_text")
    model_name = None
    models = None
    schema_module = None
    if py_file is not None:
        code_str = py_file.read().decode('utf-8')
        import types
        schema_module = types.ModuleType('user_schema')
        exec(code_str, schema_module.__dict__)
        models = {k: v for k, v in schema_module.__dict__.items() if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel}
    elif code_text.strip():
        import types
        schema_module = types.ModuleType('user_schema')
        exec(code_text, schema_module.__dict__)
        models = {k: v for k, v in schema_module.__dict__.items() if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel}
    if models:
        model_names = list(models.keys())
        # Move last element to be default
        if len(model_names) > 1:
            model_names = model_names[:-1] + [model_names[-1]]
        model_name = st.selectbox("Select model to use", model_names, index=len(model_names)-1)
        if st.button("Generate Template", key="generate_template_btn"):
            model = models[model_name]
            df_template = generate_mapping_template(model, schema_module)
            towrite = io.BytesIO()
            export_mapping_with_guide(df_template, towrite)
            towrite.seek(0)
            st.download_button(
                label="Download Mapping Template (.xlsx)",
                data=towrite,
                file_name="mapping_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# --- Section 2: Transform Data Using Mapping (NEW LOGIC) ---
if st.session_state["active_section"] == "transform":
    st.header("Step 2: Transform Data Using Mapping")
    st.write("Upload your mapping Excel or CSV, Pydantic model, and data file.")
    map_file = st.file_uploader("Upload mapping file (.xlsx or .csv)", type=["xlsx", "csv"], key="transform_map_file")
    py_file = st.file_uploader("Upload .py file with Pydantic model", type=["py"], key="transform_py_file")
    code_text = st.text_area("Or paste your Pydantic model code here", height=200, key="transform_code_text")
    data_file = st.file_uploader("Upload data file (.csv, .xlsx)", type=["csv", "xlsx"], key="transform_data_file")
    
    st.subheader("Source Data Format")
    data_format = st.radio("1️⃣ Select your source data format:", ("Wide", "Long"), key="data_format_select")
    
    patient_id_col_long = ""
    obs_name_col_long = ""
    obs_value_col_long = ""

    if data_format == "Long":
        st.write("2️⃣ Please specify the column names for the long format data:")
        patient_id_col_long = st.text_input("2.1 Patient ID Column Name", key="patient_id_col_long")
        obs_name_col_long = st.text_input("2.2 Observation Name Column Name", key="obs_name_col_long")
        obs_value_col_long = st.text_input("2.3 Observation Value Column Name", key="obs_value_col_long")

    model_name = None
    models = None
    schema_module = None
    if py_file is not None:
        code_str = py_file.read().decode('utf-8')
        import types
        schema_module = types.ModuleType('user_schema')
        exec(code_str, schema_module.__dict__)
        models = {k: v for k, v in schema_module.__dict__.items() if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel}
    elif code_text.strip():
        import types
        schema_module = types.ModuleType('user_schema')
        exec(code_text, schema_module.__dict__)
        models = {k: v for k, v in schema_module.__dict__.items() if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel}
    if models and map_file is not None and data_file is not None:
        model_names = list(models.keys())
        # Move last element to be default
        if len(model_names) > 1:
            model_names = model_names[:-1] + [model_names[-1]]
        model_name = st.selectbox("Select model to use", model_names, index=len(model_names)-1, key="transform_model_select")
        if st.button("Transform Data", key="transform_data_btn"):
            # --- Accept mapping as CSV or XLSX ---
            if map_file.name.endswith(".csv"):
                mapping_df = pd.read_csv(map_file, dtype=str).fillna("")
            else:
                mapping_df = pd.read_excel(map_file, sheet_name="mapping", dtype=str).fillna("")
            mapping_df_validated, enum_registry = validate_mapping(mapping_df, schema_module)
            # Highlight errors in output xlsx
            towrite2 = io.BytesIO()
            export_mapping_with_guide(mapping_df_validated, towrite2)
            towrite2.seek(0)
            st.download_button(
                label="Download Mapping Monitor (Excel)",
                data=towrite2,
                file_name="mapping_monitor.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            # --- Read data ---
            if data_file.name.endswith(".csv"):
                try:
                    data_df = pd.read_csv(data_file, on_bad_lines='skip')
                except Exception as e:
                    st.error(f"Error reading CSV: {e}. Some lines may have been skipped due to formatting issues.")
                    data_df = pd.read_csv(data_file, on_bad_lines='skip', engine='python')
                st.warning("Some rows in your CSV may have been skipped due to formatting errors (e.g., inconsistent number of columns). Please check your data file.")
            else:
                data_df = pd.read_excel(data_file)

            # --- Transform logic (NEW) ---
            transformed_data_for_pydantic = [] # For Pydantic model instantiation
            transformed_data_for_excel = [] # For flattened Excel output
            errors = []
            
            # Prepare mapping_df: make Observation_ColName the index for faster lookups if it's unique per pydantic_field
            # For simplicity, we'll iterate, but for performance, consider optimizing lookups.

            if data_format == "Wide":
                for idx, source_row in data_df.iterrows():
                    item_pydantic = {}
                    item_excel = {}
                    for pydantic_field_name, mapping_group in mapping_df.groupby("pydantic_field"):
                        values_for_field = []
                        for _, map_rule in mapping_group.iterrows():
                            obs_col_name = map_rule.get("Observation_ColName")
                            pydantic_target_value = map_rule.get("pydantic_value") # Value to set if condition met (e.g. True for bool)
                            
                            # Check primary observation column
                            if obs_col_name and obs_col_name in source_row and pd.notna(source_row[obs_col_name]):
                                source_val = source_row[obs_col_name]
                                if pydantic_target_value: # If pydantic_value is specified, we need to match it
                                    if str(source_val) == str(pydantic_target_value):
                                        values_for_field.append(pydantic_target_value) # Use the defined pydantic value
                                else: # No specific pydantic_value to match, so use the source value directly
                                    values_for_field.append(source_val)
                            
                            # Check Observation_Value, Observation_Value2, etc. for direct value mapping
                            # These are typically used when a specific source value maps to a pydantic_value
                            for val_col_suffix in ["", "2", "3", "4"]:
                                obs_val_col_name = map_rule.get(f"Observation_Value{val_col_suffix}")
                                if obs_col_name and obs_val_col_name in source_row and pd.notna(source_row[obs_val_col_name]):
                                    # This logic might need refinement: if Observation_ValueX is a column name
                                    # and its value matches pydantic_value, then use pydantic_value.
                                    # Current interpretation: if Observation_ValueX contains a *value* to be matched in the *source_row[obs_col_name]*
                                    # This seems more aligned with how booleans/enums are handled.
                                    if pydantic_target_value and str(source_row[obs_col_name]) == str(obs_val_col_name): # obs_val_col_name here is a value from mapping
                                        values_for_field.append(pydantic_target_value)
                                    # If Observation_ValueX is meant to be another source column to pull data from, the logic would differ.
                                    # For now, sticking to the idea that Observation_ValueX are values for matching against Observation_ColName's content.

                        # Deduplicate and handle multivalue
                        unique_values = list(pd.Series(values_for_field).dropna().unique())
                        mv_method = mapping_group.iloc[0].get("multiValue_handling_method", "haveBoth")
                        final_value = None
                        if unique_values:
                            if len(unique_values) > 1:
                                if mv_method == "first":
                                    final_value = unique_values[0]
                                elif mv_method == "join_semicolon":
                                    final_value = ";".join(str(v) for v in unique_values)
                                else: # Default is haveBoth (collect_all for list types)
                                    final_value = unique_values
                            else:
                                final_value = unique_values[0]
                        
                        # Assign to item_pydantic and item_excel (potentially nested)
                        # For item_excel, the pydantic_field_name is the flattened key
                        item_excel[pydantic_field_name] = final_value
                        # For item_pydantic, we need to unflatten if necessary
                        keys = pydantic_field_name.split("::")
                        d = item_pydantic
                        for i, key in enumerate(keys):
                            if i == len(keys) - 1:
                                d[key] = final_value
                            else:
                                d = d.setdefault(key, {})
                    
                    transformed_data_for_excel.append(item_excel)
                    try:
                        model_instance = models[model_name](**item_pydantic)
                        transformed_data_for_pydantic.append(model_instance.model_dump(exclude_none=True))
                    except Exception as e:
                        errors.append({"row_identifier": f"Wide format, index {idx}", "error": str(e), "data": item_pydantic})

            elif data_format == "Long":
                if not all([patient_id_col_long, obs_name_col_long, obs_value_col_long]):
                    st.error("For Long format, Patient ID, Observation Name, and Observation Value column names must be specified.")
                    st.stop()

                for patient_id, group_df in data_df.groupby(patient_id_col_long):
                    item_pydantic = {}
                    item_excel = {patient_id_col_long: patient_id} # Add patient ID to excel output
                    
                    for pydantic_field_name, mapping_rules_for_field in mapping_df.groupby("pydantic_field"):
                        values_for_field = []
                        # In long format, Observation_ColName in mapping refers to a value in obs_name_col_long of source data
                        for _, map_rule in mapping_rules_for_field.iterrows():
                            target_obs_name = map_rule.get("Observation_ColName") # This is the value to find in source's obs_name_col_long
                            pydantic_target_value = map_rule.get("pydantic_value") # Value to set if condition met (e.g. True for bool)
                            
                            # Find rows in the patient's data that match the target_obs_name
                            relevant_source_rows = group_df[group_df[obs_name_col_long] == target_obs_name]
                            
                            for _, source_row in relevant_source_rows.iterrows():
                                source_obs_value = source_row[obs_value_col_long]
                                
                                if pydantic_target_value: # If pydantic_value is specified, we need to match source_obs_value against it
                                                          # OR, if pydantic_value is for bool/enum, it IS the value to be used.
                                    # For bools/enums, pydantic_value is the target. We check if the source_obs_value matches an Observation_ValueX from mapping.
                                    # If Observation_ValueX is blank, then any source_obs_value for that target_obs_name implies pydantic_target_value.
                                    # This part needs careful handling based on mapping definition.
                                    
                                    # Scenario 1: pydantic_value is set (e.g. True, \"Male\").
                                    # We need to check if source_obs_value matches one of the Observation_Value[1-4] in the mapping rule.
                                    # If it does, then this pydantic_field gets pydantic_target_value.
                                    matched_observation_value = False
                                    has_specific_observation_values = False
                                    for val_col_suffix in ["", "2", "3", "4"]:
                                        map_obs_val = map_rule.get(f"Observation_Value{val_col_suffix}")
                                        if map_obs_val: # If there's a specific value in mapping's Observation_ValueX
                                            has_specific_observation_values = True
                                            if str(source_obs_value) == str(map_obs_val):
                                                values_for_field.append(pydantic_target_value)
                                                matched_observation_value = True
                                                break # Found a match for this source_row
                                    if not has_specific_observation_values and pd.notna(source_obs_value):
                                        # If no specific Observation_ValueX in mapping, but pydantic_value is set (e.g. for a direct mapping of a category)
                                        # This case is tricky. If pydantic_value is 'True', and Observation_ColName is 'is_urgent',
                                        # and source has 'is_urgent: Yes', we need Observation_Value='Yes' in mapping.
                                        # Let's assume for now that if pydantic_value is set, one of Observation_ValueX must match source_obs_value.
                                        # If this assumption is wrong, the logic for direct assignment of pydantic_value needs to be added.
                                        pass # Covered by the loop above
                                        
                                else: # pydantic_value is blank (e.g. for str, int, float where we take the value directly)
                                    if pd.notna(source_obs_value):
                                        values_for_field.append(source_obs_value)

                        # Deduplicate and handle multivalue (same as wide format)
                        unique_values = list(pd.Series(values_for_field).dropna().unique())
                        mv_method = mapping_rules_for_field.iloc[0].get("multiValue_handling_method", "haveBoth")
                        final_value = None
                        if unique_values:
                            if len(unique_values) > 1:
                                if mv_method == "first":
                                    final_value = unique_values[0]
                                elif mv_method == "join_semicolon":
                                    final_value = ";".join(str(v) for v in unique_values)
                                else: # Default is haveBoth (collect_all for list types)
                                    final_value = unique_values
                            else:
                                final_value = unique_values[0]
                        
                        item_excel[pydantic_field_name] = final_value
                        keys = pydantic_field_name.split("::")
                        d = item_pydantic
                        for i, key in enumerate(keys):
                            if i == len(keys) - 1:
                                d[key] = final_value
                            else:
                                d = d.setdefault(key, {})
                    
                    transformed_data_for_excel.append(item_excel)
                    try:
                        # Add patient_id to pydantic item if it's a field in the model
                        # This assumes patient_id_col_long from input is the value for a field named 'patient_id' or similar in Pydantic model
                        # A more robust way would be to have a mapping for patient_id itself.
                        # For now, let's check if 'patient_id' (or a common variant) is in the model and not yet set.
                        if models[model_name].model_fields.get(patient_id_col_long) and patient_id_col_long not in item_pydantic:
                             item_pydantic[patient_id_col_long] = patient_id
                        elif models[model_name].model_fields.get("patient_id") and "patient_id" not in item_pydantic:
                             item_pydantic["patient_id"] = patient_id
                        
                        model_instance = models[model_name](**item_pydantic)
                        transformed_data_for_pydantic.append(model_instance.model_dump(exclude_none=True))
                    except Exception as e:
                        errors.append({"row_identifier": f"Long format, {patient_id_col_long}: {patient_id}", "error": str(e), "data": item_pydantic})
            
            # Output JSON
            if transformed_data_for_pydantic:
                json_bytes = io.BytesIO(json.dumps(transformed_data_for_pydantic, indent=2).encode("utf-8"))
                st.download_button(
                    label="Download Transformed Data (JSON)",
                    data=json_bytes,
                    file_name="transformed_data.json",
                    mime="application/json",
                    key="download_json_transformed"
                )
            else:
                st.warning("No data was transformed. Check your mapping and source data.")

            # Output Flattened Excel
            if transformed_data_for_excel:
                df_excel_output = pd.DataFrame(transformed_data_for_excel)
                # Ensure patient_id column is first if it exists (for long format)
                if data_format == "Long" and patient_id_col_long in df_excel_output.columns:
                    cols = [patient_id_col_long] + [col for col in df_excel_output.columns if col != patient_id_col_long]
                    df_excel_output = df_excel_output[cols]
                
                excel_output_io = io.BytesIO()
                with pd.ExcelWriter(excel_output_io, engine='xlsxwriter') as writer:
                    df_excel_output.to_excel(writer, index=False, sheet_name='Flattened_Data')
                excel_output_io.seek(0)
                st.download_button(
                    label="Download Flattened Transformed Data (Excel)",
                    data=excel_output_io,
                    file_name="transformed_data_flattened.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_flattened"
                )

            if errors:
                st.error("Errors occurred during transformation:")
                st.json(errors)

# --- Section 3: Auto-Mapping (LLM-based Mapping Suggestion) ---
if st.session_state["active_section"] == "smart":
    st.header("Step 3: Auto-Mapping (LLM-based Mapping Suggestion)")
    # --- Privacy/LLM warning box ---
    st.warning("""
    ⚠️ **Privacy Warning:** This function will summarize each column (if your database is wide) or each element in the Observation column (if your database is long), create a summary, and then send this summary to an LLM (Large Language Model) to find the best Pydantic mapping for your data and provide the final mapping for you.\n\n**This process can expose your patient data to the LLM provider.** Unless you are using a local LLM (e.g., via Ollama) or have access to a secure, compliant LLM (such as Mount Sinai Azure OpenAI, which is HIPAA-compliant), your data may be sent to external servers.\n\n**Review your privacy requirements before proceeding.**
    """)
    st.write("Automatically generate a mapping template by analyzing your data and Pydantic model using an LLM.")

    # --- Data Upload & Format Selection ---
    data_file = st.file_uploader("Upload data file (.csv, .xlsx)", type=["csv", "xlsx"], key="auto_data_file")
    st.subheader("Source Data Format")
    data_format = st.radio("1️⃣ Select your source data format:", ("Wide", "Long"), key="auto_data_format_select")
    patient_id_col_long = obs_name_col_long = obs_value_col_long = ""
    data_columns = []
    df = None
    if data_file is not None:
        if data_file.name.endswith(".csv"):
            df = pd.read_csv(data_file)
        else:
            df = pd.read_excel(data_file)
        data_columns = list(df.columns)
    if data_format == "Long":
        st.write("2️⃣ Please specify the column names for the long format data:")
        if data_columns:
            patient_id_col_long = st.selectbox("2.1 Patient ID Column Name", data_columns, key="auto_patient_id_col_long")
            obs_name_col_long = st.selectbox("2.2 Observation Name Column Name", data_columns, key="auto_obs_name_col_long")
            obs_value_col_long = st.selectbox("2.3 Observation Value Column Name", data_columns, key="auto_obs_value_col_long")
        else:
            patient_id_col_long = st.text_input("2.1 Patient ID Column Name", key="auto_patient_id_col_long")
            obs_name_col_long = st.text_input("2.2 Observation Name Column Name", key="auto_obs_name_col_long")
            obs_value_col_long = st.text_input("2.3 Observation Value Column Name", key="auto_obs_value_col_long")

    # --- Pydantic Model Input ---
    py_file = st.file_uploader("Upload .py file with Pydantic model", type=["py"], key="auto_py_file")
    code_text = st.text_area("Or paste your Pydantic model code here", height=200, key="auto_code_text")
    models = None
    schema_module = None
    if py_file is not None:
        code_str = py_file.read().decode('utf-8')
        import types
        schema_module = types.ModuleType('user_schema')
        exec(code_str, schema_module.__dict__)
        models = {k: v for k, v in schema_module.__dict__.items() if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel}
    elif code_text.strip():
        import types
        schema_module = types.ModuleType('user_schema')
        exec(code_text, schema_module.__dict__)
        models = {k: v for k, v in schema_module.__dict__.items() if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel}
    model_name = None
    if models:
        model_names = list(models.keys())
        # Move last element to be default
        if len(model_names) > 1:
            model_names = model_names[:-1] + [model_names[-1]]
        model_name = st.selectbox("Select model to use", model_names, index=len(model_names)-1, key="auto_model_select")

    # --- LLM Provider Selection ---
    st.subheader("LLM Provider")
    llm_provider = st.radio(
        "Choose LLM Provider",
        ("OpenAI", "Azure OpenAI", "OpenAI-Compatible (Ollama, Fireworks, RunPod, etc.)"),
        key="llm_provider_select"
    )
    openai_api_key = st.text_input("API Key", type="password", key="llm_api_key")
    # Set default model names based on provider
    if llm_provider == "OpenAI" or llm_provider == "Azure OpenAI":
        default_model = "gpt-4.1-mini"
    else:
        default_model = "hermes3:8b-llama3.1-q8_0"
    llm_model = st.text_input("Model Name (e.g. gpt-4, gpt-3.5-turbo)", value=default_model, key="llm_model_name")
    azure_endpoint = azure_version = base_url = ""
    if llm_provider == "Azure OpenAI":
        azure_endpoint = st.text_input("Azure Endpoint (e.g. https://YOUR_RESOURCE.openai.azure.com)", key="llm_azure_endpoint")
        azure_version = st.text_input("Azure API Version (e.g. 2023-05-15)", value="2023-05-15", key="llm_azure_version")
    elif llm_provider == "OpenAI-Compatible (Ollama, Fireworks, RunPod, etc.)":
        base_url = st.text_input(
            "Base URL (must be OpenAI-compatible, e.g. http://localhost:11434/v1 for Ollama)",
            key="llm_base_url"
        )
        st.caption("Your base_url must be OpenAI-compatible. See the documentation for your provider.")

    # --- Data Summary ---
    data_summary = ""
    if df is not None:
        if data_format == "Wide":
            summaries = []
            for idx, col in enumerate(df.columns, 1):
                col_data = df[col]
                summary = f"{idx}. Column: {col}\nType: {col_data.dtype}\nUnique values: {col_data.unique()[:10]}\nSample: {col_data.head(3).tolist()}\n"
                summaries.append(summary)
            data_summary = "\n---\n".join(summaries)
        elif data_format == "Long":
            if not all([patient_id_col_long, obs_name_col_long, obs_value_col_long]):
                st.warning("Please specify all required columns for long format.")
            else:
                obs_names = df[obs_name_col_long].unique()
                summaries = []
                for idx, obs in enumerate(obs_names, 1):
                    vals = df[df[obs_name_col_long] == obs][obs_value_col_long]
                    summary = f"{idx}. Observation: {obs}\nValue type: {vals.dtype}\nUnique values: {vals.unique()[:10]}\nSample: {vals.head(3).tolist()}\n"
                    summaries.append(summary)
                data_summary = "\n---\n".join(summaries)
        st.text_area("Data Summary (for LLM)", value=data_summary, height=200, key="auto_data_summary", disabled=True)

    # --- Smart Mode User Warnings and Progress ---
    st.info("This process may take a couple of minutes depending on the model and amount of data.")
    progress_placeholder = st.empty()

    # --- LLM Prompt Construction ---
    def build_llm_prompt(pydantic_code, model_name, data_summary):
        return f"""
You are an expert medical data engineer. Your job is to create a mapping template that links EHR/database observations or columns to fields in a Pydantic model for downstream data transformation and validation.

Instructions:
- Carefully review the Pydantic model and the data summary below.
- For each observation/column in the data, select the most appropriate Pydantic field(s) to map to. If multiple database values map to a single Pydantic field, create multiple rows for that field.
- For boolean and enum fields, create a row for each possible value (e.g., True/False, or each enum option).
- For fields of type int, float, or str, leave the 'pydantic_value' column blank.
- For boolean and enum fields, set 'pydantic_value' to the value being mapped (e.g., True, False, or the enum value).
- For list fields, set 'multiValue_handling_method' to 'collect_all'. For single-value fields, use 'haveBoth'.
- Use 'smart_exact_match' for 'evaluation_method' unless a more specific method is needed.
- If a mapping requires matching specific source values, use the 'Observation_Value' columns to specify those values (up to 4 per row).
- Output your mapping as a JSON list, where each mapping is an object with the following fields:
  - pydantic_field
  - pydantic_type
  - evaluation_method
  - multiValue_handling_method
  - pydantic_value
  - Observation_ColName
  - Observation_Value
  - Observation_Value2
  - Observation_Value3
  - Observation_Value4

Pydantic Model ({model_name}):
{pydantic_code}

Data Summary:
{data_summary}

Example output:
[
  {{
    "pydantic_field": "patient_id",
    "pydantic_type": "str",
    "evaluation_method": "smart_exact_match",
    "multiValue_handling_method": "haveBoth",
    "pydantic_value": "",
    "Observation_ColName": "patient_identifier",
    "Observation_Value": "",
    "Observation_Value2": "",
    "Observation_Value3": "",
    "Observation_Value4": ""
  }},
  ...
]

Be concise, accurate, and ensure the output is valid JSON. Do not include any explanation or commentary—output only the JSON list.
"""

    # --- LLM Call & Output ---
    if st.button("Suggest Mapping with LLM", key="auto_llm_btn"):
        if not (models and model_name and data_summary and openai_api_key):
            st.error("Please provide all required inputs (model, data, API key, etc.)")
            st.stop()
        pydantic_code = code_str if py_file is not None else code_text
        prompt = build_llm_prompt(pydantic_code, model_name, data_summary)
        try:
            progress_placeholder.info("Contacting LLM provider...")
            import os
            from openai import OpenAI, AzureOpenAI
            client = None
            response = None
            if llm_provider == "Azure OpenAI":
                client = AzureOpenAI(
                    api_key=openai_api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=azure_version
                )
                progress_placeholder.info("Sending prompt to Azure OpenAI...")
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
            elif llm_provider == "OpenAI-Compatible (Ollama, Fireworks, RunPod, etc.)":
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=base_url
                )
                progress_placeholder.info("Sending prompt to OpenAI-compatible endpoint...")
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
            else:
                client = OpenAI(api_key=openai_api_key)
                progress_placeholder.info("Sending prompt to OpenAI...")
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
            progress_placeholder.info("Waiting for LLM response...")
            llm_content = response.choices[0].message.content
            progress_placeholder.info("Parsing LLM output...")
            import json
            try:
                mapping_list = json.loads(llm_content)
            except Exception:
                import re
                match = re.search(r'\[.*\]', llm_content, re.DOTALL)
                if match:
                    mapping_list = json.loads(match.group(0))
                else:
                    progress_placeholder.error("Could not parse LLM output as JSON.\n\nOutput was:\n" + llm_content)
                    st.stop()
            mapping_df = pd.DataFrame(mapping_list)
            progress_placeholder.success("Mapping generated!")
            st.dataframe(mapping_df)
            csv_bytes = mapping_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Mapping (CSV)",
                data=csv_bytes,
                file_name="auto_mapping.csv",
                mime="text/csv",
                key="auto_mapping_csv_btn"
            )
            excel_bytes = io.BytesIO()
            with pd.ExcelWriter(excel_bytes, engine='xlsxwriter') as writer:
                mapping_df.to_excel(writer, index=False, sheet_name='mapping')
            excel_bytes.seek(0)
            st.download_button(
                label="Download Mapping (Excel)",
                data=excel_bytes,
                file_name="auto_mapping.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="auto_mapping_excel_btn"
            )
        except Exception as e:
            progress_placeholder.error(f"LLM call failed: {e}")
