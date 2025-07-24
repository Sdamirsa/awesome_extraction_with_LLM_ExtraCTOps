"""
Document Handler Module for ExtraCTOps

This module provides utilities for reading various document formats including:
- CSV/Excel files (pandas)
- Text files
- DOCX files 
- PDF files
- JSON files

Used by ExtraCTOps_loops for batch processing of documents.
"""

from pathlib import Path
from typing import Union, Dict, Any, List
import pandas as pd
import json
from pydantic import BaseModel

class DocumentReadError(Exception):
    """Custom exception for document reading errors"""
    pass

def read_csv_excel(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read CSV or Excel files into a pandas DataFrame
    
    Args:
        file_path: Path to CSV or Excel file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        DocumentReadError: If there's an error reading the file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise DocumentReadError(f"File not found: {file_path}")
    
    try:
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise DocumentReadError(f"Unsupported file format for tabular data: {file_path.suffix}")
            
    except Exception as e:
        raise DocumentReadError(f"Error reading file {file_path}: {str(e)}")

def read_text_file(file_path: Union[str, Path]) -> str:
    """
    Read plain text files
    
    Args:
        file_path: Path to text file
        
    Returns:
        str: Text content of the file
        
    Raises:
        DocumentReadError: If there's an error reading the file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise DocumentReadError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise DocumentReadError(f"Error reading text file {file_path}: {str(e)}")

def read_json_file(file_path: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
    """
    Read JSON files
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Union[Dict, List]: Parsed JSON content
        
    Raises:
        DocumentReadError: If there's an error reading the file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise DocumentReadError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise DocumentReadError(f"Error reading JSON file {file_path}: {str(e)}")

def save_dataframe(df: pd.DataFrame, file_path: Union[str, Path], format: str = 'excel') -> None:
    """
    Save DataFrame to file
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        format: Output format ('excel', 'csv', 'json')
        
    Raises:
        DocumentReadError: If there's an error saving the file
    """
    file_path = Path(file_path)
    
    try:
        if format.lower() == 'excel':
            df.to_excel(file_path, index=False)
        elif format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format.lower() == 'json':
            df.to_json(file_path, orient='records', indent=2)
        else:
            raise DocumentReadError(f"Unsupported save format: {format}")
            
    except Exception as e:
        raise DocumentReadError(f"Error saving file {file_path}: {str(e)}")

def flatten_pydantic_object(obj: BaseModel, prefix: str = '') -> Dict[str, Any]:
    """
    Flatten a Pydantic object into a dictionary with dot notation
    
    Args:
        obj: Pydantic BaseModel object
        prefix: Prefix for field names
        
    Returns:
        Dict[str, Any]: Flattened dictionary
    """
    flat_dict = {}
    stack = [(obj, prefix)]
    
    while stack:
        current_obj, current_prefix = stack.pop()
        if isinstance(current_obj, BaseModel):
            current_obj = current_obj.model_dump()
            
        if isinstance(current_obj, dict):
            for key, value in current_obj.items():
                full_key = f"{current_prefix}_{key}" if current_prefix else key
                if isinstance(value, (BaseModel, dict)):
                    stack.append((value, full_key))
                    continue
                flat_dict[full_key] = value
        else:
            # If it's not a dict or BaseModel, store as is
            flat_dict[current_prefix] = current_obj
                    
    return flat_dict
