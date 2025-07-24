#!/usr/bin/env python3
"""
Test script to verify ID column selection and extraction logic.
This tests the actual behavior of the ID extraction logic.
"""

import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/Users/as/Documents/GIT/awesome_extraction_with_LLM_ExtraCTOps')

def test_id_extraction_logic():
    """Test the ID extraction logic from the app."""
    
    # Load the sample CSV
    csv_path = "/Users/as/Documents/GIT/awesome_extraction_with_LLM_ExtraCTOps/apps/Data2Pydantic_Map/sample_database_v2.csv"
    df = pd.read_csv(csv_path)
    
    print("=== CSV Data Info ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Test ID extraction logic with different scenarios
    print("\n=== Testing ID Extraction Logic ===")
    
    # Scenario 1: No ID column selected (should use row_X)
    print("\n1. No ID column selected:")
    for i in range(3):
        row_data = df.iloc[i].to_dict()
        id_col = None  # No ID column selected
        
        if id_col and id_col in row_data:
            unique_id = str(row_data[id_col])
        else:
            unique_id = f"row_{i+1}"
        
        print(f"   Row {i}: ID = '{unique_id}'")
    
    # Scenario 2: PatientID column selected (should use actual values)
    print("\n2. PatientID column selected:")
    for i in range(3):
        row_data = df.iloc[i].to_dict()
        id_col = "PatientID"  # ID column selected
        
        if id_col and id_col in row_data:
            unique_id = str(row_data[id_col])
        else:
            unique_id = f"row_{i+1}"
        
        print(f"   Row {i}: ID = '{unique_id}' (from PatientID: {row_data.get(id_col, 'N/A')})")
    
    # Scenario 3: Test with empty/falsy values
    print("\n3. Testing with falsy values:")
    test_data = pd.DataFrame({
        'PatientID': ['P001', '', 0, 'P004', None],
        'data': ['A', 'B', 'C', 'D', 'E']
    })
    
    for i in range(len(test_data)):
        row_data = test_data.iloc[i].to_dict()
        id_col = "PatientID"
        
        if id_col and id_col in row_data:
            unique_id = str(row_data[id_col])
        else:
            unique_id = f"row_{i+1}"
        
        print(f"   Row {i}: ID = '{unique_id}' (from PatientID: {repr(row_data.get(id_col, 'N/A'))})")

if __name__ == "__main__":
    test_id_extraction_logic()
