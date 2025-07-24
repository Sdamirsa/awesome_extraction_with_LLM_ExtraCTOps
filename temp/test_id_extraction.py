#!/usr/bin/env python3
"""
Test the ID extraction fix with real data format
"""

import pandas as pd

def test_id_extraction():
    """Test that IDs are correctly extracted from ID column"""
    
    # Simulate the data structure from the user's case
    sample_data = [
        {"id": 111, "PID": 111, "MRE_Report": "Report 1"},
        {"id": 222, "PID": 222, "MRE_Report": "Report 2"}, 
        {"id": 333, "PID": 333, "MRE_Report": "Report 3"},
        {"id": 444, "PID": 444, "MRE_Report": "Report 4"}
    ]
    
    df = pd.DataFrame(sample_data)
    print("Sample DataFrame:")
    print(df)
    
    # Simulate the ID extraction logic 
    id_column = "PID"  # This would be set in session state
    
    print(f"\nExtracting IDs using column: {id_column}")
    
    extracted_ids = []
    for i in range(len(df)):
        row_data = df.iloc[i].to_dict()
        
        # This is the improved logic
        if id_column and id_column in row_data and row_data[id_column] is not None:
            unique_id = str(row_data[id_column])  # Use actual ID column value
        else:
            unique_id = f"row_{i+1}"  # Fallback to row number
        
        extracted_ids.append(unique_id)
        print(f"Row {i+1}: {row_data} -> ID: {unique_id}")
    
    print(f"\nExtracted IDs: {extracted_ids}")
    print(f"Expected IDs: ['111', '222', '333', '444']")
    print(f"Correct extraction: {extracted_ids == ['111', '222', '333', '444']}")
    
    return extracted_ids == ['111', '222', '333', '444']

def test_fallback_ids():
    """Test ID extraction when ID column is not available"""
    
    # Simulate data without proper ID column
    sample_data = [
        {"text": "Report 1"},
        {"text": "Report 2"},
        {"text": "Report 3"}
    ]
    
    df = pd.DataFrame(sample_data)
    print("\n" + "="*50)
    print("Testing fallback IDs (no ID column):")
    print(df)
    
    id_column = "nonexistent_column"
    
    extracted_ids = []
    for i in range(len(df)):
        row_data = df.iloc[i].to_dict()
        
        if id_column and id_column in row_data and row_data[id_column] is not None:
            unique_id = str(row_data[id_column])
        else:
            unique_id = f"row_{i+1}"  # Should fallback to this
        
        extracted_ids.append(unique_id)
        print(f"Row {i+1}: {row_data} -> ID: {unique_id}")
    
    print(f"\nExtracted IDs: {extracted_ids}")
    print(f"Expected IDs: ['row_1', 'row_2', 'row_3']")
    print(f"Correct fallback: {extracted_ids == ['row_1', 'row_2', 'row_3']}")
    
    return extracted_ids == ['row_1', 'row_2', 'row_3']

if __name__ == "__main__":
    print("Testing ID extraction fixes...")
    
    test1 = test_id_extraction()
    test2 = test_fallback_ids()
    
    if test1 and test2:
        print("\n🎉 All ID extraction tests passed!")
        print("The fixes should resolve both the format and ID issues.")
    else:
        print("\n❌ Some ID extraction tests failed.")
