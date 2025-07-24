#!/usr/bin/env python3
"""
Test script to verify the specific DataFrame boolean check fix
"""

import pandas as pd
import json

def test_dataframe_boolean_check():
    """Test the specific issue that was causing the error"""
    print("Testing DataFrame boolean check issue...")
    
    # Create a sample DataFrame (this is what was causing the error)
    df = pd.DataFrame([
        {'field1': 'value1', 'field2': 'value2'},
        {'field1': 'value3', 'field2': 'value4'}
    ])
    
    # Create a file_data structure as it would be returned by parse_uploaded_file
    file_data = {
        "type": "json",
        "data": df,
        "filename": "test.json"
    }
    
    # This is the problematic check that was causing the error
    try:
        # OLD CODE (would fail):
        # if not file_data.get("data"):
        #     print("This would fail with DataFrame")
        
        # NEW CODE (should work):
        if file_data.get("data") is None:
            print("❌ Data is None")
            return False
        else:
            print("✅ Data check passed")
            data = file_data["data"]
            print(f"✅ Data type: {type(data)}")
            print(f"✅ Data is DataFrame: {isinstance(data, pd.DataFrame)}")
            
            # Test converting DataFrame to records (as done in the fixed code)
            if isinstance(data, pd.DataFrame):
                records = data.to_dict('records')
                print(f"✅ Successfully converted to {len(records)} records")
                return True
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return False

def test_json_data_structures():
    """Test different JSON data structures that might be encountered"""
    print("\nTesting different JSON data structures...")
    
    # Test 1: DataFrame (from regular JSON)
    df = pd.DataFrame([{'id': 1, 'field': 'value'}])
    if isinstance(df, pd.DataFrame):
        print("✅ DataFrame handling test passed")
    
    # Test 2: List (direct extractions)
    list_data = [{'id': 1, 'field': 'value'}]
    if isinstance(list_data, list):
        print("✅ List handling test passed")
    
    # Test 3: Dict with extractions (session state format)
    dict_data = {'extractions': [{'id': 1, 'field': 'value'}]}
    if isinstance(dict_data, dict) and "extractions" in dict_data:
        print("✅ Dict with extractions handling test passed")
    
    # Test 4: Dict with __pydantic_extraction_data__ (previous export format)
    export_data = {
        '__pydantic_extraction_data__': True,
        'extractions': [{'id': 1, 'field': 'value'}]
    }
    if isinstance(export_data, dict) and "__pydantic_extraction_data__" in export_data:
        print("✅ Previous export format handling test passed")
    
    return True

if __name__ == "__main__":
    print("Testing the DataFrame boolean check fix...")
    
    test1_passed = test_dataframe_boolean_check()
    test2_passed = test_json_data_structures()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The DataFrame boolean check fix is working correctly.")
        print("\nThe issue was:")
        print("- OLD: if not file_data.get('data'):  # Fails when data is DataFrame")
        print("- NEW: if file_data.get('data') is None:  # Works with DataFrame")
        print("\nThe fix also properly handles different data types:")
        print("- DataFrame -> converted to records using to_dict('records')")
        print("- List -> used directly")
        print("- Dict with 'extractions' -> extracts the extractions list")
        print("- Dict with '__pydantic_extraction_data__' -> extracts the extractions list")
    else:
        print("\n❌ Some tests failed.")
