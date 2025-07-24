#!/usr/bin/env python3
"""
Test script to verify the inject_previous_extractions fix
"""

import pandas as pd
import json
import sys
import os

# Add the app directory to the path
sys.path.insert(0, '/Users/as/Documents/GIT/awesome_extraction_with_LLM_ExtraCTOps/apps/manual_extraction')

# Mock streamlit for testing
class MockStreamlit:
    def __init__(self):
        self.session_state = {}
        self.messages = []
    
    def error(self, msg):
        self.messages.append(('error', msg))
        print(f"ERROR: {msg}")
    
    def success(self, msg):
        self.messages.append(('success', msg))
        print(f"SUCCESS: {msg}")
    
    def warning(self, msg):
        self.messages.append(('warning', msg))
        print(f"WARNING: {msg}")

# Mock uploaded file
class MockUploadedFile:
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        with open(filepath, 'r') as f:
            self.content = f.read()
        self.position = 0
    
    def read(self):
        return self.content.encode('utf-8')
    
    def seek(self, pos):
        self.position = pos

# Replace streamlit with mock
import streamlit as st
mock_st = MockStreamlit()
sys.modules['streamlit'] = mock_st
st = mock_st

# Now import the app functions
from app import parse_uploaded_file, inject_previous_extractions

def test_parse_uploaded_file():
    """Test that parse_uploaded_file correctly handles previous export format"""
    print("Testing parse_uploaded_file with previous export format...")
    
    # Create mock uploaded file
    mock_file = MockUploadedFile('/Users/as/Documents/GIT/awesome_extraction_with_LLM_ExtraCTOps/temp/test_previous_extraction.json')
    
    # Parse the file
    result = parse_uploaded_file(mock_file)
    
    print(f"Result type: {result['type']}")
    print(f"Data type: {type(result['data'])}")
    
    if result['type'] == 'previous_export':
        print("✅ Correctly identified as previous export")
        print(f"✅ Data contains {len(result['data']['extractions'])} extractions")
        return True
    else:
        print("❌ Failed to identify as previous export")
        return False

def test_inject_previous_extractions():
    """Test that inject_previous_extractions works with the fixed DataFrame handling"""
    print("\nTesting inject_previous_extractions...")
    
    # Set up mock session state
    mock_st.session_state = {
        'extractions': [
            {'id': 'row_1', 'values': {}, 'review_status': 'not_reviewed'},
            {'id': 'row_2', 'values': {}, 'review_status': 'not_reviewed'}
        ],
        'id_column': 'text'
    }
    
    # Create mock uploaded file
    mock_file = MockUploadedFile('/Users/as/Documents/GIT/awesome_extraction_with_LLM_ExtraCTOps/temp/test_previous_extraction.json')
    
    try:
        # Call the function
        inject_previous_extractions(mock_file)
        
        # Check if we got success message
        success_messages = [msg for msg_type, msg in mock_st.messages if msg_type == 'success']
        if success_messages:
            print(f"✅ Function completed successfully: {success_messages[0]}")
            return True
        else:
            print("❌ No success message found")
            error_messages = [msg for msg_type, msg in mock_st.messages if msg_type == 'error']
            if error_messages:
                print(f"Error messages: {error_messages}")
            return False
            
    except Exception as e:
        print(f"❌ Function failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("Testing the DataFrame boolean check fix...")
    
    test1_passed = test_parse_uploaded_file()
    test2_passed = test_inject_previous_extractions()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The fix is working correctly.")
    else:
        print("\n❌ Some tests failed.")
