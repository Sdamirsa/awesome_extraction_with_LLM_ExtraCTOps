#!/usr/bin/env python3
"""
Test script to understand the data format flow in the extraction process
"""

import json

# Simulate the extraction data format problem
def test_data_format_consistency():
    """Test to understand the format inconsistency"""
    
    # This is what gets created during manual save (after gather_values_from_state)
    manual_extraction_values = {
        "atria": {
            "RA": {
                "RA_dilation": "Aplastic"
            },
            "LA": {
                "LA_dilation": None,
                "LA_volume_indexed": {
                    "numeric": None,
                    "unit": ""
                }
            }
        },
        "ventricles": {
            "RV": {
                "RV_size_structure": {
                    "RV_dilation": None,
                    "RV_hypertrophy": None
                }
            }
        }
    }
    
    # This is what gets created during auto-initialization (default_values)
    auto_init_values = {
        "atria": None,
        "ventricles": None,
        "valves": None,
        "great_vessels": None
    }
    
    print("=== MANUAL EXTRACTION VALUES (nested) ===")
    print(json.dumps(manual_extraction_values, indent=2))
    
    print("\n=== AUTO INIT VALUES (top-level nulls) ===")
    print(json.dumps(auto_init_values, indent=2))
    
    # Simulate flatten_for_export on both
    def flatten_for_export(obj, prefix="", separator="::"):
        """Simulate the flatten function"""
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
    
    print("\n=== FLATTENED MANUAL VALUES ===")
    flattened_manual = flatten_for_export(manual_extraction_values)
    print(json.dumps(flattened_manual, indent=2))
    
    print("\n=== FLATTENED AUTO INIT VALUES ===")
    flattened_auto = flatten_for_export(auto_init_values)
    print(json.dumps(flattened_auto, indent=2))
    
    print("\n=== ANALYSIS ===")
    print(f"Manual flattened fields: {len(flattened_manual)}")
    print(f"Auto init flattened fields: {len(flattened_auto)}")
    print(f"Formats match: {list(flattened_manual.keys())[:3] == list(flattened_auto.keys())[:3] if flattened_auto else False}")

if __name__ == "__main__":
    test_data_format_consistency()
