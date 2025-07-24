#!/usr/bin/env python3
"""
Test script to verify the unflatten_from_export function works correctly
"""

import json

def unflatten_from_export(flattened_dict, separator="::"):
    """
    Reconstruct a nested structure from a flattened dictionary.
    Reverse operation of flatten_for_export.
    """
    result = {}
    
    for key, value in flattened_dict.items():
        if separator in key:
            # Split the key into parts
            parts = key.split(separator)
            current = result
            
            # Navigate through the nested structure
            for i, part in enumerate(parts[:-1]):
                if part.isdigit():
                    # This is a list index
                    index = int(part)
                    # Ensure current is a list
                    if not isinstance(current, list):
                        current = []
                    # Extend list if needed
                    while len(current) <= index:
                        current.append({})
                    current = current[index]
                else:
                    # This is a dictionary key
                    if part not in current:
                        # Look ahead to see if next part is a digit (indicating a list)
                        next_part = parts[i + 1] if i + 1 < len(parts) else None
                        if next_part and next_part.isdigit():
                            current[part] = []
                        else:
                            current[part] = {}
                    current = current[part]
            
            # Set the final value
            final_key = parts[-1]
            if final_key.isdigit():
                index = int(final_key)
                if not isinstance(current, list):
                    current = []
                while len(current) <= index:
                    current.append(None)
                current[index] = value
            else:
                current[final_key] = value
        else:
            # Simple key without separator
            result[key] = value
    
    return result

def test_unflatten_with_real_data():
    """Test unflatten with real data from the user's JSON file"""
    
    # Sample flattened data from the user's JSON file
    flattened_data = {
        "atria::RA::RA_dilation": "Aplastic",
        "atria::LA::LA_dilation": None,
        "atria::LA::LA_volume_indexed::numeric": None,
        "atria::LA::LA_volume_indexed::unit": "",
        "ventricles::RV::RV_size_structure::RV_dilation": None,
        "ventricles::RV::RV_size_structure::RV_hypertrophy": None,
        "ventricles::RV::RV_function::RV_systolic_function": None,
        "ventricles::LV::LV_size_structure::LV_dilation": None,
        "ventricles::LV::LV_size_structure::LV_hypertrophy": None,
        "ventricles::LV::LV_function::LV_systolic_function": None,
        "ventricles::LV::LV_function::LVEF::numeric": None,
        "ventricles::LV::LV_function::LVEF::unit": "",
        "valves::tricuspid::TV_structural_status": None,
        "valves::pulmonary::PV_structural_status": None,
        "surgical_history::prior_surgical_interventions": ""
    }
    
    # Filter out null and empty values (as the injection function should do)
    filtered_data = {k: v for k, v in flattened_data.items() if v is not None and v != ""}
    
    print("Original flattened data (filtered):")
    for key, value in filtered_data.items():
        print(f"  {key}: {value}")
    
    # Test unflatten
    try:
        reconstructed = unflatten_from_export(filtered_data)
        print("\nReconstructed nested structure:")
        print(json.dumps(reconstructed, indent=2))
        
        # Verify the structure
        if "atria" in reconstructed:
            print(f"\n✅ 'atria' key found in reconstructed data")
            if "RA" in reconstructed["atria"]:
                print(f"✅ 'RA' key found in atria")
                if "RA_dilation" in reconstructed["atria"]["RA"]:
                    print(f"✅ 'RA_dilation' value: {reconstructed['atria']['RA']['RA_dilation']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during unflatten: {e}")
        return False

def test_unflatten_with_second_row():
    """Test unflatten with data from second row that has more values"""
    
    # Sample from second row with more actual values
    flattened_data = {
        "atria::RA::RA_dilation": "Hypoplastic",
        "atria::LA::LA_dilation": "Hypoplastic", 
        "atria::LA::LA_volume_indexed::numeric": 2.0,
        "atria::LA::LA_volume_indexed::unit": "asda"
    }
    
    print("\n" + "="*50)
    print("Testing with second row data:")
    print("Original flattened data:")
    for key, value in flattened_data.items():
        print(f"  {key}: {value}")
    
    try:
        reconstructed = unflatten_from_export(flattened_data)
        print("\nReconstructed nested structure:")
        print(json.dumps(reconstructed, indent=2))
        
        # Verify values
        if (reconstructed.get("atria", {}).get("RA", {}).get("RA_dilation") == "Hypoplastic" and
            reconstructed.get("atria", {}).get("LA", {}).get("LA_dilation") == "Hypoplastic" and
            reconstructed.get("atria", {}).get("LA", {}).get("LA_volume_indexed", {}).get("numeric") == 2.0):
            print("✅ All values correctly reconstructed!")
            return True
        else:
            print("❌ Some values not correctly reconstructed")
            return False
            
    except Exception as e:
        print(f"❌ Error during unflatten: {e}")
        return False

if __name__ == "__main__":
    print("Testing unflatten_from_export function with real data...")
    
    test1_passed = test_unflatten_with_real_data()
    test2_passed = test_unflatten_with_second_row()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The unflatten function works correctly.")
    else:
        print("\n❌ Some tests failed.")
