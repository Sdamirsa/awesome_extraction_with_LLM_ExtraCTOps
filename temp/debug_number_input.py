#!/usr/bin/env python3
"""
Test script to debug the exact number input rendering issue
"""

def test_widget_rendering_logic():
    """Test the exact logic used in the widget rendering"""
    
    print("=== Testing Widget Rendering Logic ===")
    
    def test_mode_index_calculation(current_value, session_mode, scenario_name):
        """Test the mode_index calculation logic"""
        print(f"\n--- {scenario_name} ---")
        print(f"current_value: {current_value}")
        print(f"session_mode: {session_mode}")
        
        # This is the NEW logic from the fix
        if session_mode is not None:
            # User has made a selection, use it
            mode_index = 1 if session_mode == "Number" else 0
            print(f"Using session_mode: mode_index = {mode_index}")
        else:
            # No user selection yet, use current_value to determine default
            mode_index = 1 if (current_value is not None) else 0
            print(f"Using current_value: mode_index = {mode_index}")
        
        modes = ["(None)", "Number"]
        choice = modes[mode_index]
        print(f"Radio shows: {choice}")
        
        shows_number_input = (choice == "Number")
        print(f"Number input visible: {shows_number_input}")
        
        return shows_number_input
    
    # Test the exact scenarios that might be happening
    scenarios = [
        # Scenario 1: Fresh page load, no user interaction
        (None, None, "Fresh load - no value, no user selection"),
        
        # Scenario 2: User just selected "Number" 
        (None, "Number", "User selected Number on None field"),
        
        # Scenario 3: User selected None
        (None, "(None)", "User selected (None)"),
        
        # Scenario 4: Loading row with existing value
        (25, None, "Loading row with existing value"),
        
        # Scenario 5: User overrode existing value to Number
        (25, "Number", "User selected Number on existing field"),
        
        # Scenario 6: User overrode existing value to None
        (25, "(None)", "User selected (None) on existing field"),
        
        # Scenario 7: Loading default values (from generate_default_values)
        (None, None, "Loading with generate_default_values (None)"),
    ]
    
    results = []
    for current_value, session_mode, scenario_name in scenarios:
        result = test_mode_index_calculation(current_value, session_mode, scenario_name)
        results.append((scenario_name, result))
    
    print("\n=== Results Summary ===")
    for scenario, shows_input in results:
        status = "✅ SHOWS" if shows_input else "❌ HIDDEN"
        print(f"{status} | {scenario}")
    
    print("\n=== Analysis ===")
    print("The issue you're experiencing suggests:")
    print("1. When you select 'Number', session_mode should be 'Number'")
    print("2. But somehow the number input still doesn't appear")
    print("3. When you select another field, it triggers a rerun and then it works")
    print("")
    print("Possible causes:")
    print("A) The session_mode value is not being set correctly by the radio")
    print("B) The choice variable is not reflecting the session_mode correctly")
    print("C) There's a race condition in Streamlit's widget rendering")
    print("D) The key naming is inconsistent between radio and number_input")
    
    return results

def test_key_naming():
    """Test if key naming might be causing issues"""
    print("\n=== Testing Key Naming ===")
    
    # Simulate the key generation logic
    prefix = ""
    field_name = "age"
    key_base = f"{prefix}{field_name}"
    mode_key = f"{key_base}_mode"
    value_key = f"{key_base}"
    
    print(f"field_name: {field_name}")
    print(f"key_base: {key_base}")
    print(f"mode_key (radio): {mode_key}")
    print(f"value_key (number_input): {value_key}")
    
    # Check for potential conflicts
    if mode_key == value_key:
        print("❌ ERROR: Mode key and value key are the same!")
    else:
        print("✅ Keys are different (good)")

if __name__ == "__main__":
    test_widget_rendering_logic()
    test_key_naming()
    
    print("\n" + "="*50)
    print("DEBUGGING SUGGESTIONS:")
    print("1. Add debug prints in the actual app to see what session_mode value is")
    print("2. Check if the radio button key and number_input key are conflicting")
    print("3. Try adding st.write() statements to show session state values")
    print("4. The issue might be that choice != session_mode due to Streamlit timing")
