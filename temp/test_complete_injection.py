#!/usr/bin/env python3
"""
Test the complete injection process with the user's real data
"""

import json

def test_injection_process():
    """Test the complete injection process with real data"""
    
    # Load the user's real JSON file
    with open('/Users/as/Downloads/extractions_20250705_190728.json', 'r') as f:
        user_data = json.load(f)
    
    print("Loaded user data:")
    print(f"  - Total records: {len(user_data)}")
    print(f"  - Record IDs: {[record.get('id') for record in user_data]}")
    
    # Simulate current session state (what would be in the app)
    current_extractions = [
        {"id": 111, "values": {}, "review_status": "not_reviewed"},
        {"id": 222, "values": {}, "review_status": "not_reviewed"},
        {"id": "row_3", "values": {}, "review_status": "not_reviewed"},
        {"id": "row_4", "values": {}, "review_status": "not_reviewed"}
    ]
    
    print(f"\nCurrent session state:")
    print(f"  - Total extractions: {len(current_extractions)}")
    print(f"  - Current IDs: {[ext.get('id') for ext in current_extractions]}")
    
    # Test the injection logic
    injected_count = 0
    matched_count = 0
    
    # Create a mapping of previous extractions by ID
    previous_by_id = {}
    for prev_extraction in user_data:
        if isinstance(prev_extraction, dict):
            # Handle different ID field formats
            extraction_id = None
            if "id" in prev_extraction:
                extraction_id = prev_extraction["id"]
            elif "row_index" in prev_extraction:
                extraction_id = f"row_{prev_extraction['row_index'] + 1}"
            
            if extraction_id:
                previous_by_id[str(extraction_id)] = prev_extraction
    
    print(f"\nPrevious extractions mapping:")
    for id_key, data in previous_by_id.items():
        print(f"  - ID {id_key}: {data.get('review_status', 'unknown')} status")
    
    # Simulate the injection process
    def unflatten_from_export(flattened_dict, separator="::"):
        """Reconstruct a nested structure from a flattened dictionary."""
        result = {}
        
        for key, value in flattened_dict.items():
            if separator in key:
                parts = key.split(separator)
                current = result
                
                for i, part in enumerate(parts[:-1]):
                    if part.isdigit():
                        index = int(part)
                        if not isinstance(current, list):
                            current = []
                        while len(current) <= index:
                            current.append({})
                        current = current[index]
                    else:
                        if part not in current:
                            next_part = parts[i + 1] if i + 1 < len(parts) else None
                            if next_part and next_part.isdigit():
                                current[part] = []
                            else:
                                current[part] = {}
                        current = current[part]
                
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
                result[key] = value
        
        return result
    
    # Process each current extraction
    for i, current_extraction in enumerate(current_extractions):
        current_id = str(current_extraction.get("id", f"row_{i+1}"))
        
        if current_id in previous_by_id:
            matched_count += 1
            prev_data = previous_by_id[current_id]
            
            print(f"\n--- Processing ID {current_id} ---")
            
            # Separate flattened extraction data from metadata
            flattened_extraction_data = {}
            metadata = {}
            
            for key, value in prev_data.items():
                if key.startswith(("id", "row_index", "review_status", "review_timestamp", "raw_")):
                    metadata[key] = value
                else:
                    if value is not None and value != "":
                        flattened_extraction_data[key] = value
            
            print(f"  - Metadata fields: {list(metadata.keys())}")
            print(f"  - Flattened extraction fields: {len(flattened_extraction_data)}")
            
            # Reconstruct nested structure
            if flattened_extraction_data:
                try:
                    reconstructed_values = unflatten_from_export(flattened_extraction_data)
                    
                    # Update the current extraction
                    current_extraction["values"] = reconstructed_values
                    current_extraction["review_status"] = metadata.get("review_status", "manually_reviewed")
                    current_extraction["review_timestamp"] = metadata.get("review_timestamp")
                    injected_count += 1
                    
                    print(f"  - ✅ Successfully reconstructed {len(reconstructed_values)} top-level fields")
                    print(f"  - Top-level keys: {list(reconstructed_values.keys())}")
                    
                except Exception as e:
                    print(f"  - ❌ Failed to reconstruct: {e}")
    
    print(f"\n=== INJECTION SUMMARY ===")
    print(f"Matched {matched_count} records by ID")
    print(f"Successfully injected {injected_count} records")
    
    # Show the final state
    print(f"\n=== FINAL STATE ===")
    for i, ext in enumerate(current_extractions):
        print(f"ID {ext['id']}:")
        print(f"  - Review status: {ext['review_status']}")
        print(f"  - Has values: {bool(ext['values'])}")
        if ext['values']:
            print(f"  - Value keys: {list(ext['values'].keys())}")
    
    return injected_count > 0

if __name__ == "__main__":
    print("Testing complete injection process with user's real data...")
    
    success = test_injection_process()
    
    if success:
        print("\n🎉 Injection process test completed successfully!")
    else:
        print("\n❌ Injection process test failed.")
