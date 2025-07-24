"""
Simple test script for ExtraCTOps Loops functionality

This script creates minimal test data and runs a simple extraction to verify
that the ExtraCTOps_loops module works correctly with the generators.

Run with: python utils/ExtraCTOps_loops/test_simple.py
"""

import asyncio
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.ExtraCTOps_loops import ProcessingConfig, ExtraCTOpsProcessor
from the_pydantics.example_schema import ExampleModel

async def test_simple_extraction():
    """Test simple extraction with minimal data"""
    print("🚀 Testing ExtraCTOps Loops - Simple Extraction")
    print("=" * 50)
    
    # Create minimal test data
    test_data = {
        'id': [1, 2],
        'text_content': [
            'The field1 value is "hello world" and field2 is 42.',
            'Here field1 is "test data" and field2 equals 99.'
        ]
    }
    
    df = pd.DataFrame(test_data)
    
    # Ensure temp directory exists
    temp_dir = project_root / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Save test data
    test_file = temp_dir / "simple_test_data.xlsx"
    df.to_excel(test_file, index=False)
    print(f"📄 Created test data: {test_file}")
    print(f"📊 Test data:\n{df}")
    
    try:
        # Create configuration
        config = ProcessingConfig(
            input_file=str(test_file),
            text_column='text_content',
            uid_column='id',
            pydantic_model=ExampleModel,
            generator_type='ollama',
            model_name='llama3',
            experiment_label='simple_test',
            batch_size=2,
            temperature=0.1,
            max_tokens=500,
            system_message="Extract structured data from the text according to the schema.",
            pre_prompt="Return the extracted information as valid JSON."
        )
        
        print(f"⚙️  Configuration created for generator: {config.generator_type}")
        
        # Create processor
        processor = ExtraCTOpsProcessor(config)
        print("🔧 Processor initialized")
        
        # Run extraction (don't auto-open file)
        print("🔄 Starting extraction...")
        await processor.process_batch(open_file=False)
        
        # Print results summary
        successful = len([r for r in processor.results if r.success])
        failed = len([r for r in processor.results if not r.success])
        total_time = sum(r.execution_time for r in processor.results)
        avg_time = total_time / len(processor.results) if processor.results else 0
        
        print("\n" + "=" * 50)
        print("📈 EXTRACTION RESULTS")
        print("=" * 50)
        print(f"✅ Successful extractions: {successful}")
        print(f"❌ Failed extractions: {failed}")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        print(f"⏱️  Average time per item: {avg_time:.2f}s")
        print(f"📁 Output file: {config.output_file}")
        
        # Show some sample results
        if processor.results:
            print(f"\n🔍 Sample Results:")
            for i, result in enumerate(processor.results[:2]):  # Show first 2
                print(f"  Row {i+1} (UID: {result.uid}):")
                print(f"    Success: {result.success}")
                if result.success and result.extracted_data:
                    print(f"    Extracted: {result.extracted_data}")
                elif result.error:
                    print(f"    Error: {result.error}")
        
        print(f"\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    try:
        success = await test_simple_extraction()
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n💥 Tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
