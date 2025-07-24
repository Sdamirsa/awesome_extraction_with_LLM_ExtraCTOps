"""
Example Usage of ExtraCTOps Loops

This script demonstrates how to use the ExtraCTOps batch processing functionality
to extract structured data from text using different generators.

Prerequisites:
1. Have a CSV/Excel file with text data and unique IDs
2. Define a Pydantic model for extraction
3. Configure your generator settings

Run with:
    python utils/ExtraCTOps_loops/example_usage.py
"""

import asyncio
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.ExtraCTOps_loops import process_extraction_batch, ProcessingConfig, ExtraCTOpsProcessor
from the_pydantics.example_schema import ExampleModel

async def create_sample_data():
    """Create sample data for demonstration"""
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'text_content': [
            "The patient John Doe is 45 years old and has diabetes.",
            "Mary Smith, age 32, was diagnosed with hypertension.",
            "Robert Johnson, 28 years old, shows symptoms of anxiety.",
            "Lisa Brown is a 55-year-old patient with arthritis.",
            "David Wilson, age 41, has been treated for asthma."
        ],
        'source': ['clinical_note_1', 'clinical_note_2', 'clinical_note_3', 'clinical_note_4', 'clinical_note_5']
    }
    
    df = pd.DataFrame(sample_data)
    sample_file = project_root / "temp" / "sample_extraction_data.xlsx"
    sample_file.parent.mkdir(exist_ok=True)
    df.to_excel(sample_file, index=False)
    
    print(f"Created sample data file: {sample_file}")
    return str(sample_file)

async def example_simple_usage():
    """Example of simple usage with convenience function"""
    print("\n=== Simple Usage Example ===")
    
    # Create sample data
    sample_file = await create_sample_data()
    
    # Simple extraction with convenience function
    await process_extraction_batch(
        input_file=sample_file,
        text_column='text_content',
        uid_column='id',
        pydantic_model=ExampleModel,
        generator_type='ollama',
        model_name='llama3',
        experiment_label='simple_extraction',
        batch_size=3,
        temperature=0.1,
        system_message="Extract patient information from clinical text.",
        pre_prompt="Return the extracted data as valid JSON according to the schema."
    )
    
    print("Simple extraction completed!")

async def example_advanced_usage():
    """Example of advanced usage with custom configuration"""
    print("\n=== Advanced Usage Example ===")
    
    # Create sample data
    sample_file = await create_sample_data()
    
    # Advanced configuration
    config = ProcessingConfig(
        input_file=sample_file,
        text_column='text_content',
        uid_column='id',
        pydantic_model=ExampleModel,
        generator_type='ollama',
        model_name='llama3',
        experiment_label='advanced_extraction',
        batch_size=2,
        backup_interval=2,  # Save backup every 2 items for demo
        max_retries=1,
        temperature=0.05,
        max_tokens=1000,
        system_message="You are a medical information extraction expert. Extract patient data accurately.",
        pre_prompt="Analyze the clinical text and extract patient information as JSON."
    )
    
    # Create processor and run
    processor = ExtraCTOpsProcessor(config)
    await processor.process_batch(open_file=False)  # Don't auto-open file in example
    
    print("Advanced extraction completed!")
    print(f"Results saved to: {config.output_file}")
    
    # Print summary
    successful = len([r for r in processor.results if r.success])
    failed = len([r for r in processor.results if not r.success])
    avg_time = sum(r.execution_time for r in processor.results) / len(processor.results) if processor.results else 0
    
    print(f"Summary: {successful} successful, {failed} failed, avg time: {avg_time:.2f}s")

async def example_openai_usage():
    """Example using OpenAI generator"""
    print("\n=== OpenAI Generator Example ===")
    
    # Create sample data
    sample_file = await create_sample_data()
    
    try:
        await process_extraction_batch(
            input_file=sample_file,
            text_column='text_content',
            uid_column='id',
            pydantic_model=ExampleModel,
            generator_type='openai',
            model_name='gpt-4o-mini',
            experiment_label='openai_extraction',
            batch_size=2,
            temperature=0.1,
            system_message="Extract structured data from clinical text.",
            pre_prompt="Return JSON according to the provided schema."
        )
        print("OpenAI extraction completed!")
    except Exception as e:
        print(f"OpenAI extraction failed (likely no API key): {e}")

async def main():
    """Run all examples"""
    print("ExtraCTOps Loops - Example Usage")
    print("=" * 40)
    
    try:
        # Run examples
        await example_simple_usage()
        await example_advanced_usage() 
        await example_openai_usage()
        
        print("\n=== All Examples Completed ===")
        print("Check the 'temp' directory for output files.")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
