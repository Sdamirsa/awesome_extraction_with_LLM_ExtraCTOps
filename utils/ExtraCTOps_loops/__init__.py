"""
ExtraCTOps Loops Module

Batch processing utilities for extracting structured data from unstructured text
using various LLM generators.

Main Components:
- ExtraCTOpsProcessor: Main class for batch processing
- ProcessingConfig: Configuration for batch operations
- ExtractionResult: Result structure for individual extractions
- process_extraction_batch: Convenience function for simple usage

Usage:
    from utils.ExtraCTOps_loops import process_extraction_batch, ExtraCTOpsProcessor
    
    # Simple usage
    await process_extraction_batch(
        input_file='data.xlsx',
        text_column='text_content',
        uid_column='id',
        pydantic_model=MyModel,
        generator_type='ollama'
    )
    
    # Advanced usage
    config = ProcessingConfig(...)
    processor = ExtraCTOpsProcessor(config)
    await processor.process_batch()
"""

from .main import (
    ExtraCTOpsProcessor,
    ProcessingConfig,
    ExtractionResult,
    process_extraction_batch
)

__all__ = [
    'ExtraCTOpsProcessor',
    'ProcessingConfig', 
    'ExtractionResult',
    'process_extraction_batch'
]
