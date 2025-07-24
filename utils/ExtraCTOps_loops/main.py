"""
ExtraCTOps Loops - Batch Processing Module

This module provides batch processing capabilities for extracting structured data
from unstructured text using various LLM generators. It reads input data from
CSV/Excel/JSON files, processes them through generators, and saves results.

Key Features:
- Async batch processing with configurable concurrency
- Support for multiple generator types (Ollama, OpenAI, etc.)
- Robust error handling and retry mechanisms
- Progress tracking and backup saves
- Integration with ExtraCTOps config system
- Support for text columns or document file paths

Usage:
    from utils.ExtraCTOps_loops.main import ExtraCTOpsProcessor
    
    processor = ExtraCTOpsProcessor(
        input_file='data.xlsx',
        text_column='text_content',
        uid_column='id',
        generator_type='ollama',
        pydantic_model=MyModel
    )
    
    await processor.process_batch()
"""

import asyncio
import platform
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Type, Literal
from datetime import datetime
import json
import pandas as pd
from pydantic import BaseModel, Field
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.document_handler import (
    read_csv_excel, save_dataframe, flatten_pydantic_object, 
    read_text_file, DocumentReadError
)
from internal_models.generators_models import MessageConfig, ModelConfig, GenerationResult
from generators.generator_ollama import generator_Ollama
from generators.generator_openai import generator_OpenAI
from utils.the_logger import the_logger

logger = the_logger

class ProcessingConfig(BaseModel):
    """Configuration for batch processing"""
    input_file: str = Field(description="Path to input CSV/Excel/JSON file")
    output_file: Optional[str] = Field(default=None, description="Path to output file (auto-generated if None)")
    experiment_label: str = Field(default="extraction", description="Label for this extraction experiment")
    
    # Data source configuration
    text_column: Optional[str] = Field(default=None, description="Column name containing text to extract from")
    text_path_column: Optional[str] = Field(default=None, description="Column name containing paths to text files")
    uid_column: str = Field(default="id", description="Column name for unique identifier")
    
    # Generator configuration
    generator_type: Literal["ollama", "openai"] = Field(default="ollama", description="Type of generator to use")
    pydantic_model: Union[str, Type[BaseModel]] = Field(description="Pydantic model for extraction")
    
    # Processing configuration
    batch_size: int = Field(default=10, description="Number of concurrent async tasks")
    backup_interval: int = Field(default=100, description="Save backup every N processed items")
    max_retries: int = Field(default=2, description="Maximum retries for failed extractions")
    
    # LLM configuration
    model_name: str = Field(default="llama3", description="Name of the LLM model")
    system_message: str = Field(default="Extract structured data from the text according to the provided schema.", description="System message for LLM")
    pre_prompt: str = Field(default="Return the extracted information as valid JSON.", description="Pre-prompt instructions")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=2048, description="Maximum tokens to generate")
    
    # Optional advanced features
    few_shot_json_path: Optional[str] = Field(default=None, description="Path to few-shot examples JSON")
    image_paths: Optional[List[str]] = Field(default=None, description="Paths to images for multimodal input")
    
    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after model creation"""
        if not self.text_column and not self.text_path_column:
            raise ValueError("Either text_column or text_path_column must be specified")
        if self.text_column and self.text_path_column:
            raise ValueError("Cannot specify both text_column and text_path_column")

class ExtractionResult(BaseModel):
    """Result of processing a single row"""
    uid: Any = Field(description="Unique identifier from the row")
    success: bool = Field(description="Whether extraction was successful")
    execution_time: float = Field(description="Time taken for extraction")
    extracted_data: Optional[Dict[str, Any]] = Field(default=None, description="Flattened extracted data")
    raw_response: Optional[str] = Field(default=None, description="Raw LLM response")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retries attempted")

class ExtraCTOpsProcessor:
    """Main processor for batch extraction using ExtraCTOps generators"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.results: List[ExtractionResult] = []
        self.processed_count = 0
        
        # Set up output file path
        if not self.config.output_file:
            input_path = Path(self.config.input_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.output_file = str(input_path.parent / f"{input_path.stem}_{self.config.experiment_label}_{timestamp}.xlsx")
        
        logger.info(f"Initialized ExtraCTOpsProcessor with config: {self.config.experiment_label}")
    
    async def load_data(self) -> None:
        """Load input data from file"""
        try:
            self.df = read_csv_excel(self.config.input_file)
            logger.info(f"Loaded {len(self.df)} rows from {self.config.input_file}")
            
            # Validate required columns exist
            if self.config.text_column and self.config.text_column not in self.df.columns:
                raise ValueError(f"Text column '{self.config.text_column}' not found in data")
            if self.config.text_path_column and self.config.text_path_column not in self.df.columns:
                raise ValueError(f"Text path column '{self.config.text_path_column}' not found in data")
            if self.config.uid_column not in self.df.columns:
                raise ValueError(f"UID column '{self.config.uid_column}' not found in data")
                
            # Add status column if it doesn't exist
            status_col = f"{self.config.experiment_label}_status"
            if status_col not in self.df.columns:
                self.df[status_col] = ""
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_text_content(self, row: pd.Series) -> str:
        """Extract text content from a row based on configuration"""
        if self.config.text_column:
            return str(row[self.config.text_column])
        elif self.config.text_path_column:
            file_path = row[self.config.text_path_column]
            try:
                return read_text_file(file_path)
            except DocumentReadError as e:
                raise ValueError(f"Could not read text file {file_path}: {e}")
        else:
            raise ValueError("No text source configured")
    
    async def extract_from_row(self, index: int, row: pd.Series) -> ExtractionResult:
        """Extract data from a single row using the configured generator"""
        uid = row[self.config.uid_column]
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get text content
            text_content = self.get_text_content(row)
            
            # Create message config
            message_config = MessageConfig(
                system_message=self.config.system_message,
                pre_prompt=self.config.pre_prompt,
                few_shot_json_path=self.config.few_shot_json_path,
                image_paths=self.config.image_paths,
                text=text_content,
                pydantic_model=self.config.pydantic_model
            )
            
            # Create model config
            model_config = ModelConfig(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Call appropriate generator
            if self.config.generator_type == "ollama":
                result = await generator_Ollama(message_config, model_config)
            elif self.config.generator_type == "openai":
                result = await generator_OpenAI(message_config, model_config)
            else:
                raise ValueError(f"Unsupported generator type: {self.config.generator_type}")
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Process result
            if result.parsing_success and result.parsed_response:
                flattened_data = flatten_pydantic_object(result.parsed_response)
                return ExtractionResult(
                    uid=uid,
                    success=True,
                    execution_time=execution_time,
                    extracted_data=flattened_data,
                    raw_response=result.raw_response
                )
            else:
                return ExtractionResult(
                    uid=uid,
                    success=False,
                    execution_time=execution_time,
                    raw_response=result.raw_response,
                    error=result.error or "Parsing failed"
                )
                
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error processing row {index} (UID: {uid}): {e}")
            return ExtractionResult(
                uid=uid,
                success=False,
                execution_time=execution_time,
                error=str(e)
            )
    
    async def process_row_with_retry(self, index: int, row: pd.Series) -> ExtractionResult:
        """Process a row with retry logic"""
        for attempt in range(self.config.max_retries + 1):
            result = await self.extract_from_row(index, row)
            result.retry_count = attempt
            
            if result.success:
                return result
            
            if attempt < self.config.max_retries:
                logger.warning(f"Retrying row {index} (attempt {attempt + 1}/{self.config.max_retries + 1})")
                await asyncio.sleep(1)  # Brief delay before retry
        
        return result
    
    def update_dataframe_with_results(self, results: List[ExtractionResult]) -> None:
        """Update the DataFrame with extraction results"""
        status_col = f"{self.config.experiment_label}_status"
        
        for result in results:
            # Find the row with matching UID
            mask = self.df[self.config.uid_column] == result.uid
            row_indices = self.df.index[mask].tolist()
            
            if not row_indices:
                logger.warning(f"Could not find row for UID: {result.uid}")
                continue
                
            row_index = row_indices[0]
            
            if result.success and result.extracted_data:
                # Add extracted data as new columns
                for field_name, field_value in result.extracted_data.items():
                    column_name = f"{self.config.experiment_label}_{field_name}"
                    if column_name not in self.df.columns:
                        self.df[column_name] = None
                    self.df.at[row_index, column_name] = str(field_value)
                
                # Mark as successfully extracted
                self.df.at[row_index, status_col] = "EXTRACTED"
            else:
                # Mark as failed with error
                self.df.at[row_index, status_col] = f"ERROR: {result.error}"
            
            # Add metadata columns
            time_col = f"{self.config.experiment_label}_execution_time"
            if time_col not in self.df.columns:
                self.df[time_col] = None
            self.df.at[row_index, time_col] = result.execution_time
            
            if result.raw_response:
                raw_col = f"{self.config.experiment_label}_raw_response"
                if raw_col not in self.df.columns:
                    self.df[raw_col] = None
                self.df.at[row_index, raw_col] = result.raw_response
    
    def save_backup(self, suffix: str = "") -> None:
        """Save backup of current data"""
        backup_path = self.config.output_file.replace(".xlsx", f"_backup{suffix}.xlsx")
        try:
            save_dataframe(self.df, backup_path, format='excel')
            logger.info(f"Backup saved to {backup_path}")
        except Exception as e:
            logger.error(f"Error saving backup: {e}")
    
    def save_results(self) -> None:
        """Save final results to output files"""
        try:
            # Save Excel
            save_dataframe(self.df, self.config.output_file, format='excel')
            
            # Save JSON summary
            json_path = self.config.output_file.replace(".xlsx", "_summary.json")
            summary = {
                "experiment_label": self.config.experiment_label,
                "total_rows": len(self.df),
                "processed_rows": self.processed_count,
                "successful_extractions": len([r for r in self.results if r.success]),
                "failed_extractions": len([r for r in self.results if not r.success]),
                "average_execution_time": sum(r.execution_time for r in self.results) / len(self.results) if self.results else 0,
                "config": self.config.model_dump()
            }
            
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Results saved to {self.config.output_file}")
            logger.info(f"Summary saved to {json_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def open_output_file(self) -> None:
        """Open the output file in the default application"""
        try:
            if platform.system() == 'Windows':
                os.startfile(self.config.output_file)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', self.config.output_file])
            elif platform.system() == 'Linux':
                subprocess.call(['xdg-open', self.config.output_file])
        except Exception as e:
            logger.warning(f"Could not open output file: {e}")
    
    async def process_batch(self, open_file: bool = True) -> None:
        """Main method to process the entire batch"""
        try:
            # Load data
            await self.load_data()
            
            status_col = f"{self.config.experiment_label}_status"
            
            # Get rows that need processing
            unprocessed_mask = (self.df[status_col] != "EXTRACTED") & (~self.df[status_col].str.startswith("ERROR"))
            unprocessed_indices = self.df.index[unprocessed_mask].tolist()
            
            if not unprocessed_indices:
                logger.info("All rows already processed")
                return
            
            logger.info(f"Processing {len(unprocessed_indices)} unprocessed rows")
            
            # Process in batches
            for i in range(0, len(unprocessed_indices), self.config.batch_size):
                batch_indices = unprocessed_indices[i:i + self.config.batch_size]
                batch_tasks = []
                
                # Create async tasks for batch
                for idx in batch_indices:
                    row = self.df.loc[idx]
                    task = asyncio.create_task(self.process_row_with_retry(idx, row))
                    batch_tasks.append(task)
                
                # Execute batch
                try:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process results
                    valid_results = []
                    for result in batch_results:
                        if isinstance(result, Exception):
                            logger.error(f"Batch task failed: {result}")
                        else:
                            valid_results.append(result)
                            self.results.append(result)
                    
                    # Update DataFrame
                    if valid_results:
                        self.update_dataframe_with_results(valid_results)
                        self.processed_count += len(valid_results)
                    
                    logger.info(f"Completed batch {i//self.config.batch_size + 1}, "
                              f"processed {self.processed_count}/{len(unprocessed_indices)}")
                    
                    # Save backup periodically
                    if self.processed_count % self.config.backup_interval == 0:
                        self.save_backup(f"_{self.processed_count}")
                
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    continue
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
        finally:
            # Always save results
            if self.df is not None:
                self.save_results()
                if open_file:
                    self.open_output_file()

# Convenience function for simple usage
async def process_extraction_batch(
    input_file: str,
    text_column: str,
    uid_column: str,
    pydantic_model: Union[str, Type[BaseModel]],
    generator_type: Literal["ollama", "openai"] = "ollama",
    model_name: str = "llama3",
    experiment_label: str = "extraction",
    batch_size: int = 10,
    temperature: float = 0.1,
    **kwargs
) -> None:
    """
    Convenience function for simple batch extraction
    
    Args:
        input_file: Path to input CSV/Excel file
        text_column: Column name containing text to extract from
        uid_column: Column name for unique identifier
        pydantic_model: Pydantic model for extraction
        generator_type: Type of generator ("ollama" or "openai")
        model_name: Name of the LLM model
        experiment_label: Label for this experiment
        batch_size: Number of concurrent tasks
        temperature: LLM temperature
        **kwargs: Additional configuration options
    """
    config = ProcessingConfig(
        input_file=input_file,
        text_column=text_column,
        uid_column=uid_column,
        pydantic_model=pydantic_model,
        generator_type=generator_type,
        model_name=model_name,
        experiment_label=experiment_label,
        batch_size=batch_size,
        temperature=temperature,
        **kwargs
    )
    
    processor = ExtraCTOpsProcessor(config)
    await processor.process_batch()


if __name__ == "__main__":
    # Example usage
    from the_pydantics.example_schema import ExampleModel
    
    async def main():
        # Example configuration
        config = ProcessingConfig(
            input_file="data/sample_data.xlsx",
            text_column="text_content", 
            uid_column="id",
            pydantic_model=ExampleModel,
            generator_type="ollama",
            model_name="llama3",
            experiment_label="test_extraction",
            batch_size=5,
            temperature=0.1
        )
        
        processor = ExtraCTOpsProcessor(config)
        await processor.process_batch()
    
    # Run example
    # asyncio.run(main())
    print("ExtraCTOps Loops module loaded. Use process_extraction_batch() for simple usage.")
