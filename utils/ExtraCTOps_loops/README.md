# ExtraCTOps Loops - Batch Processing Module

This module provides comprehensive batch processing capabilities for extracting structured data from unstructured text using various LLM generators in the ExtraCTOps framework.

## Features

- **Async Batch Processing**: Concurrent processing with configurable batch sizes
- **Multiple Generator Support**: Works with Ollama, OpenAI, and other ExtraCTOps generators
- **Robust Error Handling**: Retry mechanisms and comprehensive error tracking
- **Progress Tracking**: Real-time progress updates and backup saves
- **Flexible Input Sources**: Support for text columns or document file paths
- **Multiple Output Formats**: Excel, CSV, and JSON output with summary statistics

## Quick Start

### Simple Usage

```python
import asyncio
from utils.ExtraCTOps_loops import process_extraction_batch
from the_pydantics.your_model import YourModel

async def extract_data():
    await process_extraction_batch(
        input_file='data.xlsx',
        text_column='text_content',
        uid_column='id',
        pydantic_model=YourModel,
        generator_type='ollama',
        model_name='llama3'
    )

asyncio.run(extract_data())
```

### Advanced Usage

```python
import asyncio
from utils.ExtraCTOps_loops import ProcessingConfig, ExtraCTOpsProcessor
from the_pydantics.your_model import YourModel

async def advanced_extraction():
    config = ProcessingConfig(
        input_file='data.xlsx',
        text_column='text_content',
        uid_column='id',
        pydantic_model=YourModel,
        generator_type='ollama',
        model_name='llama3',
        experiment_label='my_extraction',
        batch_size=10,
        backup_interval=50,
        max_retries=2,
        temperature=0.1,
        system_message="Extract data according to schema.",
        pre_prompt="Return valid JSON only."
    )
    
    processor = ExtraCTOpsProcessor(config)
    await processor.process_batch()

asyncio.run(advanced_extraction())
```

## Configuration Options

### ProcessingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_file` | str | Required | Path to input CSV/Excel/JSON file |
| `text_column` | str | None | Column name containing text to extract from |
| `text_path_column` | str | None | Column name containing paths to text files |
| `uid_column` | str | "id" | Column name for unique identifier |
| `generator_type` | str | "ollama" | Generator type ("ollama", "openai") |
| `pydantic_model` | Type[BaseModel] | Required | Pydantic model for extraction |
| `batch_size` | int | 10 | Number of concurrent async tasks |
| `backup_interval` | int | 100 | Save backup every N processed items |
| `max_retries` | int | 2 | Maximum retries for failed extractions |
| `model_name` | str | "llama3" | Name of the LLM model |
| `temperature` | float | 0.1 | Temperature for generation |
| `system_message` | str | Default | System instructions for LLM |
| `pre_prompt` | str | Default | Pre-prompt instructions |

## Input Data Format

Your input file (CSV/Excel) should contain:

| Column | Description | Required |
|--------|-------------|----------|
| ID column | Unique identifier for each row | Yes |
| Text column | Text content to extract from | Yes* |
| Text path column | Path to text files | Yes* |

*Either text_column OR text_path_column is required.

### Example Input Data

```csv
id,text_content,source
1,"Patient John Doe, age 45, diagnosed with diabetes.",clinical_note_1
2,"Mary Smith is 32 years old with hypertension.",clinical_note_2
3,"Robert Johnson, 28, shows anxiety symptoms.",clinical_note_3
```

## Output Format

The module generates:

1. **Excel file** with original data plus extracted fields
2. **JSON summary** with processing statistics
3. **Backup files** during processing (configurable interval)

### Output Columns

The output Excel file contains:
- All original columns
- `{experiment_label}_status`: Processing status ("EXTRACTED", "ERROR: ...")
- `{experiment_label}_{field_name}`: Extracted data fields
- `{experiment_label}_execution_time`: Processing time for each row
- `{experiment_label}_raw_response`: Raw LLM response (optional)

## Generator Integration

The module integrates seamlessly with ExtraCTOps generators:

### Ollama Generator
```python
config = ProcessingConfig(
    generator_type='ollama',
    model_name='llama3',
    # ... other config
)
```

### OpenAI Generator
```python
config = ProcessingConfig(
    generator_type='openai',
    model_name='gpt-4o-mini',
    # ... other config
)
```

## Error Handling

The module provides robust error handling:

- **Retry Logic**: Configurable retry attempts for failed extractions
- **Error Classification**: Different error types tracked separately
- **Partial Recovery**: Failed rows don't stop the entire batch
- **Backup Saves**: Regular backups prevent data loss
- **Detailed Logging**: Comprehensive logging for debugging

## Performance Features

- **Async Processing**: Concurrent processing for improved throughput
- **Batch Processing**: Configurable batch sizes to optimize memory usage
- **Progress Tracking**: Real-time progress updates
- **Resource Management**: Proper cleanup and resource management
- **Interruption Handling**: Graceful handling of user interruptions

## Examples

### Example 1: Clinical Text Extraction

```python
from utils.ExtraCTOps_loops import process_extraction_batch
from the_pydantics.clinical_schema import PatientInfo

await process_extraction_batch(
    input_file='clinical_notes.xlsx',
    text_column='note_text',
    uid_column='patient_id',
    pydantic_model=PatientInfo,
    generator_type='ollama',
    model_name='llama3',
    experiment_label='patient_extraction',
    system_message="Extract patient information from clinical notes.",
    temperature=0.05  # Low temperature for accuracy
)
```

### Example 2: Document Processing from Files

```python
config = ProcessingConfig(
    input_file='document_index.csv',
    text_path_column='document_path',  # Column with file paths
    uid_column='doc_id',
    pydantic_model=DocumentSchema,
    generator_type='openai',
    model_name='gpt-4',
    batch_size=5,  # Smaller batches for large documents
    max_retries=3
)

processor = ExtraCTOpsProcessor(config)
await processor.process_batch()
```

## Integration with ExtraCTOps Ecosystem

This module integrates with:

- **Generators**: Uses configured generators from `generators/` package
- **Config System**: Leverages centralized configuration from `config/`
- **Pydantic Models**: Works with any Pydantic models from `the_pydantics/`
- **Logging**: Uses ExtraCTOps logging system from `utils/`
- **Document Handlers**: Integrates with document reading utilities

## Running Examples

Run the included examples:

```bash
# From project root
python utils/ExtraCTOps_loops/example_usage.py
```

This will:
1. Create sample data
2. Run simple extraction example
3. Run advanced configuration example
4. Demonstrate OpenAI generator usage

## Dependencies

The module requires:
- pandas (for data handling)
- pydantic (for data models)
- asyncio (for async processing)
- openpyxl (for Excel support)
- ExtraCTOps generators and config system

## Best Practices

1. **Start Small**: Test with small batches first
2. **Monitor Progress**: Check logs and backup files during processing
3. **Optimize Batch Size**: Balance between speed and memory usage
4. **Use Appropriate Models**: Match model complexity to task requirements
5. **Handle Interruptions**: Always save progress and handle KeyboardInterrupt
6. **Validate Results**: Review extraction quality before scaling up

## Troubleshooting

### Common Issues

1. **Generator Not Found**: Ensure generator type is supported and configured
2. **Model Errors**: Check model name and availability
3. **Memory Issues**: Reduce batch size for large documents
4. **API Limits**: Consider rate limiting for API-based generators
5. **File Format**: Ensure input file format is supported

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger('utils.ExtraCTOps_loops').setLevel(logging.DEBUG)
```

## Contributing

To extend the module:
1. Add new generator types in the `extract_from_row` method
2. Extend `ProcessingConfig` for new configuration options
3. Add new input/output formats as needed
4. Update tests and documentation

For questions or contributions, contact: sdamirsa@gmail.com
