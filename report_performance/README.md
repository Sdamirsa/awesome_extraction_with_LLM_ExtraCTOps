# ExtraCTOps Performance Evaluation Module

This module provides comprehensive tools for evaluating the performance of extraction systems against ground truth data. It supports multiple comparison methods and generates detailed reports with visualizations.

## Features

### Core Evaluation Capabilities
- **Exact Match Comparison**: Direct value-to-value comparison with normalization
- **3x3 Confusion Matrix**: Correct, Incorrect, No Information classification
- **Advanced Metrics**: F1 score, missing rate, hallucination rate, correctness when extracted
- **Field-level Analysis**: Individual metrics for each extracted field
- **Overall Performance**: Aggregated metrics across all fields

### Extensible Architecture
- **LLM-based Evaluation**: Framework ready for semantic comparison (future)
- **Embedding-based Evaluation**: Framework ready for similarity-based comparison (future)
- **Custom Comparison Methods**: Easy to add new evaluation approaches

### Comprehensive Reporting
- **Interactive Visualizations**: Matplotlib/Seaborn charts and plots
- **Word Document Reports**: Professional formatted reports (if python-docx available)
- **JSON Exports**: Detailed and summary metrics in structured format
- **Confusion Matrix Plots**: Per-field and overall visualization

## Quick Start

```python
from report_performance.evaluator import ExtractionEvaluator
from report_performance.reporter import PerformanceReporter

# Initialize evaluator
evaluator = ExtractionEvaluator(uid_column="id")

# Load data
evaluator.load_extraction_data("path/to/extraction_results.xlsx")
evaluator.load_ground_truth_data("path/to/ground_truth.xlsx")

# Perform comparison
evaluator.compare_records()
evaluator.calculate_metrics()

# Generate reports
reporter = PerformanceReporter(evaluator)
reporter.set_output_directory("evaluation_results")
report_files = reporter.generate_complete_report()
```

## Detailed Usage

### 1. Data Loading
The evaluator supports Excel (.xlsx, .xls) and JSON files:

```python
# Excel files
evaluator.load_extraction_data("results.xlsx")
evaluator.load_ground_truth_data("ground_truth.xlsx")

# JSON files
evaluator.load_extraction_data("results.json")
evaluator.load_ground_truth_data("ground_truth.json")
```

### 2. Data Requirements
- Both datasets must have a common UID column for record matching
- Field names should match between extraction and ground truth
- Values are automatically normalized (whitespace, null handling)

### 3. Comparison Methods
Currently supports exact match with framework for future methods:

```python
from report_performance.evaluator import ComparisonMethod

# Exact match (default)
evaluator.compare_records(ComparisonMethod.EXACT_MATCH)

# Future methods (not yet implemented)
# evaluator.compare_records(ComparisonMethod.LLM_BASED)
# evaluator.compare_records(ComparisonMethod.EMBEDDING_BASED)
```

### 4. Metrics Calculation
The system calculates comprehensive metrics:

```python
evaluator.calculate_metrics()

# Get overall metrics
overall = evaluator.get_overall_metrics()
print(f"Overall Accuracy: {overall['overall_accuracy_percent']}%")
print(f"Average F1 Score: {overall['average_f1_score']}")

# Get field-specific metrics
field_metrics = evaluator.get_field_metrics_summary()
for field_name, metrics in field_metrics.items():
    print(f"{field_name}: F1={metrics['f1_score']}, Missing={metrics['missing_rate_percent']}%")
```

### 5. Report Generation
Generate comprehensive reports with visualizations:

```python
reporter = PerformanceReporter(evaluator)
reporter.set_output_directory("my_evaluation_results")

# Generate specific visualizations
summary_plot = reporter.create_performance_summary_plot()
overview_plot = reporter.create_metrics_overview_plot()
confusion_plot = reporter.create_confusion_matrix_plot("field_name")

# Generate Word report (if python-docx available)
word_report = reporter.generate_word_report()

# Generate complete report with all components
all_files = reporter.generate_complete_report()
```

## Metrics Explained

### Core Metrics
- **Correct**: Extracted value exactly matches ground truth
- **Incorrect**: Extracted value differs from ground truth
- **No Information**: Both extraction and ground truth have no value

### Specialized Rates
- **Missing Rate**: Percentage where ground truth has value but extraction doesn't
- **Hallucination Rate**: Percentage where extraction provides value but ground truth doesn't
- **Correctness When Extracted**: Accuracy among records where extraction provided a value

### Performance Scores
- **F1 Score**: Harmonic mean of precision and recall
- **Overall Accuracy**: Percentage of correct extractions across all fields

## Output Files

A complete report generates:
- `performance_summary.png`: Overall performance visualization
- `metrics_overview.png`: Field-by-field metrics comparison
- `confusion_matrix_*.png`: Individual confusion matrices for top fields
- `detailed_results.json`: Complete comparison data and metrics
- `summary_metrics.json`: High-level metrics summary
- `extraction_performance_report_*.docx`: Professional Word report (if available)

## Data Format Examples

### Excel/CSV Format
```
id,field1,field2,field3
doc001,value1,value2,value3
doc002,value4,,value6
```

### JSON Format
```json
[
  {"id": "doc001", "field1": "value1", "field2": "value2", "field3": "value3"},
  {"id": "doc002", "field1": "value4", "field2": null, "field3": "value6"}
]
```

## Dependencies

### Required
- pandas
- numpy
- matplotlib
- seaborn

### Optional
- python-docx (for Word report generation)

## Future Enhancements

### Planned Features
1. **LLM-based Evaluation**: Semantic comparison using language models
2. **Embedding-based Evaluation**: Similarity-based comparison using embeddings
3. **Custom Similarity Thresholds**: Configurable matching criteria
4. **Advanced Visualizations**: Interactive plots and dashboards
5. **Batch Evaluation**: Process multiple evaluation sets
6. **Statistical Significance Testing**: Confidence intervals and hypothesis testing

### Architecture Extension Points
- `ComparisonMethod` enum: Add new comparison methods
- `EvaluationResult` enum: Extend result categories if needed
- Custom metrics: Add domain-specific evaluation criteria
- Report templates: Customize Word document formatting

## Integration with ExtraCTOps

This module integrates seamlessly with the ExtraCTOps extraction pipeline:

```python
# After running extraction with ExtraCTOps_loops
from utils.ExtraCTOps_loops import batch_process_documents
from report_performance.evaluator import ExtractionEvaluator

# Run extraction
results = batch_process_documents(...)

# Evaluate results
evaluator = ExtractionEvaluator()
evaluator.load_extraction_data(results['excel_path'])
evaluator.load_ground_truth_data("manual_ground_truth.xlsx")
evaluator.compare_records()
evaluator.calculate_metrics()

# Generate report
reporter = PerformanceReporter(evaluator)
reporter.generate_complete_report()
```

## Error Handling

The module includes robust error handling:
- File existence validation
- Data format verification
- Common field validation
- UID matching verification
- Graceful degradation for missing dependencies

## Performance Considerations

- Memory-efficient processing for large datasets
- Vectorized operations using pandas/numpy
- Lazy loading of visualization libraries
- Configurable batch sizes for very large evaluations (future)

This module provides a solid foundation for rigorous evaluation of extraction systems and can be extended to support more sophisticated comparison methods as needed.
