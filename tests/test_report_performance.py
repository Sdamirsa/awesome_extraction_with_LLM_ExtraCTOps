"""
Unit tests for the ExtraCTOps Performance Evaluation Module
"""

import unittest
import tempfile
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report_performance.evaluator import (
    ExtractionEvaluator, 
    ComparisonMethod, 
    EvaluationResult,
    FieldComparison,
    FieldMetrics
)
from report_performance.reporter import PerformanceReporter


class TestExtractionEvaluator(unittest.TestCase):
    """Test cases for ExtractionEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ExtractionEvaluator(uid_column="id")
        
        # Create sample data
        self.sample_extraction_data = pd.DataFrame({
            'id': ['doc1', 'doc2', 'doc3', 'doc4'],
            'field1': ['value1', 'value2', '', 'value4'],
            'field2': ['correct', 'wrong', 'missing', 'hallucination'],
            'field3': [10, 20, None, 40]
        })
        
        self.sample_ground_truth_data = pd.DataFrame({
            'id': ['doc1', 'doc2', 'doc3', 'doc4'],
            'field1': ['value1', 'different', 'value3', ''],
            'field2': ['correct', 'correct', '', ''],
            'field3': [10, 25, 30, None]
        })
        
        # Set up temporary files
        self.temp_dir = tempfile.mkdtemp()
        self.extraction_excel = Path(self.temp_dir) / "extraction.xlsx"
        self.ground_truth_excel = Path(self.temp_dir) / "ground_truth.xlsx"
        self.extraction_json = Path(self.temp_dir) / "extraction.json"
        self.ground_truth_json = Path(self.temp_dir) / "ground_truth.json"
        
        # Save test data
        self.sample_extraction_data.to_excel(self.extraction_excel, index=False)
        self.sample_ground_truth_data.to_excel(self.ground_truth_excel, index=False)
        
        self.sample_extraction_data.to_json(self.extraction_json, orient='records')
        self.sample_ground_truth_data.to_json(self.ground_truth_json, orient='records')
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ExtractionEvaluator()
        self.assertEqual(evaluator.uid_column, "id")
        self.assertIsNone(evaluator.extraction_data)
        self.assertIsNone(evaluator.ground_truth_data)
        
        evaluator_custom = ExtractionEvaluator(uid_column="custom_id")
        self.assertEqual(evaluator_custom.uid_column, "custom_id")
        
    def test_load_excel_data(self):
        """Test loading Excel files."""
        # Test extraction data loading
        self.evaluator.load_extraction_data(self.extraction_excel)
        self.assertIsNotNone(self.evaluator.extraction_data)
        self.assertEqual(len(self.evaluator.extraction_data), 4)
        self.assertIn('field1', self.evaluator.extraction_data.columns)
        
        # Test ground truth data loading
        self.evaluator.load_ground_truth_data(self.ground_truth_excel)
        self.assertIsNotNone(self.evaluator.ground_truth_data)
        self.assertEqual(len(self.evaluator.ground_truth_data), 4)
        
    def test_load_json_data(self):
        """Test loading JSON files."""
        # Test extraction data loading
        self.evaluator.load_extraction_data(self.extraction_json)
        self.assertIsNotNone(self.evaluator.extraction_data)
        self.assertEqual(len(self.evaluator.extraction_data), 4)
        
        # Test ground truth data loading
        self.evaluator.load_ground_truth_data(self.ground_truth_json)
        self.assertIsNotNone(self.evaluator.ground_truth_data)
        self.assertEqual(len(self.evaluator.ground_truth_data), 4)
        
    def test_load_nonexistent_file(self):
        """Test loading non-existent files raises error."""
        with self.assertRaises(FileNotFoundError):
            self.evaluator.load_extraction_data("nonexistent.xlsx")
            
        with self.assertRaises(FileNotFoundError):
            self.evaluator.load_ground_truth_data("nonexistent.json")
            
    def test_load_unsupported_format(self):
        """Test loading unsupported file format raises error."""
        unsupported_file = Path(self.temp_dir) / "test.txt"
        unsupported_file.write_text("some text")
        
        with self.assertRaises(ValueError):
            self.evaluator.load_extraction_data(unsupported_file)
            
    def test_normalize_value(self):
        """Test value normalization."""
        # Test None and NaN
        self.assertIsNone(self.evaluator._normalize_value(None))
        self.assertIsNone(self.evaluator._normalize_value(np.nan))
        self.assertIsNone(self.evaluator._normalize_value(pd.NA))
        
        # Test empty strings and null values
        self.assertIsNone(self.evaluator._normalize_value(""))
        self.assertIsNone(self.evaluator._normalize_value("  "))
        self.assertIsNone(self.evaluator._normalize_value("null"))
        self.assertIsNone(self.evaluator._normalize_value("None"))
        self.assertIsNone(self.evaluator._normalize_value("n/a"))
        self.assertIsNone(self.evaluator._normalize_value("NA"))
        
        # Test regular values
        self.assertEqual(self.evaluator._normalize_value("  value  "), "value")
        self.assertEqual(self.evaluator._normalize_value(123), 123)
        
    def test_exact_match_comparison(self):
        """Test exact match comparison logic."""
        # Test matching values
        result = self.evaluator._exact_match_comparison("value", "value")
        self.assertEqual(result, EvaluationResult.CORRECT)
        
        # Test non-matching values
        result = self.evaluator._exact_match_comparison("value1", "value2")
        self.assertEqual(result, EvaluationResult.INCORRECT)
        
        # Test both None/empty
        result = self.evaluator._exact_match_comparison(None, "")
        self.assertEqual(result, EvaluationResult.NO_INFORMATION)
        
        result = self.evaluator._exact_match_comparison("", None)
        self.assertEqual(result, EvaluationResult.NO_INFORMATION)
        
        # Test one None, one value
        result = self.evaluator._exact_match_comparison(None, "value")
        self.assertEqual(result, EvaluationResult.INCORRECT)
        
        result = self.evaluator._exact_match_comparison("value", None)
        self.assertEqual(result, EvaluationResult.INCORRECT)
        
    def test_compare_records(self):
        """Test record comparison functionality."""
        # Load data
        self.evaluator.extraction_data = self.sample_extraction_data.copy()
        self.evaluator.ground_truth_data = self.sample_ground_truth_data.copy()
        
        # Perform comparison
        self.evaluator.compare_records()
        
        # Verify comparisons were created
        self.assertGreater(len(self.evaluator.field_comparisons), 0)
        
        # Check specific comparisons
        field1_comparisons = [c for c in self.evaluator.field_comparisons if c.field_name == 'field1']
        self.assertEqual(len(field1_comparisons), 4)  # 4 records
        
        # Test specific comparison results
        doc1_field1 = next(c for c in field1_comparisons if c.uid == 'doc1')
        self.assertEqual(doc1_field1.result, EvaluationResult.CORRECT)  # 'value1' == 'value1'
        
        doc2_field1 = next(c for c in field1_comparisons if c.uid == 'doc2')
        self.assertEqual(doc2_field1.result, EvaluationResult.INCORRECT)  # 'value2' != 'different'
        
    def test_compare_records_no_data(self):
        """Test comparison fails when data not loaded."""
        with self.assertRaises(ValueError):
            self.evaluator.compare_records()
            
    def test_compare_records_unsupported_method(self):
        """Test comparison fails with unsupported method."""
        self.evaluator.extraction_data = self.sample_extraction_data.copy()
        self.evaluator.ground_truth_data = self.sample_ground_truth_data.copy()
        
        with self.assertRaises(NotImplementedError):
            self.evaluator.compare_records(ComparisonMethod.LLM_BASED)
            
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        # Load data and compare
        self.evaluator.extraction_data = self.sample_extraction_data.copy()
        self.evaluator.ground_truth_data = self.sample_ground_truth_data.copy()
        self.evaluator.compare_records()
        
        # Calculate metrics
        self.evaluator.calculate_metrics()
        
        # Verify metrics were calculated
        self.assertGreater(len(self.evaluator.field_metrics), 0)
        
        # Check field1 metrics
        field1_metrics = self.evaluator.field_metrics['field1']
        self.assertIsInstance(field1_metrics, FieldMetrics)
        self.assertEqual(field1_metrics.field_name, 'field1')
        self.assertEqual(field1_metrics.total_comparisons, 4)
        
        # Test that F1 scores are calculated
        self.assertGreaterEqual(field1_metrics.f1_score, 0)
        self.assertLessEqual(field1_metrics.f1_score, 1)
        
    def test_calculate_metrics_no_comparisons(self):
        """Test metrics calculation fails when no comparisons available."""
        with self.assertRaises(ValueError):
            self.evaluator.calculate_metrics()
            
    def test_get_overall_metrics(self):
        """Test overall metrics calculation."""
        # Set up full evaluation
        self.evaluator.extraction_data = self.sample_extraction_data.copy()
        self.evaluator.ground_truth_data = self.sample_ground_truth_data.copy()
        self.evaluator.compare_records()
        self.evaluator.calculate_metrics()
        
        # Get overall metrics
        overall = self.evaluator.get_overall_metrics()
        
        # Verify structure
        required_keys = [
            'total_comparisons', 'total_correct', 'total_incorrect', 
            'total_no_information', 'overall_accuracy_percent',
            'average_missing_rate_percent', 'average_hallucination_rate_percent',
            'average_correctness_when_extracted_percent', 'average_f1_score'
        ]
        for key in required_keys:
            self.assertIn(key, overall)
            
        # Verify values make sense
        self.assertGreaterEqual(overall['overall_accuracy_percent'], 0)
        self.assertLessEqual(overall['overall_accuracy_percent'], 100)
        self.assertGreaterEqual(overall['average_f1_score'], 0)
        self.assertLessEqual(overall['average_f1_score'], 1)
        
    def test_get_field_metrics_summary(self):
        """Test field metrics summary."""
        # Set up full evaluation
        self.evaluator.extraction_data = self.sample_extraction_data.copy()
        self.evaluator.ground_truth_data = self.sample_ground_truth_data.copy()
        self.evaluator.compare_records()
        self.evaluator.calculate_metrics()
        
        # Get field summary
        summary = self.evaluator.get_field_metrics_summary()
        
        # Verify structure
        self.assertIsInstance(summary, dict)
        self.assertIn('field1', summary)
        self.assertIn('field2', summary)
        self.assertIn('field3', summary)
        
        # Check field1 summary structure
        field1_summary = summary['field1']
        required_keys = [
            'field_name', 'total_comparisons', 'correct', 'incorrect',
            'no_information', 'missing_rate_percent', 'hallucination_rate_percent',
            'correctness_when_extracted_percent', 'f1_score'
        ]
        for key in required_keys:
            self.assertIn(key, field1_summary)
            
    def test_save_detailed_results(self):
        """Test saving detailed results to JSON."""
        # Set up full evaluation
        self.evaluator.extraction_data = self.sample_extraction_data.copy()
        self.evaluator.ground_truth_data = self.sample_ground_truth_data.copy()
        self.evaluator.compare_records()
        self.evaluator.calculate_metrics()
        
        # Save results
        results_path = Path(self.temp_dir) / "detailed_results.json"
        self.evaluator.save_detailed_results(results_path)
        
        # Verify file was created
        self.assertTrue(results_path.exists())
        
        # Load and verify content
        with open(results_path) as f:
            results = json.load(f)
            
        # Verify structure
        self.assertIn('metadata', results)
        self.assertIn('overall_metrics', results)
        self.assertIn('field_metrics', results)
        self.assertIn('detailed_comparisons', results)
        
        # Verify metadata
        metadata = results['metadata']
        self.assertEqual(metadata['total_records_compared'], 4)
        self.assertEqual(metadata['comparison_method'], 'exact_match')
        
        # Verify detailed comparisons
        comparisons = results['detailed_comparisons']
        self.assertGreater(len(comparisons), 0)
        
        # Check comparison structure
        first_comparison = comparisons[0]
        required_keys = ['field_name', 'extracted_value', 'ground_truth_value', 'result', 'uid']
        for key in required_keys:
            self.assertIn(key, first_comparison)


class TestFieldComparison(unittest.TestCase):
    """Test cases for FieldComparison dataclass."""
    
    def test_field_comparison_creation(self):
        """Test FieldComparison creation and conversion."""
        comparison = FieldComparison(
            field_name="test_field",
            extracted_value="extracted",
            ground_truth_value="ground_truth",
            result=EvaluationResult.INCORRECT,
            uid="test_uid"
        )
        
        # Test attributes
        self.assertEqual(comparison.field_name, "test_field")
        self.assertEqual(comparison.extracted_value, "extracted")
        self.assertEqual(comparison.ground_truth_value, "ground_truth")
        self.assertEqual(comparison.result, EvaluationResult.INCORRECT)
        self.assertEqual(comparison.uid, "test_uid")
        
        # Test to_dict conversion
        comparison_dict = comparison.to_dict()
        expected_keys = ['field_name', 'extracted_value', 'ground_truth_value', 'result', 'uid']
        for key in expected_keys:
            self.assertIn(key, comparison_dict)
        self.assertEqual(comparison_dict['result'], 'incorrect')


class TestFieldMetrics(unittest.TestCase):
    """Test cases for FieldMetrics dataclass."""
    
    def test_field_metrics_creation(self):
        """Test FieldMetrics creation and conversion."""
        metrics = FieldMetrics(
            field_name="test_field",
            total_comparisons=100,
            correct=80,
            incorrect=15,
            no_information=5,
            missing_rate=0.1,
            hallucination_rate=0.05,
            correctness_when_extracted=0.9,
            f1_score=0.85
        )
        
        # Test attributes
        self.assertEqual(metrics.field_name, "test_field")
        self.assertEqual(metrics.total_comparisons, 100)
        self.assertEqual(metrics.correct, 80)
        self.assertEqual(metrics.incorrect, 15)
        self.assertEqual(metrics.no_information, 5)
        
        # Test to_dict conversion
        metrics_dict = metrics.to_dict()
        expected_keys = [
            'field_name', 'total_comparisons', 'correct', 'incorrect',
            'no_information', 'missing_rate_percent', 'hallucination_rate_percent',
            'correctness_when_extracted_percent', 'f1_score'
        ]
        for key in expected_keys:
            self.assertIn(key, metrics_dict)
            
        # Test percentage conversions
        self.assertEqual(metrics_dict['missing_rate_percent'], 10.0)
        self.assertEqual(metrics_dict['hallucination_rate_percent'], 5.0)
        self.assertEqual(metrics_dict['correctness_when_extracted_percent'], 90.0)


class TestPerformanceReporter(unittest.TestCase):
    """Test cases for PerformanceReporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create evaluator with sample data
        self.evaluator = ExtractionEvaluator()
        
        sample_extraction = pd.DataFrame({
            'id': ['doc1', 'doc2', 'doc3'],
            'field1': ['value1', 'value2', 'value3'],
            'field2': ['correct', 'wrong', 'missing']
        })
        
        sample_ground_truth = pd.DataFrame({
            'id': ['doc1', 'doc2', 'doc3'],
            'field1': ['value1', 'different', 'value3'],
            'field2': ['correct', 'correct', '']
        })
        
        self.evaluator.extraction_data = sample_extraction
        self.evaluator.ground_truth_data = sample_ground_truth
        self.evaluator.compare_records()
        self.evaluator.calculate_metrics()
        
        # Create reporter
        self.reporter = PerformanceReporter(self.evaluator)
        
        # Set up temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.reporter.set_output_directory(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_reporter_initialization(self):
        """Test reporter initialization."""
        reporter = PerformanceReporter(self.evaluator)
        self.assertEqual(reporter.evaluator, self.evaluator)
        self.assertIsNone(reporter.output_dir)
        
    def test_set_output_directory(self):
        """Test setting output directory."""
        test_dir = Path(self.temp_dir) / "test_output"
        self.reporter.set_output_directory(test_dir)
        
        self.assertEqual(self.reporter.output_dir, test_dir)
        self.assertTrue(test_dir.exists())
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_performance_summary_plot(self, mock_close, mock_savefig):
        """Test performance summary plot creation."""
        save_path = self.reporter.create_performance_summary_plot()
        
        # Verify plot was "saved"
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        # Verify path
        expected_path = Path(self.temp_dir) / "performance_summary.png"
        self.assertEqual(save_path, expected_path)
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_metrics_overview_plot(self, mock_close, mock_savefig):
        """Test metrics overview plot creation."""
        save_path = self.reporter.create_metrics_overview_plot()
        
        # Verify plot was "saved"
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        # Verify path
        expected_path = Path(self.temp_dir) / "metrics_overview.png"
        self.assertEqual(save_path, expected_path)
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_confusion_matrix_plot(self, mock_close, mock_savefig):
        """Test confusion matrix plot creation."""
        save_path = self.reporter.create_confusion_matrix_plot('field1')
        
        # Verify plot was "saved"
        mock_savefig.assert_called_once()
        # Note: matplotlib may call close multiple times, so just verify it was called
        self.assertTrue(mock_close.called)
        
        # Verify path
        expected_path = Path(self.temp_dir) / "confusion_matrix_field1.png"
        self.assertEqual(save_path, expected_path)
        
    def test_create_confusion_matrix_plot_invalid_field(self):
        """Test confusion matrix plot with invalid field."""
        with self.assertRaises(ValueError):
            self.reporter.create_confusion_matrix_plot('nonexistent_field')
            
    @patch('report_performance.reporter.DOCX_AVAILABLE', False)
    def test_generate_word_report_unavailable(self):
        """Test Word report generation when docx not available."""
        result = self.reporter.generate_word_report()
        self.assertIsNone(result)
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_all_visualizations(self, mock_close, mock_savefig):
        """Test generating all visualizations."""
        visualizations = self.reporter.generate_all_visualizations()
        
        # Verify expected visualizations were created
        self.assertIn('performance_summary', visualizations)
        self.assertIn('metrics_overview', visualizations)
        
        # Should also have confusion matrices for fields
        field_confusion_keys = [k for k in visualizations.keys() if k.startswith('confusion_matrix')]
        self.assertGreater(len(field_confusion_keys), 0)
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_complete_report(self, mock_close, mock_savefig):
        """Test generating complete report."""
        report_files = self.reporter.generate_complete_report()
        
        # Verify expected files
        expected_keys = [
            'performance_summary', 'metrics_overview', 
            'detailed_results_json', 'summary_metrics_json'
        ]
        for key in expected_keys:
            self.assertIn(key, report_files)
            
        # Verify JSON files were actually created
        detailed_path = report_files['detailed_results_json']
        summary_path = report_files['summary_metrics_json']
        
        self.assertTrue(detailed_path.exists())
        self.assertTrue(summary_path.exists())
        
        # Verify JSON content
        with open(summary_path) as f:
            summary_data = json.load(f)
        self.assertIn('overall_metrics', summary_data)
        self.assertIn('field_metrics', summary_data)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
