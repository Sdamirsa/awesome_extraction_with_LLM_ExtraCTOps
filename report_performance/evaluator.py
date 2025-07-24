"""
Performance Evaluation Core Module for ExtraCTOps

This module provides the core functionality for evaluating extraction performance
against ground truth data using various comparison methods.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class ComparisonMethod(Enum):
    """Supported comparison methods for evaluation."""
    EXACT_MATCH = "exact_match"
    LLM_BASED = "llm_based"  # Future implementation
    EMBEDDING_BASED = "embedding_based"  # Future implementation


class EvaluationResult(Enum):
    """Possible evaluation results for each field comparison."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    NO_INFORMATION = "no_information"


@dataclass
class FieldComparison:
    """Represents the comparison result for a single field."""
    field_name: str
    extracted_value: Any
    ground_truth_value: Any
    result: EvaluationResult
    uid: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "field_name": self.field_name,
            "extracted_value": self.extracted_value,
            "ground_truth_value": self.ground_truth_value,
            "result": self.result.value,
            "uid": self.uid
        }


@dataclass
class FieldMetrics:
    """Metrics for a single field across all records."""
    field_name: str
    total_comparisons: int
    correct: int
    incorrect: int
    no_information: int
    missing_rate: float  # GT has value, extraction doesn't
    hallucination_rate: float  # GT has no value, extraction does
    correctness_when_extracted: float  # Correct when extraction has value
    f1_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "field_name": self.field_name,
            "total_comparisons": self.total_comparisons,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "no_information": self.no_information,
            "missing_rate_percent": round(self.missing_rate * 100, 2),
            "hallucination_rate_percent": round(self.hallucination_rate * 100, 2),
            "correctness_when_extracted_percent": round(self.correctness_when_extracted * 100, 2),
            "f1_score": round(self.f1_score, 4)
        }


class ExtractionEvaluator:
    """
    Main evaluation class for comparing extraction results against ground truth.
    
    This class handles loading data, performing comparisons, calculating metrics,
    and generating reports.
    """
    
    def __init__(self, uid_column: str = "id"):
        """
        Initialize the evaluator.
        
        Args:
            uid_column: Column name containing unique identifiers for matching records
        """
        self.uid_column = uid_column
        self.extraction_data: Optional[pd.DataFrame] = None
        self.ground_truth_data: Optional[pd.DataFrame] = None
        self.field_comparisons: List[FieldComparison] = []
        self.field_metrics: Dict[str, FieldMetrics] = {}
        
    def load_extraction_data(self, file_path: Union[str, Path]) -> None:
        """
        Load extraction results from Excel or JSON file.
        
        Args:
            file_path: Path to the extraction results file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Extraction file not found: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.extraction_data = pd.DataFrame(data)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            self.extraction_data = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        print(f"✅ Loaded extraction data: {self.extraction_data.shape}")
        print(f"   Columns: {len(self.extraction_data.columns)}")
        
    def load_ground_truth_data(self, file_path: Union[str, Path]) -> None:
        """
        Load ground truth data from Excel or JSON file.
        
        Args:
            file_path: Path to the ground truth file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.ground_truth_data = pd.DataFrame(data)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            self.ground_truth_data = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        print(f"✅ Loaded ground truth data: {self.ground_truth_data.shape}")
        print(f"   Columns: {len(self.ground_truth_data.columns)}")
        
    def _normalize_value(self, value: Any) -> Any:
        """
        Normalize values for comparison.
        
        Args:
            value: Raw value to normalize
            
        Returns:
            Normalized value
        """
        if pd.isna(value) or value is None:
            return None
        
        if isinstance(value, str):
            # Strip whitespace and convert empty strings to None
            value = value.strip()
            if value == "" or value.lower() in ["null", "none", "n/a", "na"]:
                return None
        
        return value
        
    def _exact_match_comparison(self, extracted_value: Any, ground_truth_value: Any) -> EvaluationResult:
        """
        Perform exact match comparison between two values.
        
        Args:
            extracted_value: Value from extraction results
            ground_truth_value: Value from ground truth
            
        Returns:
            Comparison result
        """
        # Normalize values
        extracted_norm = self._normalize_value(extracted_value)
        ground_truth_norm = self._normalize_value(ground_truth_value)
        
        # Both values are None/empty
        if extracted_norm is None and ground_truth_norm is None:
            return EvaluationResult.NO_INFORMATION
        
        # Values match exactly
        if extracted_norm == ground_truth_norm:
            return EvaluationResult.CORRECT
        
        # Values don't match
        return EvaluationResult.INCORRECT
        
    def compare_records(self, method: ComparisonMethod = ComparisonMethod.EXACT_MATCH) -> None:
        """
        Compare extraction results with ground truth data.
        
        Args:
            method: Comparison method to use
        """
        if self.extraction_data is None:
            raise ValueError("Extraction data not loaded")
        if self.ground_truth_data is None:
            raise ValueError("Ground truth data not loaded")
        
        if method != ComparisonMethod.EXACT_MATCH:
            raise NotImplementedError(f"Method {method.value} not yet implemented")
        
        self.field_comparisons = []
        
        # Get common UIDs
        extraction_uids = set(self.extraction_data[self.uid_column].astype(str))
        ground_truth_uids = set(self.ground_truth_data[self.uid_column].astype(str))
        common_uids = extraction_uids.intersection(ground_truth_uids)
        
        print(f"📊 Comparing records:")
        print(f"   Extraction UIDs: {len(extraction_uids)}")
        print(f"   Ground truth UIDs: {len(ground_truth_uids)}")
        print(f"   Common UIDs: {len(common_uids)}")
        
        if not common_uids:
            raise ValueError("No common UIDs found between extraction and ground truth data")
        
        # Get common fields (excluding UID column)
        extraction_fields = set(self.extraction_data.columns) - {self.uid_column}
        ground_truth_fields = set(self.ground_truth_data.columns) - {self.uid_column}
        common_fields = extraction_fields.intersection(ground_truth_fields)
        
        print(f"   Common fields: {len(common_fields)}")
        
        if not common_fields:
            print("⚠️  Warning: No common fields found for comparison")
            return
        
        # Perform comparisons
        for uid in common_uids:
            # Get records for this UID
            extraction_record = self.extraction_data[
                self.extraction_data[self.uid_column].astype(str) == uid
            ].iloc[0]
            
            ground_truth_record = self.ground_truth_data[
                self.ground_truth_data[self.uid_column].astype(str) == uid
            ].iloc[0]
            
            # Compare each field
            for field in common_fields:
                extracted_value = extraction_record[field]
                ground_truth_value = ground_truth_record[field]
                
                if method == ComparisonMethod.EXACT_MATCH:
                    result = self._exact_match_comparison(extracted_value, ground_truth_value)
                
                comparison = FieldComparison(
                    field_name=field,
                    extracted_value=extracted_value,
                    ground_truth_value=ground_truth_value,
                    result=result,
                    uid=str(uid)
                )
                
                self.field_comparisons.append(comparison)
        
        print(f"✅ Completed {len(self.field_comparisons)} field comparisons")
        
    def calculate_metrics(self) -> None:
        """Calculate performance metrics for each field."""
        if not self.field_comparisons:
            raise ValueError("No comparisons available. Run compare_records() first.")
        
        # Group comparisons by field
        field_groups = {}
        for comparison in self.field_comparisons:
            field_name = comparison.field_name
            if field_name not in field_groups:
                field_groups[field_name] = []
            field_groups[field_name].append(comparison)
        
        self.field_metrics = {}
        
        for field_name, comparisons in field_groups.items():
            # Count results
            correct = sum(1 for c in comparisons if c.result == EvaluationResult.CORRECT)
            incorrect = sum(1 for c in comparisons if c.result == EvaluationResult.INCORRECT)
            no_info = sum(1 for c in comparisons if c.result == EvaluationResult.NO_INFORMATION)
            total = len(comparisons)
            
            # Calculate specific rates
            missing_count = 0  # GT has value, extraction doesn't
            hallucination_count = 0  # GT has no value, extraction does
            extracted_with_value_count = 0  # Extraction has a value
            correct_when_extracted = 0  # Correct among those with extracted values
            
            for comparison in comparisons:
                gt_has_value = self._normalize_value(comparison.ground_truth_value) is not None
                extracted_has_value = self._normalize_value(comparison.extracted_value) is not None
                
                # Missing: GT has value but extraction doesn't
                if gt_has_value and not extracted_has_value:
                    missing_count += 1
                
                # Hallucination: GT has no value but extraction does
                if not gt_has_value and extracted_has_value:
                    hallucination_count += 1
                
                # Count extractions with values
                if extracted_has_value:
                    extracted_with_value_count += 1
                    if comparison.result == EvaluationResult.CORRECT:
                        correct_when_extracted += 1
            
            # Calculate rates
            gt_with_value_count = sum(1 for c in comparisons 
                                    if self._normalize_value(c.ground_truth_value) is not None)
            
            missing_rate = missing_count / gt_with_value_count if gt_with_value_count > 0 else 0
            hallucination_rate = hallucination_count / total if total > 0 else 0
            correctness_when_extracted = (correct_when_extracted / extracted_with_value_count 
                                        if extracted_with_value_count > 0 else 0)
            
            # Calculate F1 score from 3x3 confusion matrix
            # For F1, we treat "correct" as positive class
            precision = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
            recall = correct / (correct + incorrect + missing_count) if (correct + incorrect + missing_count) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = FieldMetrics(
                field_name=field_name,
                total_comparisons=total,
                correct=correct,
                incorrect=incorrect,
                no_information=no_info,
                missing_rate=missing_rate,
                hallucination_rate=hallucination_rate,
                correctness_when_extracted=correctness_when_extracted,
                f1_score=f1_score
            )
            
            self.field_metrics[field_name] = metrics
        
        print(f"✅ Calculated metrics for {len(self.field_metrics)} fields")
        
    def get_overall_metrics(self) -> Dict[str, float]:
        """
        Calculate overall metrics across all fields.
        
        Returns:
            Dictionary containing overall performance metrics
        """
        if not self.field_metrics:
            raise ValueError("No metrics available. Run calculate_metrics() first.")
        
        # Aggregate counts across all fields
        total_correct = sum(m.correct for m in self.field_metrics.values())
        total_incorrect = sum(m.incorrect for m in self.field_metrics.values())
        total_no_info = sum(m.no_information for m in self.field_metrics.values())
        total_comparisons = sum(m.total_comparisons for m in self.field_metrics.values())
        
        # Calculate averages
        avg_missing_rate = np.mean([m.missing_rate for m in self.field_metrics.values()])
        avg_hallucination_rate = np.mean([m.hallucination_rate for m in self.field_metrics.values()])
        avg_correctness_when_extracted = np.mean([m.correctness_when_extracted for m in self.field_metrics.values()])
        avg_f1_score = np.mean([m.f1_score for m in self.field_metrics.values()])
        
        # Overall accuracy
        overall_accuracy = total_correct / total_comparisons if total_comparisons > 0 else 0
        
        return {
            "total_comparisons": total_comparisons,
            "total_correct": total_correct,
            "total_incorrect": total_incorrect,
            "total_no_information": total_no_info,
            "overall_accuracy_percent": round(overall_accuracy * 100, 2),
            "average_missing_rate_percent": round(avg_missing_rate * 100, 2),
            "average_hallucination_rate_percent": round(avg_hallucination_rate * 100, 2),
            "average_correctness_when_extracted_percent": round(avg_correctness_when_extracted * 100, 2),
            "average_f1_score": round(avg_f1_score, 4)
        }
        
    def get_field_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics summary for all fields.
        
        Returns:
            Dictionary with field names as keys and metrics as values
        """
        if not self.field_metrics:
            raise ValueError("No metrics available. Run calculate_metrics() first.")
        
        return {field_name: metrics.to_dict() 
                for field_name, metrics in self.field_metrics.items()}
        
    def save_detailed_results(self, output_path: Union[str, Path]) -> None:
        """
        Save detailed comparison results to JSON file.
        
        Args:
            output_path: Path where to save the results
        """
        output_path = Path(output_path)
        
        results = {
            "metadata": {
                "total_records_compared": len(set(c.uid for c in self.field_comparisons)),
                "total_field_comparisons": len(self.field_comparisons),
                "fields_evaluated": list(self.field_metrics.keys()),
                "comparison_method": "exact_match"
            },
            "overall_metrics": self.get_overall_metrics(),
            "field_metrics": self.get_field_metrics_summary(),
            "detailed_comparisons": [c.to_dict() for c in self.field_comparisons]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Detailed results saved to: {output_path}")
