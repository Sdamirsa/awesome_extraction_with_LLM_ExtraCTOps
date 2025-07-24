# Performance Evaluation Module for ExtraCTOps
# This module provides tools to evaluate extraction performance against ground truth

from .evaluator import (
    ExtractionEvaluator,
    ComparisonMethod,
    EvaluationResult,
    FieldComparison,
    FieldMetrics
)

from .reporter import PerformanceReporter

__version__ = "1.0.0"
__author__ = "ExtraCTOps Team"

__all__ = [
    "ExtractionEvaluator",
    "PerformanceReporter", 
    "ComparisonMethod",
    "EvaluationResult",
    "FieldComparison",
    "FieldMetrics"
]
