#!/usr/bin/env python3
"""
Command-line interface for ExtraCTOps Performance Evaluation

This script provides a convenient way to evaluate extraction performance
against ground truth data from the command line.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from report_performance.evaluator import ExtractionEvaluator, ComparisonMethod
from report_performance.reporter import PerformanceReporter


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate extraction performance against ground truth data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_performance.py \\
    --extraction results.xlsx \\
    --ground-truth manual_validation.json \\
    --uid-column id

  # Custom output directory
  python evaluate_performance.py \\
    --extraction results.xlsx \\
    --ground-truth ground_truth.xlsx \\
    --output-dir my_evaluation_2024 \\
    --uid-column patient_id

  # Quick summary only (no visualizations)
  python evaluate_performance.py \\
    --extraction results.json \\
    --ground-truth ground_truth.json \\
    --summary-only

Supported file formats: .xlsx, .xls, .json
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--extraction", "-e",
        type=str,
        required=True,
        help="Path to extraction results file (Excel or JSON)"
    )
    
    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        required=True,
        help="Path to ground truth file (Excel or JSON)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--uid-column", "-u",
        type=str,
        default="id",
        help="Name of the UID column for matching records (default: id)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for reports (default: auto-generated)"
    )
    
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["exact_match"],
        default="exact_match",
        help="Comparison method to use (default: exact_match)"
    )
    
    parser.add_argument(
        "--summary-only", "-s",
        action="store_true",
        help="Generate only summary metrics (no visualizations or Word report)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    parser.add_argument(
        "--json-output",
        type=str,
        help="Save summary metrics to specified JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    extraction_path = Path(args.extraction)
    ground_truth_path = Path(args.ground_truth)
    
    if not extraction_path.exists():
        print(f"❌ Error: Extraction file not found: {extraction_path}")
        sys.exit(1)
        
    if not ground_truth_path.exists():
        print(f"❌ Error: Ground truth file not found: {ground_truth_path}")
        sys.exit(1)
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"performance_evaluation_{timestamp}")
    
    if not args.quiet:
        print("🔍 EXTRACTION PERFORMANCE EVALUATION")
        print("=" * 50)
        print(f"📥 Extraction file: {extraction_path}")
        print(f"📥 Ground truth file: {ground_truth_path}")
        print(f"🔑 UID column: {args.uid_column}")
        print(f"📁 Output directory: {output_dir}")
        print(f"⚙️ Method: {args.method}")
        print()
    
    try:
        # Initialize evaluator
        if not args.quiet:
            print("🚀 Initializing evaluator...")
        
        evaluator = ExtractionEvaluator(uid_column=args.uid_column)
        
        # Load data
        if not args.quiet:
            print("📂 Loading extraction data...")
        evaluator.load_extraction_data(extraction_path)
        
        if not args.quiet:
            print("📂 Loading ground truth data...")
        evaluator.load_ground_truth_data(ground_truth_path)
        
        # Perform comparison
        if not args.quiet:
            print("🔄 Comparing records...")
        
        if args.method == "exact_match":
            method = ComparisonMethod.EXACT_MATCH
        else:
            raise ValueError(f"Unsupported method: {args.method}")
            
        evaluator.compare_records(method)
        
        # Calculate metrics
        if not args.quiet:
            print("📊 Calculating metrics...")
        evaluator.calculate_metrics()
        
        # Get and display summary
        overall_metrics = evaluator.get_overall_metrics()
        
        if not args.quiet:
            print("\n📈 PERFORMANCE SUMMARY")
            print("=" * 30)
            print(f"Total Comparisons: {overall_metrics['total_comparisons']:,}")
            print(f"Overall Accuracy: {overall_metrics['overall_accuracy_percent']:.1f}%")
            print(f"Average F1 Score: {overall_metrics['average_f1_score']:.3f}")
            print(f"Average Missing Rate: {overall_metrics['average_missing_rate_percent']:.1f}%")
            print(f"Average Hallucination Rate: {overall_metrics['average_hallucination_rate_percent']:.1f}%")
            print(f"Correctness When Extracted: {overall_metrics['average_correctness_when_extracted_percent']:.1f}%")
            print()
        
        # Save summary JSON if requested
        if args.json_output:
            summary_data = {
                "overall_metrics": overall_metrics,
                "field_metrics": evaluator.get_field_metrics_summary(),
                "metadata": {
                    "extraction_file": str(extraction_path),
                    "ground_truth_file": str(ground_truth_path),
                    "uid_column": args.uid_column,
                    "comparison_method": args.method,
                    "evaluation_timestamp": datetime.now().isoformat()
                }
            }
            
            json_path = Path(args.json_output)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            if not args.quiet:
                print(f"✅ Summary saved to: {json_path}")
        
        # Generate full report if not summary-only
        if not args.summary_only:
            if not args.quiet:
                print("📊 Generating comprehensive report...")
            
            reporter = PerformanceReporter(evaluator)
            reporter.set_output_directory(output_dir)
            
            report_files = reporter.generate_complete_report()
            
            if not args.quiet:
                print(f"\n🎉 EVALUATION COMPLETE!")
                print(f"📁 Report location: {output_dir}")
                print(f"📄 Generated {len(report_files)} files:")
                for name, path in report_files.items():
                    print(f"   • {name}: {path.name}")
        
        else:
            if not args.quiet:
                print("✅ Summary-only evaluation complete!")
        
        # Top problematic fields
        field_metrics = evaluator.get_field_metrics_summary()
        low_f1_fields = [(name, metrics['f1_score']) for name, metrics in field_metrics.items() 
                        if metrics['f1_score'] < 0.5]
        
        if low_f1_fields and not args.quiet:
            print(f"\n⚠️  Fields with F1 < 0.5 (need attention):")
            low_f1_fields.sort(key=lambda x: x[1])  # Sort by F1 score
            for field_name, f1_score in low_f1_fields[:5]:  # Top 5 problematic
                print(f"   • {field_name}: F1 = {f1_score:.3f}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
