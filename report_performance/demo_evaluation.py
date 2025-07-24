#!/usr/bin/env python3
"""
Demonstration script for ExtraCTOps Performance Evaluation

This script demonstrates how to use the performance evaluation module
with sample data, including the provided ground truth JSON file.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from report_performance.evaluator import ExtractionEvaluator, ComparisonMethod
from report_performance.reporter import PerformanceReporter


def create_sample_extraction_data() -> pd.DataFrame:
    """
    Create sample extraction data that partially matches the ground truth.
    This simulates the output from an extraction system.
    """
    # Note: Based on the provided ground truth, we have these fields to work with
    sample_data = [
        {
            "id": "row_1",
            "row_index": 0,
            "atria::RA::RA_dilation": "Mild",  # Different from GT "Aplastic"
            "atria::LA::LA_dilation": "Mild",  # GT is null
            "atria::LA::LA_volume_indexed::numeric": 35.0,  # GT is null
            "atria::LA::LA_volume_indexed::unit": "mL/m²",  # GT is empty
            "ventricles::RV::RV_size_structure::RV_dilation": "Normal",  # GT is null
            "ventricles::RV::RV_size_structure::RV_hypertrophy": "None",  # GT is null
            "ventricles::RV::RV_function::RV_systolic_function": "Normal",  # GT is null
            "ventricles::LV::LV_size_structure::LV_dilation": "Normal",  # GT is null
            "ventricles::LV::LV_size_structure::LV_hypertrophy": "Mild",  # GT is null
            "ventricles::LV::LV_size_structure::LV_volume_systole::numeric": 45.0,  # GT is null
            "ventricles::LV::LV_size_structure::LV_volume_systole::unit": "mL",  # GT is empty
            "ventricles::LV::LV_size_structure::LV_volume_diastole::numeric": 120.0,  # GT is null
            "ventricles::LV::LV_size_structure::LV_volume_diastole::unit": "mL",  # GT is empty
            "ventricles::LV::LV_function::LV_systolic_function": "Normal",  # GT is null
            "ventricles::LV::LV_function::LV_systolic_function_other": "",  # GT is empty
            "ventricles::LV::LV_function::LVEF::numeric": 60.0,  # GT is null
            "ventricles::LV::LV_function::LVEF::unit": "%",  # GT is empty
            "valves::tricuspid::TV_structural_status": "Normal",  # GT is null
            "valves::tricuspid::TV_structural_status_other": "",  # GT is empty
            "valves::tricuspid::TV_regurgitation_severity": "Mild",  # GT is null
            "valves::pulmonary::PV_annulus_size::numeric": 22.0,  # GT is null
            "valves::pulmonary::PV_annulus_size::unit": "mm",  # GT is empty
            "raw_PID": 111,  # Should match GT
            "raw_MRE_Report": "Patient ID: 1004827\nDate of Encounter: 2025-03-12\nFacility: St. Helena General Hospital"  # Partial match
        }
    ]
    
    return pd.DataFrame(sample_data)


def load_ground_truth_from_attachment() -> pd.DataFrame:
    """
    Load the ground truth data from the provided JSON attachment.
    This simulates loading the manual extraction results.
    """
    # The ground truth data provided in the attachment
    ground_truth_data = [
        {
            "id": "row_1",
            "row_index": 0,
            "atria::RA::RA_dilation": "Aplastic",
            "atria::LA::LA_dilation": None,
            "atria::LA::LA_volume_indexed::numeric": None,
            "atria::LA::LA_volume_indexed::unit": "",
            "ventricles::RV::RV_size_structure::RV_dilation": None,
            "ventricles::RV::RV_size_structure::RV_hypertrophy": None,
            "ventricles::RV::RV_function::RV_systolic_function": None,
            "ventricles::LV::LV_size_structure::LV_dilation": None,
            "ventricles::LV::LV_size_structure::LV_hypertrophy": None,
            "ventricles::LV::LV_size_structure::LV_volume_systole::numeric": None,
            "ventricles::LV::LV_size_structure::LV_volume_systole::unit": "",
            "ventricles::LV::LV_size_structure::LV_volume_diastole::numeric": None,
            "ventricles::LV::LV_size_structure::LV_volume_diastole::unit": "",
            "ventricles::LV::LV_function::LV_systolic_function": None,
            "ventricles::LV::LV_function::LV_systolic_function_other": "",
            "ventricles::LV::LV_function::LVEF::numeric": None,
            "ventricles::LV::LV_function::LVEF::unit": "",
            "valves::tricuspid::TV_structural_status": None,
            "valves::tricuspid::TV_structural_status_other": "",
            "valves::tricuspid::TV_regurgitation_severity": None,
            "valves::pulmonary::PV_annulus_size::numeric": None,
            "valves::pulmonary::PV_annulus_size::unit": "",
            "valves::pulmonary::PV_stenosis_severity": None,
            "valves::pulmonary::PV_structural_status": None,
            "valves::pulmonary::PV_structural_status_other": "",
            "valves::pulmonary::PV_regurgitation_severity": None,
            "valves::pulmonary::PV_pressure_gradient::numeric": None,
            "valves::pulmonary::PV_pressure_gradient::unit": "",
            "valves::mitral::MV_stenosis_severity": None,
            "valves::mitral::MV_structural_status": None,
            "valves::mitral::MV_structural_status_other": "",
            "valves::mitral::MV_regurgitation_severity": None,
            "valves::aortic::AV_structural_status": None,
            "valves::aortic::AV_structural_status_other": "",
            "valves::aortic::AV_leaflets": None,
            "valves::aortic::AV_stenosis_severity": None,
            "valves::aortic::AV_regurgitation_severity": None,
            "valves::aortic::AV_peak_pressure_gradient::numeric": None,
            "valves::aortic::AV_peak_pressure_gradient::unit": "",
            "valves::aortic::AV_mean_pressure_gradient::numeric": None,
            "valves::aortic::AV_mean_pressure_gradient::unit": "",
            "great_vessels::aorta::arch_sidedness": "",
            "great_vessels::aorta::aortic_root_size::numeric": None,
            "great_vessels::aorta::aortic_root_size::unit": "",
            "great_vessels::aorta::ascending_aorta_diameter::numeric": None,
            "great_vessels::aorta::ascending_aorta_diameter::unit": "",
            "great_vessels::aorta::aortic_isthmus_size::numeric": None,
            "great_vessels::aorta::aortic_isthmus_size::unit": "",
            "great_vessels::aorta::coarctation": None,
            "great_vessels::aorta::coarctation_gradient::numeric": None,
            "great_vessels::aorta::coarctation_gradient::unit": "",
            "pHTN::severity": "",
            "pHTN::TR_jet_gradient::numeric": None,
            "pHTN::TR_jet_gradient::unit": "",
            "pHTN::IVS_flattening_in_systole": None,
            "asd::atrial_communication_present": None,
            "asd::atrial_communication_present_other": "",
            "asd::size": None,
            "asd::direction_of_flow": "",
            "vsd::ventricular_communication_present": None,
            "vsd::ventricular_communication_present_other": "",
            "vsd::size": None,
            "vsd::direction_of_flow": "",
            "vsd::peak_gradient::numeric": None,
            "vsd::peak_gradient::unit": "",
            "pda::present": None,
            "pda::direction_of_flow": "",
            "pda::size::numeric": None,
            "pda::size::unit": "",
            "pda::peak_gradient::numeric": None,
            "pda::peak_gradient::unit": "",
            "surgical_history::prior_surgical_interventions": "",
            "raw_PID": 111,
            "raw_MRE_Report": "Patient ID: 1004827\nDate of Encounter: 2025-03-12\nFacility: St. Helena General Hospital\nProvider: Dr. Maria Lopez\n\nChief Complaint: Chest pain and shortness of breath\n\nExtracted Diagnoses:\n\t•\tAcute Myocardial Infarction (ICD-10: I21.9)\n\t•\tHypertension (ICD-10: I10)\n\nMedications Extracted:\n\t•\tAspirin 81 mg, PO, daily\n\t•\tAtorvastatin 40 mg, PO, daily\n\t•\tLisinopril 10 mg, PO, daily\n\nKey Procedures:\n\t•\tECG performed, abnormal ST-elevation\n\t•\tCardiac catheterization with stent placement\n\nFollow-up Recommendations:\n\t•\tCardiologist visit in 1 week\n\t•\tLifestyle counseling (diet + exercise)"
        }
    ]
    
    return pd.DataFrame(ground_truth_data)


def run_demonstration():
    """Run the complete demonstration of the performance evaluation system."""
    print("🔬 PERFORMANCE EVALUATION DEMONSTRATION")
    print("=" * 50)
    print("This demonstration shows how to evaluate extraction performance")
    print("using the ExtraCTOps Performance Evaluation module.\n")
    
    # Create sample data
    print("📊 Creating sample extraction data...")
    extraction_data = create_sample_extraction_data()
    print(f"   ✓ Created extraction data: {extraction_data.shape}")
    
    print("📋 Loading ground truth data...")
    ground_truth_data = load_ground_truth_from_attachment()
    print(f"   ✓ Loaded ground truth data: {ground_truth_data.shape}")
    
    # Save data to temporary files for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        extraction_file = temp_path / "sample_extraction.xlsx"
        ground_truth_file = temp_path / "sample_ground_truth.xlsx"
        
        extraction_data.to_excel(extraction_file, index=False)
        ground_truth_data.to_excel(ground_truth_file, index=False)
        
        print(f"\n🔍 EVALUATION PROCESS")
        print("-" * 30)
        
        # Initialize evaluator
        print("1. Initializing evaluator...")
        evaluator = ExtractionEvaluator(uid_column="id")
        
        # Load data
        print("2. Loading data files...")
        evaluator.load_extraction_data(extraction_file)
        evaluator.load_ground_truth_data(ground_truth_file)
        
        # Perform comparison
        print("3. Comparing records...")
        evaluator.compare_records(ComparisonMethod.EXACT_MATCH)
        
        print("4. Calculating metrics...")
        evaluator.calculate_metrics()
        
        # Display results
        print(f"\n📈 RESULTS SUMMARY")
        print("=" * 30)
        
        overall_metrics = evaluator.get_overall_metrics()
        print(f"Total Comparisons: {overall_metrics['total_comparisons']:,}")
        print(f"Overall Accuracy: {overall_metrics['overall_accuracy_percent']:.1f}%")
        print(f"Average F1 Score: {overall_metrics['average_f1_score']:.3f}")
        print(f"Average Missing Rate: {overall_metrics['average_missing_rate_percent']:.1f}%")
        print(f"Average Hallucination Rate: {overall_metrics['average_hallucination_rate_percent']:.1f}%")
        print(f"Correctness When Extracted: {overall_metrics['average_correctness_when_extracted_percent']:.1f}%")
        
        # Show field-level breakdown for interesting fields
        print(f"\n📋 FIELD-LEVEL ANALYSIS (Top 10 Fields)")
        print("-" * 40)
        field_metrics = evaluator.get_field_metrics_summary()
        
        # Sort fields by total comparisons (most active fields first)
        sorted_fields = sorted(field_metrics.items(), 
                             key=lambda x: x[1]['total_comparisons'], 
                             reverse=True)
        
        print(f"{'Field Name':<40} {'F1':<6} {'Miss%':<6} {'Hall%':<6} {'Corr%':<6}")
        print("-" * 70)
        
        for field_name, metrics in sorted_fields[:10]:
            # Truncate long field names
            display_name = field_name[:39] if len(field_name) > 39 else field_name
            print(f"{display_name:<40} "
                  f"{metrics['f1_score']:<6.3f} "
                  f"{metrics['missing_rate_percent']:<6.1f} "
                  f"{metrics['hallucination_rate_percent']:<6.1f} "
                  f"{metrics['correctness_when_extracted_percent']:<6.1f}")
        
        # Show some specific examples
        print(f"\n🔍 DETAILED EXAMPLES")
        print("-" * 30)
        
        # Find some interesting comparisons
        examples = []
        for comparison in evaluator.field_comparisons[:10]:  # First 10 comparisons
            if comparison.result.value in ['correct', 'incorrect']:
                examples.append(comparison)
                if len(examples) >= 3:
                    break
        
        for i, comp in enumerate(examples, 1):
            print(f"\nExample {i}: {comp.field_name}")
            print(f"  Extracted: '{comp.extracted_value}'")
            print(f"  Ground Truth: '{comp.ground_truth_value}'")
            print(f"  Result: {comp.result.value.upper()}")
        
        # Generate reports
        print(f"\n📊 GENERATING REPORTS")
        print("-" * 30)
        
        # Create output directory in project
        output_dir = project_root / "temp" / f"demo_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {output_dir}")
        
        reporter = PerformanceReporter(evaluator)
        reporter.set_output_directory(output_dir)
        
        # Generate individual components
        print("Creating visualizations...")
        summary_plot = reporter.create_performance_summary_plot()
        print(f"  ✓ Performance summary: {summary_plot.name}")
        
        overview_plot = reporter.create_metrics_overview_plot()
        print(f"  ✓ Metrics overview: {overview_plot.name}")
        
        # Try to create confusion matrix for an interesting field
        interesting_field = sorted_fields[0][0] if sorted_fields else None
        if interesting_field:
            confusion_plot = reporter.create_confusion_matrix_plot(interesting_field)
            print(f"  ✓ Confusion matrix for '{interesting_field}': {confusion_plot.name}")
        
        # Save detailed results
        detailed_path = output_dir / "detailed_results.json"
        evaluator.save_detailed_results(detailed_path)
        print(f"  ✓ Detailed results: {detailed_path.name}")
        
        # Generate Word report if possible
        word_report = reporter.generate_word_report()
        if word_report:
            print(f"  ✓ Word report: {word_report.name}")
        else:
            print(f"  ⚠️  Word report skipped (python-docx not available)")
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print(f"Report files saved to: {output_dir}")
        print(f"\nTo use this module with your own data:")
        print(f"1. Prepare your extraction results (Excel/JSON)")
        print(f"2. Prepare your ground truth data (Excel/JSON)")
        print(f"3. Use the command-line tool:")
        print(f"   python report_performance/evaluate_performance.py \\")
        print(f"     --extraction your_results.xlsx \\")
        print(f"     --ground-truth your_ground_truth.xlsx \\")
        print(f"     --uid-column id")
        
        # Show some key insights
        print(f"\n💡 KEY INSIGHTS FROM THIS DEMO:")
        print("-" * 40)
        
        # Calculate some interesting stats
        total_fields = len(field_metrics)
        fields_with_data = len([f for f in field_metrics.values() 
                               if f['correct'] > 0 or f['incorrect'] > 0])
        avg_f1 = overall_metrics['average_f1_score']
        
        print(f"• Total fields evaluated: {total_fields}")
        print(f"• Fields with extracted data: {fields_with_data}")
        print(f"• Most ground truth fields are empty (as expected for clinical data)")
        print(f"• Hallucination rate shows how often extraction produces values")
        print(f"  when ground truth is empty - this can be normal in clinical extraction")
        print(f"• F1 score of {avg_f1:.3f} indicates room for improvement")
        
        return output_dir


if __name__ == "__main__":
    try:
        output_directory = run_demonstration()
        print(f"\n✅ Demonstration completed successfully!")
        print(f"📁 Check the output directory for generated reports: {output_directory}")
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
