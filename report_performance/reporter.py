"""
Report Generation Module for ExtraCTOps Performance Evaluation

This module handles generating comprehensive reports including visualizations
and Word documents for extraction performance analysis.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Word document generation
try:
    from docx import Document
    from docx.shared import Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️  python-docx not available. Word report generation will be disabled.")

from .evaluator import ExtractionEvaluator, FieldMetrics


class PerformanceReporter:
    """
    Generates comprehensive performance reports with visualizations and documentation.
    """
    
    def __init__(self, evaluator: ExtractionEvaluator):
        """
        Initialize the reporter with an evaluator instance.
        
        Args:
            evaluator: ExtractionEvaluator instance with completed analysis
        """
        self.evaluator = evaluator
        self.output_dir: Optional[Path] = None
        
    def set_output_directory(self, output_dir: Union[str, Path]) -> None:
        """
        Set the output directory for reports and visualizations.
        
        Args:
            output_dir: Directory where reports will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory set to: {self.output_dir}")
        
    def create_confusion_matrix_plot(self, field_name: str, save_path: Optional[Path] = None) -> Path:
        """
        Create a confusion matrix visualization for a specific field.
        
        Args:
            field_name: Name of the field to visualize
            save_path: Optional path to save the plot
            
        Returns:
            Path where the plot was saved
        """
        if field_name not in self.evaluator.field_metrics:
            raise ValueError(f"Field '{field_name}' not found in metrics")
        
        metrics = self.evaluator.field_metrics[field_name]
        
        # Create confusion matrix data
        matrix_data = np.array([
            [metrics.correct, metrics.incorrect, 0],  # When GT has value
            [0, 0, metrics.no_information],  # When GT has no value
            [0, 0, 0]  # Placeholder for formatting
        ])
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix_data, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=['Correct', 'Incorrect', 'No Information'],
                   yticklabels=['GT Has Value', 'GT No Value', ''])
        
        plt.title(f'Confusion Matrix: {field_name}')
        plt.xlabel('Extraction Result')
        plt.ylabel('Ground Truth Status')
        
        if save_path is None:
            save_path = self.output_dir / f"confusion_matrix_{field_name.replace(':', '_')}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def create_metrics_overview_plot(self, save_path: Optional[Path] = None) -> Path:
        """
        Create an overview plot of key metrics across all fields.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Path where the plot was saved
        """
        # Prepare data
        field_names = list(self.evaluator.field_metrics.keys())
        f1_scores = [self.evaluator.field_metrics[f].f1_score for f in field_names]
        missing_rates = [self.evaluator.field_metrics[f].missing_rate * 100 for f in field_names]
        hallucination_rates = [self.evaluator.field_metrics[f].hallucination_rate * 100 for f in field_names]
        correctness_when_extracted = [self.evaluator.field_metrics[f].correctness_when_extracted * 100 for f in field_names]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # F1 Scores
        axes[0, 0].barh(field_names, f1_scores, color='skyblue')
        axes[0, 0].set_title('F1 Scores by Field')
        axes[0, 0].set_xlabel('F1 Score')
        axes[0, 0].set_xlim(0, 1)
        
        # Missing Rates
        axes[0, 1].barh(field_names, missing_rates, color='lightcoral')
        axes[0, 1].set_title('Missing Rates by Field (%)')
        axes[0, 1].set_xlabel('Missing Rate (%)')
        
        # Hallucination Rates
        axes[1, 0].barh(field_names, hallucination_rates, color='lightgreen')
        axes[1, 0].set_title('Hallucination Rates by Field (%)')
        axes[1, 0].set_xlabel('Hallucination Rate (%)')
        
        # Correctness When Extracted
        axes[1, 1].barh(field_names, correctness_when_extracted, color='gold')
        axes[1, 1].set_title('Correctness When Extracted (%)')
        axes[1, 1].set_xlabel('Correctness (%)')
        
        # Adjust layout
        for ax in axes.flat:
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(axis='x', alpha=0.3)
        
        if save_path is None:
            save_path = self.output_dir / "metrics_overview.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def create_performance_summary_plot(self, save_path: Optional[Path] = None) -> Path:
        """
        Create a summary plot of overall performance.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Path where the plot was saved
        """
        overall_metrics = self.evaluator.get_overall_metrics()
        
        # Prepare data for pie chart of results
        labels = ['Correct', 'Incorrect', 'No Information']
        sizes = [
            overall_metrics['total_correct'],
            overall_metrics['total_incorrect'],
            overall_metrics['total_no_information']
        ]
        colors = ['lightgreen', 'lightcoral', 'lightgray']
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart of results
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribution of Extraction Results')
        
        # Bar chart of key metrics
        metrics_names = ['Accuracy', 'Avg F1', 'Avg Missing Rate', 'Avg Hallucination Rate', 'Avg Correctness When Extracted']
        metrics_values = [
            overall_metrics['overall_accuracy_percent'],
            overall_metrics['average_f1_score'] * 100,
            overall_metrics['average_missing_rate_percent'],
            overall_metrics['average_hallucination_rate_percent'],
            overall_metrics['average_correctness_when_extracted_percent']
        ]
        
        bars = ax2.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'orange', 'gold'])
        ax2.set_title('Overall Performance Metrics')
        ax2.set_ylabel('Percentage / Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        if save_path is None:
            save_path = self.output_dir / "performance_summary.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def generate_word_report(self, report_path: Optional[Path] = None) -> Optional[Path]:
        """
        Generate a comprehensive Word document report.
        
        Args:
            report_path: Optional path for the Word document
            
        Returns:
            Path where the report was saved, or None if docx not available
        """
        if not DOCX_AVAILABLE:
            print("⚠️  Word report generation skipped - python-docx not available")
            return None
        
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"extraction_performance_report_{timestamp}.docx"
        
        # Create document
        doc = Document()
        
        # Title
        title = doc.add_heading('ExtraCTOps Extraction Performance Report', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Executive Summary
        doc.add_heading('Executive Summary', level=1)
        overall_metrics = self.evaluator.get_overall_metrics()
        
        summary_text = f"""
This report presents a comprehensive analysis of extraction performance comparing automated results against ground truth data.

Key Findings:
• Total Comparisons: {overall_metrics['total_comparisons']:,}
• Overall Accuracy: {overall_metrics['overall_accuracy_percent']:.1f}%
• Average F1 Score: {overall_metrics['average_f1_score']:.3f}
• Average Missing Rate: {overall_metrics['average_missing_rate_percent']:.1f}%
• Average Hallucination Rate: {overall_metrics['average_hallucination_rate_percent']:.1f}%
• Correctness When Extracted: {overall_metrics['average_correctness_when_extracted_percent']:.1f}%
        """
        doc.add_paragraph(summary_text.strip())
        
        # Methodology
        doc.add_heading('Methodology', level=1)
        methodology_text = """
This evaluation compares automated extraction results with manually validated ground truth data using exact match comparison. The analysis includes:

1. Exact Match Comparison: Direct comparison of extracted values with ground truth
2. Three-Category Classification: Results classified as Correct, Incorrect, or No Information
3. Specialized Metrics:
   - Missing Rate: Ground truth has value but extraction doesn't
   - Hallucination Rate: Ground truth has no value but extraction provides one
   - Correctness When Extracted: Accuracy among records where extraction provided a value
4. F1 Score Calculation: Based on precision and recall considering the three-category system
        """
        doc.add_paragraph(methodology_text.strip())
        
        # Overall Results
        doc.add_heading('Overall Results', level=1)
        
        # Add performance summary plot
        summary_plot_path = self.create_performance_summary_plot()
        doc.add_paragraph("Performance Summary:")
        doc.add_picture(str(summary_plot_path), width=Inches(6))
        
        # Field-by-Field Analysis
        doc.add_heading('Field-by-Field Analysis', level=1)
        
        # Add metrics overview plot
        overview_plot_path = self.create_metrics_overview_plot()
        doc.add_paragraph("Metrics Overview Across All Fields:")
        doc.add_picture(str(overview_plot_path), width=Inches(6))
        
        # Detailed field metrics table
        doc.add_heading('Detailed Field Metrics', level=2)
        
        # Create table
        table = doc.add_table(rows=1, cols=7)
        table.style = 'Light Grid Accent 1'
        
        # Header row
        header_cells = table.rows[0].cells
        headers = ['Field Name', 'F1 Score', 'Missing Rate (%)', 'Hallucination Rate (%)', 
                  'Correctness When Extracted (%)', 'Correct', 'Incorrect']
        for i, header in enumerate(headers):
            header_cells[i].text = header
        
        # Data rows
        for field_name, metrics in self.evaluator.field_metrics.items():
            row_cells = table.add_row().cells
            row_cells[0].text = field_name
            row_cells[1].text = f"{metrics.f1_score:.3f}"
            row_cells[2].text = f"{metrics.missing_rate * 100:.1f}"
            row_cells[3].text = f"{metrics.hallucination_rate * 100:.1f}"
            row_cells[4].text = f"{metrics.correctness_when_extracted * 100:.1f}"
            row_cells[5].text = str(metrics.correct)
            row_cells[6].text = str(metrics.incorrect)
        
        # Recommendations
        doc.add_heading('Recommendations', level=1)
        
        # Identify problem fields
        low_f1_fields = [name for name, metrics in self.evaluator.field_metrics.items() 
                        if metrics.f1_score < 0.5]
        high_missing_fields = [name for name, metrics in self.evaluator.field_metrics.items() 
                              if metrics.missing_rate > 0.3]
        high_hallucination_fields = [name for name, metrics in self.evaluator.field_metrics.items() 
                                   if metrics.hallucination_rate > 0.2]
        
        recommendations_text = f"""
Based on the analysis, the following recommendations are provided:

Performance Issues Identified:
• {len(low_f1_fields)} fields with F1 score < 0.5: Require attention for overall improvement
• {len(high_missing_fields)} fields with missing rate > 30%: Need improved recall
• {len(high_hallucination_fields)} fields with hallucination rate > 20%: Need improved precision

Recommended Actions:
1. Review extraction prompts for low-performing fields
2. Enhance training data for fields with high missing rates
3. Implement stricter validation for fields with high hallucination rates
4. Consider field-specific extraction strategies for complex structured data
        """
        doc.add_paragraph(recommendations_text.strip())
        
        # Technical Details
        doc.add_heading('Technical Details', level=1)
        doc.add_paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc.add_paragraph(f"Total fields analyzed: {len(self.evaluator.field_metrics)}")
        doc.add_paragraph(f"Comparison method: Exact Match")
        doc.add_paragraph(f"UID column used: {self.evaluator.uid_column}")
        
        # Save document
        doc.save(report_path)
        print(f"✅ Word report saved to: {report_path}")
        
        return report_path
        
    def generate_all_visualizations(self) -> Dict[str, Path]:
        """
        Generate all visualizations and save them.
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if self.output_dir is None:
            raise ValueError("Output directory not set. Call set_output_directory() first.")
        
        visualizations = {}
        
        # Overall plots
        visualizations['performance_summary'] = self.create_performance_summary_plot()
        visualizations['metrics_overview'] = self.create_metrics_overview_plot()
        
        # Individual field confusion matrices (limit to top 10 fields by total comparisons)
        field_counts = [(name, metrics.total_comparisons) 
                       for name, metrics in self.evaluator.field_metrics.items()]
        field_counts.sort(key=lambda x: x[1], reverse=True)
        
        for field_name, _ in field_counts[:10]:  # Top 10 fields
            try:
                path = self.create_confusion_matrix_plot(field_name)
                visualizations[f'confusion_matrix_{field_name}'] = path
            except Exception as e:
                print(f"⚠️  Warning: Could not create confusion matrix for {field_name}: {e}")
        
        print(f"✅ Generated {len(visualizations)} visualizations")
        return visualizations
        
    def generate_complete_report(self) -> Dict[str, Path]:
        """
        Generate a complete report with all visualizations and documents.
        
        Returns:
            Dictionary mapping report components to file paths
        """
        if self.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.set_output_directory(Path(f"extraction_performance_report_{timestamp}"))
        
        print("📊 Generating complete performance report...")
        
        report_files = {}
        
        # Generate visualizations
        visualizations = self.generate_all_visualizations()
        report_files.update(visualizations)
        
        # Generate detailed JSON results
        json_path = self.output_dir / "detailed_results.json"
        self.evaluator.save_detailed_results(json_path)
        report_files['detailed_results_json'] = json_path
        
        # Generate Word report if available
        word_report = self.generate_word_report()
        if word_report:
            report_files['word_report'] = word_report
        
        # Generate summary metrics JSON
        summary_path = self.output_dir / "summary_metrics.json"
        summary_data = {
            "overall_metrics": self.evaluator.get_overall_metrics(),
            "field_metrics": self.evaluator.get_field_metrics_summary()
        }
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        report_files['summary_metrics_json'] = summary_path
        
        print(f"🎉 Complete report generated in: {self.output_dir}")
        print(f"   Generated {len(report_files)} files")
        
        return report_files
