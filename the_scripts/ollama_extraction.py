#!/usr/bin/env python3
"""
Ollama-Based Clinical Text Extraction Script

This script performs structured data extraction from clinical reports using the Ollama generator
with the comprehensive EchoReport Pydantic model. Designed to run in the venv_ollama environment.

Usage:
    ./the_venvs/venv_ollama/bin/python the_scripts/ollama_extraction.py
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Data processing
import pandas as pd

# ExtraCTOps imports
from utils.ExtraCTOps_loops import ProcessingConfig, ExtraCTOpsProcessor
from the_pydantics.EchoReport import EchoReport


class OllamaExtractor:
    """Ollama-specific extraction orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = project_root
        self.setup_output_directory()
    
    def setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        output_dir = self.project_root / "exports" / "ollama_extractions"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config["OUTPUT_DIR"] = output_dir
        print(f"📁 Output directory: {output_dir}")
    
    def create_sample_data(self) -> str:
        """Create sample echo report data for testing."""
        sample_reports = [
            {
                "patient_id": "ECHO_001",
                "study_date": "2024-01-15",
                "echo_report_text": """
ECHOCARDIOGRAM REPORT

Patient: 45-year-old male
Indication: Chest pain, rule out cardiac cause

FINDINGS:
Left Ventricle: The left ventricle is normal in size. Left ventricular systolic function is normal with an estimated ejection fraction of 65%. No regional wall motion abnormalities. LV diastolic volume 110 mL, systolic volume 38 mL.

Right Ventricle: The right ventricle is normal in size and systolic function.

Atria: The left atrium is mildly dilated. Right atrium is normal in size.

Valves: 
- Mitral valve is structurally normal with mild regurgitation
- Tricuspid valve shows mild regurgitation with estimated PA pressure 25 mmHg
- Aortic valve is structurally normal, trileaflet, no stenosis or regurgitation
- Pulmonary valve is normal with trivial regurgitation

Aorta: Aortic root measures 32 mm, ascending aorta 28 mm. Left aortic arch.

No pericardial effusion. No evidence of pulmonary hypertension.

IMPRESSION: Normal left ventricular size and systolic function. Mild left atrial dilation. Mild mitral and tricuspid regurgitation.
                """
            },
            {
                "patient_id": "ECHO_002", 
                "study_date": "2024-01-16",
                "echo_report_text": """
ECHOCARDIOGRAM REPORT

Patient: 62-year-old female
Indication: Hypertension, assessment of cardiac function

FINDINGS:
Left Ventricle: Moderate left ventricular hypertrophy. Estimated ejection fraction 45%, mildly depressed systolic function. LV diastolic volume 145 mL, systolic volume 80 mL.

Right Ventricle: Normal right ventricular size and function.

Atria: Both atria are moderately dilated. Left atrial volume indexed 38 mL/m².

Valves:
- Mitral valve shows mild stenosis and moderate regurgitation
- Aortic valve has mild stenosis with peak gradient 35 mmHg, mean gradient 20 mmHg
- Tricuspid regurgitation is moderate with elevated PA pressure 45 mmHg
- Pulmonary valve is normal

Great Vessels: Aortic root 35 mm, ascending aorta 40 mm.

Moderate pulmonary hypertension present with interventricular septal flattening in systole.

IMPRESSION: Moderate LV hypertrophy with mild systolic dysfunction. Moderate pulmonary hypertension. Mild aortic stenosis, moderate mitral regurgitation.
                """
            },
            {
                "patient_id": "ECHO_003",
                "study_date": "2024-01-17", 
                "echo_report_text": """
PEDIATRIC ECHOCARDIOGRAM REPORT

Patient: 8-year-old male
Indication: Heart murmur

FINDINGS:
Left Ventricle: Normal left ventricular size and systolic function, EF 65%.

Right Ventricle: Mild right ventricular dilation with normal systolic function.

Atria: Normal atrial sizes.

Septal Defects: 
- Small perimembranous ventricular septal defect, 4 mm, with left-to-right shunt
- Peak gradient across VSD 65 mmHg
- No atrial septal defect

Valves: All valves are structurally normal and competent.

Great Vessels: Normal aortic arch, no coarctation. Patent ductus arteriosus is absent.

Mild elevation of right heart pressures secondary to VSD.

IMPRESSION: Small perimembranous VSD with left-to-right shunt. Mild RV dilation. Normal valves and great vessels.
                """
            }
        ]
        
        df = pd.DataFrame(sample_reports)
        sample_file = self.project_root / "data" / "sample_echo_reports_ollama.csv"
        sample_file.parent.mkdir(exist_ok=True)
        df.to_csv(sample_file, index=False)
        
        print(f"✅ Created sample data: {sample_file}")
        print(f"📊 Sample data shape: {df.shape}")
        return str(sample_file)
    
    def load_and_validate_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV data and validate required columns."""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded data: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Validate required columns
            required_cols = [self.config["UID_COLUMN"], self.config["TEXT_COLUMN"]]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                print(f"   Available columns: {list(df.columns)}")
                return None
            
            # Show data preview
            print(f"\n📊 Data Preview:")
            print(df[required_cols].head())
            
            # Check for empty text fields
            empty_text = df[self.config["TEXT_COLUMN"]].isna().sum()
            if empty_text > 0:
                print(f"⚠️  Warning: {empty_text} rows have empty text fields")
                
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    async def run_extraction(self, input_file: str) -> Optional[ExtraCTOpsProcessor]:
        """Run Ollama extraction on the provided data."""
        # Load and validate data
        df = self.load_and_validate_data(input_file)
        if df is None:
            print("❌ Data loading failed. Cannot proceed with extraction.")
            return None
        
        print(f"\n🦙 Starting Ollama extraction...")
        print(f"   Model: {self.config['OLLAMA_MODEL']}")
        print(f"   Batch size: {self.config['BATCH_SIZE']}")
        print(f"   Temperature: {self.config['TEMPERATURE']}")
        print(f"   Records to process: {len(df)}")
        
        try:
            # Configure Ollama extraction
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = str(self.config["OUTPUT_DIR"] / f"echo_ollama_{timestamp}.xlsx")
            
            extraction_config = ProcessingConfig(
                input_file=input_file,
                text_column=self.config["TEXT_COLUMN"],
                uid_column=self.config["UID_COLUMN"],
                pydantic_model=EchoReport,
                generator_type="ollama",
                model_name=self.config["OLLAMA_MODEL"],
                experiment_label="ollama_echo_extraction",
                batch_size=self.config["BATCH_SIZE"],
                backup_interval=self.config["BACKUP_INTERVAL"],
                max_retries=self.config["MAX_RETRIES"],
                temperature=self.config["TEMPERATURE"],
                max_tokens=self.config["MAX_TOKENS"],
                system_message=self.config["SYSTEM_MESSAGE"],
                pre_prompt=self.config["PRE_PROMPT"],
                output_file=output_file
            )
            
            # Run extraction
            processor = ExtraCTOpsProcessor(extraction_config)
            await processor.process_batch(open_file=False)
            
            # Show results
            successful = len([r for r in processor.results if r.success])
            failed = len([r for r in processor.results if not r.success])
            avg_time = sum(r.execution_time for r in processor.results) / len(processor.results) if processor.results else 0
            
            print(f"\n✅ Ollama extraction completed!")
            print(f"   ✓ Successful: {successful}")
            print(f"   ✗ Failed: {failed}")
            print(f"   ⏱️ Average time: {avg_time:.2f}s per report")
            print(f"   📁 Output: {output_file}")
            
            return processor
            
        except Exception as e:
            print(f"❌ Ollama extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_summary_report(self, processor: ExtraCTOpsProcessor, input_file: str):
        """Save a summary report of the extraction."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        successful = len([r for r in processor.results if r.success])
        failed = len([r for r in processor.results if not r.success])
        avg_time = sum(r.execution_time for r in processor.results) / len(processor.results) if processor.results else 0
        
        summary_report = {
            "extraction_session": {
                "timestamp": timestamp,
                "generator": "ollama",
                "input_file": input_file,
                "total_reports": len(processor.results),
                "successful_extractions": successful,
                "failed_extractions": failed,
                "success_rate": (successful / len(processor.results)) * 100 if processor.results else 0,
                "average_time_per_report": avg_time,
                "configuration": {
                    "model": self.config["OLLAMA_MODEL"],
                    "batch_size": self.config["BATCH_SIZE"],
                    "temperature": self.config["TEMPERATURE"],
                    "max_tokens": self.config["MAX_TOKENS"]
                }
            },
            "pydantic_model": {
                "name": "EchoReport",
                "total_possible_fields": len(EchoReport.model_fields)
            }
        }
        
        # Save summary as JSON
        summary_file = self.config["OUTPUT_DIR"] / f"ollama_extraction_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        print(f"✅ Summary report saved: {summary_file}")
        return summary_report


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for Ollama extraction."""
    return {
        # Data settings
        "TEXT_COLUMN": "echo_report_text",
        "UID_COLUMN": "patient_id",
        
        # Processing settings
        "BATCH_SIZE": 3,  # Conservative for local processing
        "BACKUP_INTERVAL": 5,
        "MAX_RETRIES": 2,
        
        # LLM settings
        "TEMPERATURE": 0.1,
        "MAX_TOKENS": 3000,
        "OLLAMA_MODEL": "llama3.2:3b",  # Good balance of speed and quality
        
        # Prompts optimized for medical extraction
        "SYSTEM_MESSAGE": """You are an expert cardiologist and medical data extraction specialist. 
Extract structured echocardiogram information from clinical reports with high accuracy. 
Focus on cardiac anatomy, function, measurements, and pathology. 
Return only valid JSON that matches the provided schema exactly.""",
        
        "PRE_PROMPT": """Analyze this echocardiogram report and extract all relevant cardiac information. 
Include measurements with units, anatomical descriptions, functional assessments, and any abnormalities. 
Be precise with medical terminology and numerical values. Return valid JSON only."""
    }


def main():
    """Main function for Ollama extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract structured data from clinical reports using Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use sample data
  python ollama_extraction.py --sample

  # Process your own CSV file
  python ollama_extraction.py --input /path/to/your/reports.csv

  # Custom model and batch size
  python ollama_extraction.py --input data.csv --model llama3:8b --batch-size 5
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to CSV file containing echo reports"
    )
    
    parser.add_argument(
        "--sample", "-s",
        action="store_true",
        help="Use sample data for testing"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3.2:3b",
        help="Ollama model to use (default: llama3.2:3b)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=3,
        help="Number of concurrent extractions (default: 3)"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.sample and not args.input:
        print("❌ Error: Must specify either --sample or --input")
        parser.print_help()
        sys.exit(1)
    
    # Setup configuration
    config = get_default_config()
    config["OLLAMA_MODEL"] = args.model
    config["BATCH_SIZE"] = args.batch_size
    config["TEMPERATURE"] = args.temperature
    
    print("🦙 OLLAMA CLINICAL TEXT EXTRACTION")
    print("=" * 50)
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Project root: {project_root}")
    print(f"🔧 Model: {config['OLLAMA_MODEL']}")
    print(f"⚙️ Batch size: {config['BATCH_SIZE']}")
    print(f"🌡️ Temperature: {config['TEMPERATURE']}")
    
    # Initialize extractor
    extractor = OllamaExtractor(config)
    
    # Determine input file
    if args.sample:
        print("\n📊 Creating sample data...")
        input_file = extractor.create_sample_data()
    else:
        input_file = args.input
        if not Path(input_file).exists():
            print(f"❌ Error: Input file not found: {input_file}")
            sys.exit(1)
    
    print(f"\n📥 Input file: {input_file}")
    
    # Run extraction
    async def run_async_extraction():
        processor = await extractor.run_extraction(input_file)
        if processor:
            extractor.save_summary_report(processor, input_file)
            print(f"\n🎉 Extraction complete! Check {config['OUTPUT_DIR']} for results.")
        else:
            print("❌ Extraction failed!")
            sys.exit(1)
    
    # Run the async extraction
    asyncio.run(run_async_extraction())


if __name__ == "__main__":
    main()
