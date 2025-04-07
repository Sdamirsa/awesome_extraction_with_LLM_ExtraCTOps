

## 📂 ExtraCTOps – Project Summary

**ExtraCTOps** is a modular system designed to extract, evaluate, and optimize structured data from unstructured sources such as clinical notes, PDFs, HTML documents, and voice recordings. It integrates both local and API-based LLM/VLM backends, while offering robust tooling for pseudonymization, schema definition, prompt refinement, and explainability — all in a workflow-friendly setup.

---

### 🧩 Key Features

- **Extraction Engines (LLMs)**: Modular wrappers that generate structured outputs (e.g., JSON, Pydantic) from unstructured inputs.
- **Evaluation Pipelines**: Assess model outputs using classical metrics (e.g., F1, accuracy) and LLM-based comparisons.
- **Schema Assistant**: Conversational design and validation of Pydantic schemas.
- **Prompt & Parameter Tuning**: Human-in-the-loop and automated loops to iteratively improve prompt templates and generation parameters.
- **Explainability Tools**: Highlight source text spans and assign confidence scores to extractions.
- **Rich Ingestion Utilities**: Handle formats like FHIR, PDF, audio, Excel, HTML, and more.
- **Local & Remote Compatibility**: Supports local (e.g., Ollama) and remote (e.g., OpenAI, Fireworks) models.
- **Streamlit Apps**: Enable interactive evaluation, annotation, and dataset creation workflows.

---

## 🧱 Architecture

Each module in ExtraCTOps is:

- **Isolated** in its own Python virtual environment if potential dependency conflicts exist (e.g., in `generators/`).
- **Directly imported** for low-level utilities shared across modules (e.g., file handling).
- **API-exposed** for high-level modules (e.g., `evaluators/`, `pseudonymizer/`) using lightweight FastAPI servers.
- **Designed** for both batch and streaming use cases.
- **Powered** by interchangeable LLM/VLM providers (OpenAI, Ollama, Fireworks, etc.).

---

## Structure of project directory

```
ExtraCTOps/
│
├── generators/             # All generation engines and wrappers
│   └── local_pseudonymizer/ # A local LLM to pseudonymize clinical text
├── evaluators/             # Evaluation logic and LLM-based assessors
│   └── clustering/         # Embedding-based sampling & stratification
├── apps/                   # Streamlit frontends
├── pydantic_assistant/     # Conversational schema assistant
├── report_performance/     # Calculate perfromance and generate reports
├── tuning_loop/            # Parameter sweep and optimization logic
├── prompt_loop/            # Prompt-tuning workflows
├── explainers/             # Highlighting & certainty scorers
├── utils/                  # Common tools
│   ├── document_handler/   # Ingestion of PDF/HTML/Docx/XLSX/CSV/JSON
│   ├── pro_pdf_handler/    # Advanced PDF reader with OCR+LLM+VLM
│   ├── FHIR_handler/       # Ingestion of FHIR
│   ├── Voice_handler/      # Ingestion of Voice
│   └── ExtraCTOps_loops/   # Loading a batch of tests, looping over them, and returning results
│ 
├── third_party_licenses/   
│
├── the_pydantics/          # The Pydantic schemas for use
│
├── the_venvs/              # The venvs for each module
│                
├── the_example_notebooks/  # Example notebooks to run modules or workflows 
└── thirdparty_use_case/    # Example notebooks incorporating external tools
```


## Example of .py file

Each .py should be self-contained and include a description of the function at the top, the libraries it uses, and the path to the virtual environment if it requires a separate one. Then the code should be organized into sections for imports, functions, and the main function at the end. And then example code for usage in terminal and in a nootebook. 

Codes should have compatibility for async run. 

Each file should start with a section for description, engine, identity, changelog, and to-do list. Then imports, then logging, then functions, then the main function, and finally example usage in terminal (with argparse) and notebook.

```python
"""
# Description
    [Hint: One paragraph describing the function and its purpose. This should be a high-level overview of what the function does, its inputs, and outputs.]

    - Arguments:
        - input_arg1 (type): Description of arg1
        - input_arg2 (type): Description of arg2

    - Enviroment Arguments:
        - env_arg1 (type): Description of env_arg1

    - Returns
        - output_arg1 (type_): Description of the return value


# Engine:
    - Serve (utils/data/main-function/sub-function): 
    - Served by (API/Direct/Subprocess):
    - Path to venv, if require separate venv: 
    - libraries to import: [] 

# Identity
    - Last Status (future/in-progress/complete/published):
    - Publish Date: 
    - Version: 
    - License: MIT
    - Author: Seyed Amir Ahmad Safavi-Naini Safavi-Naini, sdamirsa@gmail.com (the nominee for the longest name ever)
    - Source: https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps

# Changelog
    - 2000-00-00: version 0.0

# To-do: 
    - []
"""


###################
####  imports  ####
###################
import subprocess
import sys

libraries = [
    "numpy",
    "pandas",
    "scikit-learn"
]

for lib in libraries:
    try:
        __import__(lib.replace("-", "_"))  # crude guess for pip-import mismatch
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

###################
####  logging  ####
###################
from utils import the_logger # our logger

# Example: Log an informational message
the_logger.info("This is an informational log message.")

# Example: Log a warning message
the_logger.warning("This is a warning log message.")

# Example: Log an error message
the_logger.error("This is an error log message.")

# =====================================================
# 0) Enviroment Arguments
# =====================================================



# =====================================================
# 1) Functions for task 1 
# =====================================================


# =====================================================
# 2) Functions for task 2
# =====================================================



# =====================================================
# 2) Main Function
# =====================================================



###################################
####  Example use in terminal  ####
###################################
"""
python path_to_script.py --arg1 value1 --arg2 value2
"""


###################################
####  Example use in notebook  ####
###################################
"""
from ExtractOps import module_name
module_name.function_name(
    arg1 = value1,
    arg2 = value2
)
"""
```
