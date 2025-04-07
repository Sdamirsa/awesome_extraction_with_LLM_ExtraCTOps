# 🩺 awesome LLM extraction + ExtraCTOps

**ExtraCTOps** is a modular and extensible framework for **extraction, evaluation, and improvement of structured data** from unstructured text — with examples on **clinical text processing**. It leverages the power of **Large Language Models (LLMs)**, **Vision-Language Models (VLMs)** (watchlist), and **Pydantic** to bring intelligence, transparency, and precision to every step of the information extraction lifecycle.

This repo contains:
- **"ExtraCTOps"** Package: We build comprehensive solutions for extracting structured output from unstructured data (text, voice, image). Each module is self-contained, allowing users to pick and choose components that fit their needs. This modularity facilitates easy integration with existing systems.

- **"awesome extraction with LLM"** Watchlist: We plan to maintain an ongoing summary of advances in structuring data using LLMs and emerging VLMs, highlighting papers, LLMs, and modules that can be integrated into ExtraCTOps or any other pipeline.

---

## 📚 awesome extraction with LLM

The repo [imaurer/awesome-llm-json](https://github.com/imaurer/awesome-llm-json) is the best source for looking for anything related to json output. We plan to maintain an **ongoing summary** of advances in structuring data using **LLMs and emerging VLMs**, highlighting:

- **Papers**: Latest techniques in text and multimodal extraction and performance comparisons.
- **LLMs**: Models for extraction and structured output generation from our experience (unofficial) and papers.
- **Modules**: Performance differences between different modules (JSONify, Hermes, LangChain, Pydantic, etc.), primarily focused on clinical text.

This ensures ExtraCTOps stays current with evolving research, providing recommendations and best practices across the ecosystem.

---

## 📂 ExtraCTOps

This is the overview of modules and package structure. 

```bash
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
│   └── ExtraCTOps_loops/   # Loading a batch, loop, and returning results
│   └── ExtraCTOps_DynamicMemory_loops/ # Loading a batch, extract, add new values to pydantic enum, continue, return result
│   └── unify_string_extractions/ # Get the extraction and turn unstructured strings of a varibale into a unified labels (like what we have with enum)
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

<details>
<summary>Modules Description</summary>

### 1. 🧠 Generators

Engines to extract or transform data (often into Pydantic objects) from unstructured text, leveraging various LLM backends.
- **Core Features**:
    - LLM Integration with providers like OpenAI, Fireworks, RunPod, Ollama
    - Support for different generation strategies (LangChain, JSONify, Hermes-Function-Calling)
- **Submodules**:
    - **local_pseudonymizer/**: A specialized local LLM workflow for pseudonymizing clinical text. Helpful for HIPAA/GDPR compliance, ensuring no identifiable patient data is revealed.

### 2. ✅ Evaluators

Evaluate the quality of your generated (extracted or pseudonymized) data.
- **Core Evaluation Methods**:
    - LLM-based comparisons to ground truth
    - Rule-based metrics (F1-score, valid-option checks)
    - Custom domain metrics for clinical contexts
- **Submodules**:
    - **clustering/**: Clustering-based sampling for diverse, representative test sets. Groups text by semantic similarity (embedding-based) to ensure broad coverage of possible scenarios.

### 3. 🧪 Apps

A collection of Streamlit frontends for interactive usage and rapid prototyping:
- Manually evaluate extracted Pydantic outputs
- Compare model output vs. ground truth
- Annotate or correct results to build a labeled dataset

### 4. 🗣️ Pydantic Assistant

A conversational agent (default: GPT-4o or your chosen LLM) that helps define and refine Pydantic schemas:
- Suggests fields and data types based on user input
- Validates objects against the schema
- Asks clarifying questions to ensure completeness

Ideal for quickly iterating on how your extracted data should be structured.

### 5. 📊 Report Performance
Generates reports on the performance of your extraction or pseudonymization tasks:
- Compares model outputs against ground truth
- Provides visualizations of performance metrics
    - F1 score (including the scenario that there is no information in the text)
    - Accuracy
    - Edge case analysis for each variable
- Generates summary reports for easy sharing and review
- Can be integrated with the Evaluators module for a seamless workflow


### 6. 🔁 Tuning Loop

Automated or semi-automated parameter sweep logic to find the best performing model settings:
- Function–LLM–Parameter Tuning: Vary temperature, top-p, or top-k across multiple LLM providers
- Bayesian optimization to minimize trial-and-error overhead
- Logs results for comparison and potential regression checks

### 7. 🔄 Prompt Loop

A human-in-the-loop workflow to iteratively improve prompts, schema descriptions, and examples:
- Tracks performance gains or regressions with each iteration
- Reduces guesswork in prompt engineering
- Can integrate with the Tuning Loop for a more holistic optimization approach

### 8. 🩻 Explainers

Provides interpretability for LLM outputs:
- Assigns confidence or uncertainty scores to each extracted field
- Highlights relevant text spans for improved traceability
- Ideal for auditing correctness in high-stakes (e.g., clinical) environments

### 9. 🛠️ Utils

A collection of utility modules offering ingestion, handling, and batch-processing capabilities:
- **document_handler/**
    - Ingests PDFs, HTML, Docx, Excel, CSV, and JSON files
    - Basic text extraction workflow for downstream modules
- **pro_pdf_handler/**
    - Advanced PDF reading with OCR (for scanned docs) + LLM/VLM enhancements
    - Useful for complex or image-heavy clinical docs
- **FHIR_handler/**
    - Ingestion logic specific to FHIR (Fast Healthcare Interoperability Resources) data
    - Facilitates direct transformations between FHIR objects and Pydantic schemas
- **Voice_handler/**
    - Processes audio or voice data (e.g., transcripts, TTS or speech-to-text flows)
    - Enables ingestion of spoken notes or dictations
- **ExtraCTOps_loops/**
    - Utilities to batch-load data, run extraction/pseudonymization across a dataset, and aggregate results
    - Supports logging intermediate states in JSON or .log files for inspection/retry

### Additional Folders
- **third_party_licenses/**: Licenses for any external modules or code used within ExtraCTOps.
- **the_pydantics/**: Stores base or example Pydantic schemas for various domain uses.
- **the_example_notebooks/**: Example Jupyter notebooks demonstrating typical workflows or usage patterns.
- **thirdparty_use_case/**: Additional notebooks focused on integrating ExtraCTOps with external modules or real-world third-party applications.

### Putting It All Together
1. Select a Module (e.g., Generators) to perform text extraction or pseudonymization.
2. Evaluate Outputs using the Evaluators folder's metrics and sampling strategies.
3. Refine your schemas and prompt strategies with the Pydantic Assistant, Tuning Loop, or Prompt Loop.
4. Explain & Validate results with Explainers and Streamlit Apps.
5. Utilize the various Utils submodules to handle different file types or data ingestion needs.

Each module or submodule is independently runnable but can be integrated for a complete end-to-end solution.

</details>

---

## System Design & Description

<details>
<summary>ExtraCTOps – System Design & Package Description</summary>

### 1. Project Requirements & Goals

**Functional Requirements**
- **Problem**: A large portion of data is unstructured (e.g., conversations, PDFs, clinical notes), making query and retrieval difficult.
- **Solution**: Provide an environment to easily test and compare different LLM/VLM setups for extracting structured data (JSON/Pydantic).
- **Challenges**:
    - Environmental setup & dependency isolation
    - Unclear cost–efficiency trade-offs (local vs. API-based, large vs. small models)
    - Defining & refining target structures
- **Users**: Primarily researchers/developers (academic and industry), clinicians, or anyone needing robust extraction from unstructured text. Assumed hardware: ~32GB RAM and mid-high laptop. GPU is optional.

**Non-Functional Requirements**
- **Performance**: Should support asynchronous operations; user expects 3–7 days of testing various setups to find the best setup and the trade-off on a validation set (100-300 instances).
- **Security**: Support local (Ollama) and API-based (OpenAI, Fireworks) generative models, mindful of potential clinical/HIPAA contexts.

### 2. Data Flows & Pipelines
- **Sources**: Excel, JSON, PDF (including complex PDFs), HTML, Docx, images, voice.
- **Ingestion**: Primarily async; batch or streaming, but must remain easy to use.
- **Volume**: Up to ~100K documents, with clustering-based sampling for test sets.
- **Storage**: Inputs and outputs are JSON-based for reusability, plus logging to .log.

### 3. System Architecture & Module Boundaries
- **Structure**: Highly modular, with each module in its own Python virtual environment for dependency isolation.
- **Communication**: Modules expose lightweight HTTP APIs (JSON requests/responses) only for the high-level functions. A high-level orchestrator calls these module APIs rather than importing them directly.
- **Benefits**:
    - Loose coupling and independent scalability
    - Clear separation of concerns
    - Flexibility in swapping or upgrading components

### 4. Technology Stack & Dependencies
- **Languages & Frameworks**:
    - Python 3.12.7, FastAPI (0.115), Pydantic (v2), LangChain (v0.3), OpenAI (1.7), Ollama (0.4.7)
- **Versioning**:
    - Pin versions to avoid breaking changes
    - Each module has its own venv, tested and stored together for compatibility
- **Services & Integrations**:
    - LLM providers: OpenAI, Fireworks, RunPod
    - Local LLM hosting: Ollama

### 5. Evaluation & Quality Assurance
- **Testing**:
    - Unit tests (each module)
    - Integration tests (full pipeline)
    - Regression tests (ensure no breakage from updates) --> no plan for this yet
- **Metrics for extraction evaluation**:
    - F1 for extraction, domain-specific clinical metrics
    - Accuracy for extraction, domain-specific clinical metrics
    - Edge cases analysis and explainability modules
    - Report speed, resource usage, cost analysis

### 6. Security & Compliance
- **Privacy**: Potential HIPAA/GDPR if clinical data is involved; local or on-premise options via Ollama. Local LLM-based pseudonymization for sensitive data, before sending to external APIs.
- **Licensing**: MIT license with some Apache-2.0 modules. Keep third-party licenses in third_party_licenses/.

### 7. Deployment & Operational Concerns
- **Environment**: Can run LLMs locally or call external APIs for LLM/VLM. The conversational agents are mainly based on GPT-4o as we assumed the description of variables to extract are not confidential.
- **Infrastructure**: Standard Python environments imported directly or served via FastAPI; GPU optional but can be leveraged if available.
- **Logging & Monitoring**: Real-time logs and error tracking; failures stored as .log entries and in the JSON output.

### 8. Performance & Scalability Planning
- **Load Testing**:
    - Pilot on ~100–300 samples to find best setup, then scale to 100K.
- **Caching**:
    - If a step completes successfully, skip re-processing.
    - Error states captured with reasons for failure.
- **Optimization**:
    - Batch processing where possible
    - Hyperparameters stored in a file; potential UI for easy config

### 9. Documentation & Onboarding
- **Documentation**:
    - Each .py file has an intro + usage examples (both for terminal use and notebook use).
    - Sample notebooks in the_examples/.
- **User Guides**:
    - Outline how to install modules, create venvs, and run scripts.
    - Potential API references if a library or SDK is provided.

### 10. Long-Term Maintenance & Community
- **Upkeep**: Maintained by the main developer at present (no formal open-source release schedule yet).
- **Releases**: Modules tagged by push date until a stable, minimal-viable release is ready.
- **Collaboration**:
    - Email: sdamirsa@gmail.com for contributions/questions
    - A to-do list will track roadmap items and tasks

</details>

---

## 🚧 Roadmap and To-Do

<details>
<summary>v0.01</summary>

```bash
ExtraCTOps/
│
├── generators/             # All generation engines and wrappers
│   └── generator_Ollama.py [ ]
│   └── generator_Openai.py [ ]
│   └── generator_Firework.py [ ]
│   └── generator_Lang_Ollama.py [ ]
│   └── generator_jsonformer.py [ ]    
│   └── generator_dspy.py [ ]   
├── evaluators/             # Evaluation logic and LLM-based assessors
│   └── evaluator_Openai.py [ ]
├── apps/                   # Streamlit frontends
│   └── manual_evaluation.py [ ] # an app for manual evaluation of outputs
│   └── manual_extraction.py [X] # an app to help extracting ground truths
├── report_performance/     # Calculate perfromance and generate reports
│   └── calculate_f1.py [ ] 
├── utils/                  # Common tools
│   ├── document_handler/   # Ingestion of PDF/HTML/Docx/XLSX/CSV/JSON
│   │    └── Input_pdf.py [ ]
│   │    └── Input_docx.py [ ]
│   │    └── Input_excel.py [ ]
│   │    └── Input_csv.py [ ]
│   └── loops/   # Loading a batch of tests, looping over them, and returning results
│       └── Loop_for_generators.py [ ]
│       └── Loop_for_evaluators.py [ ]
│ 
├── the_example_notebooks/  # Example notebooks to run modules or workflows 
└── thirdparty_use_case/    # Example notebooks incorporating external tools
    └── Loop_for_generators.py [ ]
```

</details>

<details>
<summary>v0.02</summary>


</details>

<details>
<summary>v0.03</summary>


</details>


<details>
<summary>v0.1 (dream)</summary>

</details>



---

For inquiries or contributions, please email sdamirsa@gmail.com.
