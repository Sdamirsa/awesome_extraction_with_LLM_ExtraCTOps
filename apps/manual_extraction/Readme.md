# Pydantic Extraction Tool Documentation

Pydantic Extraction is a painless Streamlit application for structured data extraction from various document formats. It provides a flexible framework for defining extraction schemas using Pydantic models and extracting data into structured formats.

The UI renders nested Pydantic models (in a way that won't drive you crazy, with proper colors and containers), even complex ones with multiple levels of nesting, and allows you to extract data according to your Pydantic data types (dropdown for enums, checkbox for booleans, text box for numbers, etc.). It parses your Pydantic model and creates a form for you to fill out.

![manual_extraction app demo](<manual_extraction app demo.gif>)

## Use Cases

## Use Cases

- **Create Ground Truth Datasets**: Build high-quality training data for LLM-based extraction and structured output tasks
- **Structured Form Processing**: 
    - Create schema-based data entry forms for consistent information capture
    - Validate all data entries against predefined schemas
    - Transform unstructured information into structured, machine-readable formats
- **Data Extraction & Transformation**:
    - Extract specific data points from documents following a consistent schema
    - Convert free-text documents into standardized data structures
    - Maintain hierarchical relationships between extracted entities
- **Legacy Data Migration**: Transform legacy documents and unstructured archives into modern structured formats for analysis or integration
- **Compliance Documentation**: Extract and validate required fields from regulatory documents into auditable structured formats

## Getting Started

### Installation

<details>
<summary>macOS</summary>

```bash
# Clone the repository
git clone https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps.git
cd apps/manual_extraction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
</details>

<details>
<summary>Windows</summary>

```batch
:: Clone the repository
git clone https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps.git
cd apps/manual_extraction

:: Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt

:: Run the app
streamlit run app.py
```
</details>

<details>
<summary>Linux</summary>

```bash
# Clone the repository
git clone https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps.git
cd awesome_extraction_with_LLM_ExtraCTOps

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
</details>

### Required Dependencies

Make sure your `requirements.txt` file includes:

```
streamlit>=1.22.0
pandas>=1.5.3
pydantic>=2.0.0
docx2txt>=0.8
PyPDF2>=3.0.0
```

## Using the Application

### 1. Define Your Extraction Schema

The first step is to define a Pydantic model that represents the structure of the data you want to extract. Here's an example:

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: Optional[str] = Field(description="State or province")
    zip_code: str = Field(description="Postal code")

class Person(BaseModel):
    name: str = Field(description="Full name")
    age: Optional[int] = Field(description="Age in years")
    email: str = Field(description="Email address")
    address: Address = Field(description="Person's address")
    skills: List[str] = Field(description="List of skills")
```

Save this in a .py file or paste code in the "Pydantic Model Setup" section and click "Parse Pydantic Code".

### 2. Load Your Data Source

The application supports various data formats:
- Excel (.xlsx)
- CSV (.csv)
- JSON (.json)
- Text (.txt)
- Word documents (.docx)
- PDF files (.pdf)

Upload your source document in the "Data Source" section.

### 3. Extract Data

Once your schema and data are loaded:
1. Navigate through your data source using the navigation controls
2. Fill in the form fields based on the information in your source document
3. Click "Save Extraction" to save the current extraction
4. Navigate to the next item in your dataset

### 4. Review and Export

The "Review & Export" section at the bottom shows all extracted data in both table and JSON format. You can:
- Export all your data as JSON or CSV
- Download the complete session state to resume extraction later

## Extraction Output Structure

The tool exports data in a flattened structure using the `::` separator for nested fields. For example:

### Original Structure:
```json
{
  "name": "John Doe",
  "address": {
    "street": "123 Main St",
    "city": "New York"
  },
  "skills": ["Python", "Data Analysis"]
}
```

### Flattened Structure:
```json
{
  "name": "John Doe",
  "address::street": "123 Main St", 
  "address::city": "New York",
  "skills::0": "Python",
  "skills::1": "Data Analysis",
  "row_index": 0,
  "id": "row_1"
}
```

This flattened structure makes it easy to:
- View the data in tabular formats
- Import directly into spreadsheets
- Process with other data tools

The separator `::` is used to maintain the hierarchy while ensuring compatibility with most data processing tools.

## Advanced Features

### Resuming a Previous Session

1. Save your session using the "Download Complete Session State (JSON)" button
2. When you restart the application, select "Continue Previous" in the sidebar
3. Upload your saved session JSON file
4. Continue your extraction from where you left off

### Structured vs. Unstructured Data

The application handles both structured data (like CSV or Excel) and unstructured text data (like PDFs or text files):

- **Structured Data**: The app will show row-by-row navigation
- **Unstructured Data**: The entire text will be shown as a single source

## Troubleshooting

- **Model parsing errors**: Check that your Pydantic models are correctly defined
- **PDF extraction issues**: Some PDFs may not extract properly if they contain scanned images
- **Large files**: For very large files, consider splitting them into smaller chunks

---

For more information or support, please open an issue on the GitHub repository. 



```python
"""
Pydantic Extraction App

# Description
    [A manual extraction application that allows users to upload documents and extract information 
    with assistance from LLMs. Provides a user interface for document processing and extraction operations.]

    - Arguments:
        - Data (Excel, CSV, JSON, TXT, DOCX, PDF): The file having the unstructured text for review  .
        - Pydantic Model (.py) Or Code (string): The file having the pydantic model code for the extraction.
        - Previous Session (JSON): The file having the previous extraction data for review.

    - Enviroment Arguments:
        - COLOR_PALETTE (list): A list of hex color codes for the pydantic top level fields.
        - BRIGHTER_COLOR_RATE (float):  The rate of brightness increase for each nested field.
        - LONG_TEXT_FIELD_LIST (list): A list of field names that are considered long text fields.
        - flatten_for_export_SEPARATOR (str): The separator used for flattening nested structures for export.
 
    - Returns
        - Session (JSON): The file having the app memory (including Data and Pydantic Model and previous extractions). This is usable for saving and loading to continue the extraction.
        - Extractions (JSON)
        - Extractions (CSV)

# Engine:
    - Serve (utils/data/main-function/sub-function): main-function
    - Served by (API/Direct/Subprocess): Subprocess
    - Path to venv, if require separate venv: the_venvs/venv_streamlit
    - libraries to import: [pydantic,PyPDF2,docx2txt,pandas,openpyxl] 

# Identity
    - Last Status (future/in-progress/complete/published): published
    - Publish Date: 2025-04-07
    - Version: 0.1
    - License: MIT
    - Author: Seyed Amir Ahmad Safavi-Naini Safavi-Naini, sdamirsa@gmail.com (the nominee for the longest name ever)
    - Source: https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps

# Changelog
    - 2025-04-07: version 0.1

# To-do: 
    - []
"""
```