# Data2Pydantic_Map

A Streamlit application for mapping and transforming EHR/database data to user-defined Pydantic models. Supports mapping template generation, data transformation, and LLM-assisted auto-mapping.

---

## Features

- **Mapping Template Generation:** Generate a customizable mapping template (Excel/CSV) from your Pydantic model.
- **Data Transformation:** Map and transform your EHR/database data (CSV/Excel, wide or long format) to Pydantic-compliant JSON and flattened Excel.
- **LLM Auto-Mapping:** Use OpenAI or Azure OpenAI to automatically generate mapping templates from your data and Pydantic model.
- **Sample Files Provided:** Example Pydantic model, data, and mapping files included.

---

## Quick Start

### 1. **Clone the Repository**

```sh
git clone https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps.git
cd awesome_extraction_with_LLM_ExtraCTOps/apps/Data2Pydantic_Map
```

### 2. **Set Up a Python Environment**

- **Recommended:** Use a virtual environment (venv, conda, or mamba).
- **Python version:** 3.9–3.12

#### On macOS/Linux/Minerva

```sh
python3 -m venv venv
source venv/bin/activate
```

#### On Windows

```sh
python -m venv venv
venv\Scripts\activate
```

### 3. **Install Requirements**

```sh
pip install -r requirements.txt
```

If you plan to use LLM auto-mapping, ensure you have an OpenAI or Azure OpenAI API key.

---

## Running the App

```sh
streamlit run app.py
```

- The app will open in your browser (usually at http://localhost:8501).
- If running on a remote server (e.g., Minerva), use `--server.port` and `--server.address` options and set up port forwarding as needed.

---

## App Workflow

### **Section 1: Generate Mapping Template**

1. **Upload your Pydantic model** (`.py` file) or paste the code.
2. **Select the model** (if multiple are found).
3. **Click "Generate Template"** to download a mapping template (Excel).

### **Section 2: Transform Data Using Mapping**

1. **Upload your mapping file** (Excel or CSV, e.g., sample_mapping_v2.csv).
2. **Upload your Pydantic model** (same as above).
3. **Upload your data file** (CSV or Excel, e.g., sample_database_v2.csv).
4. **Select data format:**
   - **Wide:** Each column is a variable.
   - **Long:** Each row is an observation; specify Patient ID, Observation Name, and Observation Value columns.
5. **Click "Transform Data":**
   - Download transformed data as JSON (Pydantic-compliant).
   - Download a flattened Excel file (nested fields separated by `::`).
   - Download a mapping monitor Excel file (shows mapping validation).

### **Section 3: Smart (LLM) Mapping**

1. **Upload your data file** (CSV/Excel).
2. **Select data format** and specify columns if "Long".
3. **Upload or paste your Pydantic model**.
4. **Select LLM provider** (OpenAI or Azure OpenAI), enter your API key, and model details.
5. **Click "Suggest Mapping with LLM":**
   - The app summarizes your data and model, sends them to the LLM, and generates a mapping file.
   - Download the mapping as CSV or Excel.

---

## Sample Files

- sample_echo_model_v2.py — Example Pydantic model.
- sample_database_v2.csv — Example EHR data (long format).
- sample_mapping_v2.csv — Example mapping file.

Use these to test the app or as templates for your own data.

---

## Requirements

See requirements.txt. Key packages:

- `streamlit`
- `pandas`
- `openpyxl`
- `xlsxwriter`
- `pydantic`
- `openai` (for LLM mapping)
- `PyPDF2`, `docx2txt` (for future/unstructured data support)

**If you encounter missing package errors, install them with:**

```sh
pip install -r requirements.txt
```

---

## Troubleshooting

- **Malformed CSVs:** If your CSV has inconsistent columns, the app will skip problematic rows and warn you.
- **LLM API errors:** Ensure your API key is correct and you have access to the selected model.
- **Excel file errors:** Make sure your mapping/data files are not open in Excel while uploading.
- **Port issues on Minerva/server:** Use `streamlit run app.py --server.port 8502 --server.address 0.0.0.0` and set up SSH port forwarding.

---

## Advanced Usage

- **Custom Pydantic Models:** You can use any valid Pydantic model. Nested models and enums are supported.
- **Custom Mapping Logic:** Edit the mapping template to define complex mappings, multi-value fields, or custom evaluation methods.
- **LLM Prompt Engineering:** The LLM prompt is customizable in the code for advanced users.

---

## FAQ

**Q: Can I use this on Windows/Mac/Linux/Minerva?**  
A: Yes! The app is pure Python and Streamlit. Just follow the setup instructions for your platform.

**Q: What if my data is in a different format?**  
A: The app supports both wide and long formats. For other formats, convert to CSV/Excel first.

**Q: How do I get an OpenAI API key?**  
A: Sign up at https://platform.openai.com/ and create an API key.

**Q: Is it safe to use my patient data with the smart mapping feature?**
A: The app does not store or log your data. However, be cautious with sensitive data and consider anonymizing it before use. The LLM may log your data for training purposes, so check OpenAI's privacy policy. If you can use Mount Sinai's Azure OpenAI, it is recommended as it is HIPAA-compliant.

**Q: What if I don't have an API key?**
A: You can still use the app for manual mapping and transformation without LLM features. For LLM features, you need an API key. You may want to use ollama and provide the base_url (haven't tested yet).

**Q: How do I get an Azure OpenAI API key?**
A: Ask a PI at your team (at Mount Sinai). 

---

## Citation & License

- **License:** MIT
- **Author:** Seyed Amir Ahmad Safavi-Naini Safavi-Naini, sdamirsa@gmail.com
- **Source:** https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps

---

## Changelog

- **2025-05-21:** Initial public release.

---

**For questions or issues, please open an issue on GitHub or contact the author (sdamirsa@gmail.com or Amir Safavi on Slack).**

