# ExrtraCTOps-Apps-pydantic-based-manual-extraction-app
A powerful manual data extraction application built with Streamlit that allows users to extract structured information from unstructured text using Pydantic models. 


# Enhanced Pydantic Extraction App 📋

A powerful **manual data extraction application** built with Streamlit that allows users to extract structured information from unstructured text using Pydantic models. This app is part of the **ExtraCTOps project** - a comprehensive toolkit for improving LLM performance through structured data extraction and evaluation.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.0%2B-green)](https://pydantic.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Part of ExtraCTOps**: This application is a component of the larger [ExtraCTOps project](https://github.com/Sdamirsa/ExrtraCTOps-Apps-pydantic-based-manual-extraction-app) - a suite of tools designed to enhance LLM performance through structured extraction and evaluation workflows.

## 🎯 What Can You Do With This App?

<details>
<summary><strong>📊 Manual Data Extraction & Gold Standard Creation</strong></summary>

- **Create High-Quality Training Data**: Extract structured information from unstructured text to create gold standard datasets
- **Standardize Information**: Convert messy documents into consistent, structured formats using Pydantic models
- **Quality Control**: Manual review process ensures accuracy and completeness of extracted data
- **Multi-format Support**: Process various document types (PDF, DOCX, CSV, JSON) into unified structures

</details>

<details>
<summary><strong>🤖 LLM Performance Evaluation & Improvement</strong></summary>

- **Benchmark Creation**: Generate reference datasets for evaluating LLM extraction performance
- **Prompt Engineering**: Use manually extracted data to test and refine LLM prompts within the ExtraCTOps ecosystem
- **Parameter Optimization**: Compare manual extractions with LLM outputs to optimize model parameters
- **Performance Metrics**: Establish baselines for measuring LLM accuracy on specific extraction tasks
- **Training Data Generation**: Create supervised learning datasets for fine-tuning specialized models

</details>

<details>
<summary><strong>🔬 Research & Analysis Applications</strong></summary>

- **Clinical Data Extraction**: Extract patient information from medical records for research studies
- **Legal Document Processing**: Structure contracts, agreements, and legal documents for analysis
- **Academic Research**: Convert research papers and documents into structured data for meta-analyses
- **Business Intelligence**: Extract key information from reports, emails, and business documents
- **Content Analysis**: Systematically analyze text content for patterns and insights

</details>

<details>
<summary><strong>🔄 Integration with ExtraCTOps Workflow</strong></summary>

1. **Extract**: Use this app to manually extract information in standardized formats
2. **Export**: Generate clean, structured datasets with your extractions
3. **Import to ExtraCTOps**: Use your manual extractions as reference data in the main ExtraCTOps platform
4. **Compare & Improve**: Test LLM performance against your manual extractions
5. **Iterate**: Refine prompts and parameters based on comparison results
6. **Scale**: Apply improved LLM configurations to larger datasets

</details>

## 🚀 Setup Guide

<details>
<summary><strong>💻 For Non-Developers (Easy Setup)</strong></summary>

### Windows Users

1. **Install Python**
   - Go to [python.org](https://python.org) and download Python 3.8 or newer
   - During installation, **check "Add Python to PATH"**
   - Restart your computer after installation

2. **Download the App**
   - Go to the [GitHub repository](https://github.com/Sdamirsa/ExrtraCTOps-Apps-pydantic-based-manual-extraction-app)
   - Click the green "Code" button → "Download ZIP"
   - Extract the ZIP file to your Desktop

3. **Install Dependencies**
   - Open Command Prompt (press `Win + R`, type `cmd`, press Enter)
   - Navigate to the app folder: `cd Desktop\enhanced-pydantic-extraction-app-main`
   - Install requirements: `pip install -r requirements.txt`

4. **Run the App**
   - In the same Command Prompt, type: `streamlit run app.py`
   - Your browser will automatically open the app

### Mac Users

1. **Install Python**
   - Open Terminal (press `Cmd + Space`, type "Terminal")
   - Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
   - Install Python: `brew install python`

2. **Download the App**
   - Go to the [GitHub repository](https://github.com/Sdamirsa/ExrtraCTOps-Apps-pydantic-based-manual-extraction-app)
   - Click the green "Code" button → "Download ZIP"
   - Extract to your Downloads folder

3. **Install and Run**
   - Open Terminal and navigate: `cd Downloads/enhanced-pydantic-extraction-app-main`
   - Install dependencies: `pip3 install -r requirements.txt`
   - Run the app: `streamlit run app.py`

### Linux Users

1. **Install Python** (if not already installed)
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```

2. **Download and Setup**
   ```bash
   git clone https://github.com/Sdamirsa/ExrtraCTOps-Apps-pydantic-based-manual-extraction-app.git
   cd enhanced-pydantic-extraction-app
   pip3 install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

### 🌐 Alternative: Online Setup (Streamlit Cloud)

1. **Fork the Repository**
   - Go to the [GitHub repository](https://github.com/Sdamirsa/ExrtraCTOps-Apps-pydantic-based-manual-extraction-app)
   - Click "Fork" to create your own copy

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app" and select your forked repository
   - The app will deploy automatically and be accessible via web browser

### ⚠️ Troubleshooting

**Python not found?**
- Windows: Reinstall Python and ensure "Add to PATH" is checked
- Mac: Try `python3` instead of `python`
- Linux: Use `python3` and `pip3`

**Permission errors?**
- Windows: Run Command Prompt as Administrator
- Mac/Linux: Add `sudo` before commands if needed

**App won't start?**
- Check if port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

</details>

<details>
<summary><strong>👨‍💻 For Developers (Advanced Setup)</strong></summary>

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Sdamirsa/ExrtraCTOps-Apps-pydantic-based-manual-extraction-app.git
cd enhanced-pydantic-extraction-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
streamlit run app.py --server.runOnSave true

# Run tests
python -m pytest tests/
```

### Docker Setup

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

</details>

## 📖 Usage Guide

<details>
<summary><strong>🔧 Basic Setup & Configuration</strong></summary>

### 1. Setup Your Pydantic Model

Create or paste your Pydantic model in the sidebar:

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str = Field(description="Task title")
    description: Optional[str] = Field(description="Detailed description")
    priority: Priority = Field(description="Task priority level")
    completed: bool = Field(default=False, description="Completion status")
    tags: List[str] = Field(default=[], description="Task tags")
```

### 2. Upload Your Data

Support for multiple formats:
- **CSV/Excel**: Structured data with ID and text columns
- **JSON**: Structured or unstructured JSON data
- **Text Files**: Plain text, DOCX, or PDF documents

### 3. Configure Columns

For structured data:
- **ID Column**: Unique identifier (optional)
- **Text Column**: Content to extract from

</details>

<details>
<summary><strong>📝 Extraction Workflow</strong></summary>

### 4. Initialize Extraction

Click "🚀 Initialize Extraction Session" to:
- Prepare all rows for extraction
- Set up form state management
- Initialize progress tracking

### 5. Extract Data

- Navigate through rows using Previous/Next buttons
- Fill out the generated form fields
- Save extractions with the "💾 Save Extraction" button
- Track progress with visual indicators

### 6. Export Results

Choose from multiple export options:
- **Session Export**: Complete session for resuming later
- **CSV Export**: Tabular data for analysis
- **JSON Export**: Structured data for processing

</details>

<details>
<summary><strong>🔄 Integration with ExtraCTOps</strong></summary>

### Using Exports in ExtraCTOps

1. **Export your manual extractions** as JSON or CSV
2. **Import into ExtraCTOps** as reference/gold standard data
3. **Configure LLM prompts** for the same extraction task
4. **Run LLM extractions** on the same source documents
5. **Compare results** to identify areas for improvement
6. **Iterate and improve** LLM performance

### Best Practices for LLM Training

- **Consistent Schema**: Use the same Pydantic model across manual and automated extraction
- **Quality Control**: Review manual extractions before using as training data
- **Balanced Dataset**: Ensure diverse examples in your manual extraction set
- **Clear Instructions**: Document extraction guidelines for consistency

</details>

## ✨ Features

<details>
<summary><strong>🎯 Core Functionality</strong></summary>

- **Dynamic Pydantic Model Support**: Load any Pydantic model from code or .py files
- **Multi-format Data Support**: Excel, CSV, JSON, TXT, DOCX, PDF files
- **Interactive Extraction Interface**: User-friendly forms for manual data extraction
- **Session Persistence**: Save and resume extraction sessions
- **Progress Tracking**: Visual progress indicators and review status
- **Export Options**: JSON, CSV, and complete session exports

</details>

<details>
<summary><strong>🛡️ Robust State Management</strong></summary>

- **Thread-safe Operations**: Atomic state transactions
- **State Validation**: Automatic consistency checks and recovery
- **Form Synchronization**: Seamless widget-to-data synchronization
- **Navigation Stability**: Reliable row-by-row navigation
- **Error Recovery**: Graceful handling of state corruption

</details>

<details>
<summary><strong>🎨 Enhanced UI/UX</strong></summary>

- **Colored Containers**: Visual hierarchy for nested structures
- **Field Completion Indicators**: ✅ marks for completed fields
- **Responsive Design**: Adjustable column heights and layouts
- **Intuitive Navigation**: Previous/Next buttons with jump-to functionality
- **Progress Visualization**: Review status tracking and metrics

</details>

## 📦 Dependencies

<details>
<summary><strong>📋 Required Packages</strong></summary>

```txt
streamlit>=1.28.0
pandas>=1.5.0
pydantic>=2.0.0
openpyxl>=3.0.0
python-docx>=0.8.11
PyPDF2>=3.0.0
python-dateutil>=2.8.0
```

</details>

## 🎨 Field Type Support

<details>
<summary><strong>📊 Supported Pydantic Types</strong></summary>

| Pydantic Type | UI Component | Features |
|---------------|--------------|----------|
| `str` | Text Input/Area | Auto-detection for long text |
| `int`/`float` | Number Input | Optional/Required modes |
| `bool` | Radio Buttons | True/False/None options |
| `Enum` | Radio/Selectbox | Dynamic enum value loading |
| `Literal` | Radio/Selectbox | Literal value selection |
| `List[T]` | Dynamic List | Add/remove items interface |
| `BaseModel` | Nested Form | Expandable nested structures |
| `Optional[T]` | Mode Selection | None option for all types |

</details>

## 🧪 Example Use Cases

<details>
<summary><strong>🏥 Medical Record Extraction</strong></summary>

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class PatientStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISCHARGED = "discharged"

class Medication(BaseModel):
    name: str = Field(description="Medication name")
    dosage: str = Field(description="Dosage information")
    frequency: str = Field(description="How often to take")

class Patient(BaseModel):
    patient_id: str = Field(description="Unique patient identifier")
    name: str = Field(description="Patient full name")
    age: Optional[int] = Field(description="Patient age")
    status: PatientStatus = Field(description="Current patient status")
    medications: List[Medication] = Field(default=[], description="Current medications")
    notes: Optional[str] = Field(description="Additional notes")
    last_visit: Optional[str] = Field(description="Last visit date")
```

**Sample Data:**
```csv
patient_id,name,medical_record
P001,John Doe,"Patient presents with hypertension. Prescribed Lisinopril 10mg daily. Next visit in 3 months."
P002,Jane Smith,"Routine checkup. Blood pressure normal. Continue current medication regimen."
```

</details>

<details>
<summary><strong>⚖️ Legal Document Processing</strong></summary>

```python
class ContractType(Enum):
    SERVICE = "service"
    EMPLOYMENT = "employment"
    NDA = "nda"
    PURCHASE = "purchase"

class Contract(BaseModel):
    contract_id: str = Field(description="Contract identifier")
    contract_type: ContractType = Field(description="Type of contract")
    parties: List[str] = Field(description="Contracting parties")
    effective_date: Optional[str] = Field(description="Contract effective date")
    expiration_date: Optional[str] = Field(description="Contract expiration date")
    key_terms: List[str] = Field(default=[], description="Key contract terms")
    value: Optional[float] = Field(description="Contract value if applicable")
```

</details>

<details>
<summary><strong>📰 News Article Analysis</strong></summary>

```python
class Sentiment(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class NewsArticle(BaseModel):
    headline: str = Field(description="Article headline")
    author: Optional[str] = Field(description="Article author")
    publication_date: Optional[str] = Field(description="Publication date")
    category: str = Field(description="Article category")
    sentiment: Sentiment = Field(description="Overall sentiment")
    key_entities: List[str] = Field(default=[], description="People, organizations, locations mentioned")
    summary: str = Field(description="Brief article summary")
```

</details>

## 🔧 Advanced Configuration

<details>
<summary><strong>⚙️ Environment Variables & Settings</strong></summary>

```python
# Color palette for UI elements
COLOR_PALETTE = [
    "#FFCDD2", "#C8E6C9", "#BBDEFB", "#FFE0B2",
    "#D1C4E9", "#B2DFDB", "#F8BBD0", "#FFF9C4"
]

# Brightness adjustment for nested fields
BRIGHTER_COLOR_RATE = 0.22

# Fields treated as long text (uses text_area)
LONG_TEXT_FIELD_LIST = ["description", "comment", "notes", "information", "text"]

# Separator for flattening nested structures
flatten_for_export_SEPARATOR = "::"
```

### UI Configuration

- **Column Height**: Adjustable extraction interface height (300-1500px)
- **Color Coding**: Automatic color coding for review status
- **Progress Tracking**: Visual indicators for completion status

</details>

## 🐛 Troubleshooting

<details>
<summary><strong>🚨 Common Issues & Solutions</strong></summary>

### Navigation Issues

**Q: Navigation buttons are not working**
A: This usually indicates a state management issue. Try refreshing the page or resetting the session.

**Q: Form fields are not saving**
A: Ensure you click "Save Extraction" after filling out fields. Check that your Pydantic model is valid.

### Import/Export Issues

**Q: Session import fails**
A: Verify the JSON file is a valid session export from this app. Check for file corruption.

**Q: CSV export is missing data**
A: Ensure you've saved all extractions. Check that nested fields are properly flattened.

### Model Issues

**Q: Application crashes on model parsing**
A: Check your Pydantic model syntax. Ensure all required imports are included in the code.

**Q: Fields not rendering correctly**
A: Verify your field types are supported. Check for circular references in nested models.

### Performance Issues

**Q: App is running slowly**
A: Try reducing column height, limit nested object depth, or process smaller batches of data.

</details>

<details>
<summary><strong>🔧 Debug Mode & Performance Tips</strong></summary>

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

- Use shorter text fields when possible
- Limit the number of nested objects
- Export sessions regularly to prevent data loss
- Use the progress indicators to track completion
- Process data in smaller batches for large datasets

</details>

## 🏗️ Project Structure

<details>
<summary><strong>📁 Repository Organization</strong></summary>

```
enhanced-pydantic-extraction-app/
│
├── app.py                      # Main application file
├── requirements.txt            # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
```

</details>

## 🤝 Contributing

<details>
<summary><strong>🔄 How to Contribute</strong></summary>

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/enhanced-pydantic-extraction-app.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run with development mode
streamlit run app.py --server.runOnSave true
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Add docstrings for all functions
- Maintain test coverage above 80%

</details>

## 🔗 Related Projects

<details>
<summary><strong>🌟 ExtraCTOps Ecosystem</strong></summary>

- **[ExtraCTOps Main Platform](https://github.com/Sdamirsa/ExrtraCTOps-Apps-pydantic-based-manual-extraction-app)**: Complete LLM evaluation and improvement toolkit
- **Manual Extraction App** (This Repository): Create gold standard datasets
- **LLM Evaluation Tools**: Compare and benchmark extraction performance
- **Prompt Engineering Suite**: Optimize prompts based on manual extractions

</details>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ExtraCTOps Project**: Part of the comprehensive LLM improvement toolkit
- **Streamlit** for the amazing web app framework
- **Pydantic** for data validation and parsing
- **Pandas** for data manipulation capabilities
- The open-source community for inspiration and support

## 📧 Contact

- **Author**: Seyed Amir Ahmad Safavi-Naini
- **Email**: sdamirsa@gmail.com
- **ExtraCTOps Project**: https://github.com/Sdamirsa/ExrtraCTOps-Apps-pydantic-based-manual-extraction-app
- **This Repository**: https://github.com/Sdamirsa/enhanced-pydantic-extraction-app

## 🔄 Version History

- **v0.2** (Current) - Enhanced state management, session persistence, improved navigation, ExtraCTOps integration
- **v0.1** - Initial release with basic extraction functionality

---

⭐ **Star this repository** if you find it helpful!

🐛 **Report bugs** by opening an issue

💡 **Suggest features** through discussions or issues

🔗 **Check out ExtraCTOps** for the complete LLM improvement workflow

Happy extracting! 🎉