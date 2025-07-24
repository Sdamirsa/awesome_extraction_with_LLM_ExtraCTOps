# 📚 The Scripts — `the_scripts/`

This folder contains production-ready scripts and Jupyter notebooks for the ExtraCTOps project.

---

## 📁 Contents

### 📓 Notebooks
- **`echo_extraction_production.ipynb`** — Production notebook demonstrating real-world extraction using both Ollama and OpenAI generators with the EchoReport Pydantic model.

### 🔧 Helper Scripts
- **`kernel_setup_helper.py`** — Verification script to check that Jupyter kernels are properly configured for use in VS Code.

---

## 🚀 Getting Started

### 1. **Kernel Setup**

Before running notebooks, ensure your Jupyter kernels are properly configured:

```bash
# Run the kernel verification script
cd /path/to/awesome_extraction_with_LLM_ExtraCTOps
./the_venvs/venv_main/bin/python the_scripts/kernel_setup_helper.py
```

If the verification passes, you should see:
- ✅ Both `venv_main` and `venv_ollama` Python environments working
- ✅ ExtraCTOps kernels registered: `extractops_main` and `extractops_ollama`
- ✅ All key imports working in both environments

### 2. **Running the Production Notebook**

1. **Open in VS Code**: Open `echo_extraction_production.ipynb` in VS Code
2. **Select Kernel**: Click "Select Kernel" in the top right and choose:
   - `ExtraCTOps Main (Python 3.12)` — Uses venv_main (has both Ollama and OpenAI)
   - `ExtraCTOps Ollama (Python 3.12)` — Uses venv_ollama (specialized for Ollama)
3. **Run Cells**: Execute the notebook cells step by step

---

## 📓 Production Notebook Overview

The `echo_extraction_production.ipynb` notebook demonstrates:

### 🔄 **Complete Workflow**
1. **Data Loading** — Load CSV data from `the_pydantics/example_MRE_data.xlsx`
2. **Dual Generator Setup** — Configure both Ollama and OpenAI generators
3. **Batch Processing** — Extract structured data using `ExtraCTOpsProcessor`
4. **Automatic Saving** — Results saved as Excel, JSON, and backup files
5. **Analysis & Export** — Compare results and export final outputs

### 📊 **Output Files**
All outputs are automatically saved to `exports/` with timestamps:
- `extraction_results_YYYYMMDD_HHMMSS.xlsx` — Main extraction results
- `extraction_summary_YYYYMMDD_HHMMSS.json` — Processing summary and statistics
- `extraction_backup_YYYYMMDD_HHMMSS.json` — Complete backup of all data

### 🎯 **Key Features**
- **Error Handling** — Robust error handling with detailed logging
- **Progress Tracking** — Real-time progress indicators during batch processing
- **Dual LLM Support** — Demonstrates both local (Ollama) and cloud (OpenAI) LLMs
- **Production Ready** — Includes proper logging, backup, and recovery mechanisms

---

## 🛠️ Troubleshooting

### Kernel Issues
If kernels don't appear in VS Code:
1. **Restart VS Code** completely
2. **Check Extensions**: Ensure Python and Jupyter extensions are installed
3. **Re-run Setup**: Run `kernel_setup_helper.py` again
4. **Manual Registration**: Re-register kernels if needed:
   ```bash
   ./the_venvs/venv_main/bin/python -m ipykernel install --user --name=extractops_main --display-name="ExtraCTOps Main (Python 3.12)"
   ```

### Import Errors
If you get import errors in the notebook:
1. **Check Kernel**: Ensure you selected the correct ExtraCTOps kernel
2. **Verify Setup**: Run `kernel_setup_helper.py` to verify imports
3. **Environment Path**: Check that `sys.path` includes the project root

### LLM Connection Issues
- **Ollama**: Ensure Ollama is running locally (`ollama serve`)
- **OpenAI**: Verify your API key is set in environment variables or config

---

## 🔧 Configuration

### Environment Variables
Set these before running notebooks:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
# Or use .env file in project root
```

### LLM Models
The notebook uses these models by default:
- **Ollama**: `llama3.2:3b` (or fallback to available models)
- **OpenAI**: `gpt-4o-mini` (cost-effective for structured extraction)

---

## 📈 Performance Tips

1. **Batch Size**: Adjust batch size in `ExtraCTOpsProcessor` based on your data size
2. **Concurrent Processing**: Configure `max_concurrent_requests` based on your system
3. **Model Selection**: Choose appropriate models for your accuracy/speed requirements
4. **Logging Level**: Set appropriate logging levels in production

---

## 🚨 Important Notes

- **API Costs**: Be mindful of OpenAI API costs when processing large datasets
- **Local Models**: Ensure sufficient RAM for local Ollama models
- **Data Privacy**: Local Ollama processing keeps data on your machine
- **Backups**: All processing creates automatic backups — check `exports/` folder

---

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `kernel_setup_helper.py` for diagnostic information
3. Check logs in `utils/inference_logs/` and `utils/logs/`
4. Review the main project README.md for additional setup instructions
