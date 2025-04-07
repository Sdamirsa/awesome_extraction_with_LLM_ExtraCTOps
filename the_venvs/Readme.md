# 🧪 Virtual Environment Management — `the_venvs/`

This folder contains **modular virtual environments** for your project. Each environment is associated with a specific script or pipeline that may have unique dependencies. This design enables isolation of potentially conflicting packages, while maintaining a shared main environment for common workflows.

---

## 📁 Contents

- `venv_main/`  
  The main virtual environment used across shared pipelines and general workflows.

- `venv_<script_name>/`  
  Script-specific virtual environments created for scripts that require different or conflicting libraries.

- `requirements_<script_name>.txt`  
  A requirements file for each script-specific venv. You should manually add the necessary packages for the corresponding script here.

- `venv_info.json`  
  The central configuration file. It contains metadata about each environment, including:
  - Path to the script it supports
  - Path to the Python executable
  - Path to the requirements file
  - Whether it’s currently **enabled** (for selective activation)

---

## ⚙️ How to Use

### 🔧 1. Modify `venv_info.json`

Enable only the environments you need by changing:

```json
"enabled": true

For example:

"venv_ollama": {
  "enabled": true,
  ...
}

📦 2. Install Dependencies

Run the appropriate environment management script to install packages for the enabled environments:
	•	On macOS/Linux:

./manage_venvs_mac.sh


	•	On Windows:

manage_venvs_windows.bat



Only environments marked as "enabled": true will be processed.

⸻

📝 Notes
	•	If a requirements_*.txt file is empty, it will be skipped during installation.
	•	If an environment is not "enabled", it will be ignored.
	•	You can safely edit, copy, or reuse this structure across projects.

⸻

✅ Best Practices
	•	Use venv_main for general tasks and shared libraries.
	•	Use script-specific venvs only when necessary (e.g. GPU libraries, conflicting versions).
	•	Keep venv_info.json organized and consistent.

⸻

📍 Example Workflow
	1.	Create or update a script: generation/gpt4_ollama.py
	2.	Edit venv_info.json:

"venv_gpt4_ollama": {
  "enabled": true,
  ...
}


	3.	Add packages to requirements_gpt4_ollama.txt
	4.	Run ./manage_venvs_mac.sh or manage_venvs_windows.bat

⸻