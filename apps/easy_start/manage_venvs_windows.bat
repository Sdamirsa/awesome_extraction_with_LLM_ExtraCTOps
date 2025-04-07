@echo off
setlocal

echo.
echo 🛠️ INSTRUCTIONS:
echo 1. Open ..\..\the_venvs\venv_info.json
echo 2. Set "enabled": true for any venv you want to create or install.
echo 3. Then run this script.
echo.

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%..\..\the_venvs"
set "INFO_FILE=%VENV_DIR%\venv_info.json"

if not exist "%INFO_FILE%" (
    echo ❌ venv_info.json not found at %INFO_FILE%
    exit /b
)

set /p CONTINUE=❓ Do you want to create venvs and install requirements for enabled environments? (y/n): 
if /i not "%CONTINUE%"=="y" (
    echo 🚫 Aborted by user. No actions taken.
    exit /b
)

echo 🔄 Processing enabled virtual environments...

powershell -NoProfile -Command ^
    "$json = Get-Content '%INFO_FILE%' | ConvertFrom-Json; ^
    foreach ($entry in $json.PSObject.Properties) { ^
        $name = $entry.Name; ^
        $venv = $entry.Value; ^
        if ($venv.enabled -eq $true) { ^
            $venvPath = Join-Path '%VENV_DIR%' (Split-Path -Leaf $venv.venv_path); ^
            $reqPath = if ($venv.requirements_file -ne $null) { Join-Path '%VENV_DIR%' (Split-Path -Leaf $venv.requirements_file) } else { $null }; ^
            if (-not (Test-Path $venvPath)) { ^
                Write-Host '🛠️  Creating venv for' $name '...'; ^
                python -m venv $venvPath; ^
            } else { ^
                Write-Host '✅' $name 'venv already exists. Skipping creation.'; ^
            } ^
            if ($reqPath -and (Test-Path $reqPath) -and ((Get-Content $reqPath).Length -gt 0)) { ^
                Write-Host '📦 Installing from' $reqPath; ^
                & (Join-Path $venvPath 'Scripts\python.exe') -m pip install -r $reqPath ^
            } else { ^
                Write-Host '⚠️  No requirements to install or file is empty for' $name; ^
            } ^
        } ^
    }"

echo.
echo ✅ All enabled environments processed.

endlocal