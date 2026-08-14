<#
Creates a virtualenv, installs `requirements.txt`, and writes pinned versions to `requirements-pinned.txt`.
Run in PowerShell (may require temporary execution policy change).

Usage (PowerShell):
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\scripts\pin_requirements.ps1
#>

$venvDir = ".venv_pin"

Write-Output "Creating virtual environment in $venvDir..."
python -m venv $venvDir

$activate = Join-Path $venvDir "Scripts\Activate.ps1"
if (-Not (Test-Path $activate)) {
    Write-Error "Activation script not found at $activate. Ensure Python is installed and on PATH."
    exit 1
}

Write-Output "Activating virtual environment..."
& $activate

Write-Output "Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

Write-Output "Freezing installed packages to requirements-pinned.txt..."
pip freeze | Out-File -Encoding ASCII requirements-pinned.txt

Write-Output "Pinned requirements written to requirements-pinned.txt"
Write-Output "Tip: review the file and commit requirements-pinned.txt if satisfied."
