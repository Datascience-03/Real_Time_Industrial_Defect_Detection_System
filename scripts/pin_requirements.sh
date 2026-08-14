#!/usr/bin/env bash
# Create a virtualenv, install project requirements, and pin exact versions.
set -euo pipefail

VENV_DIR=.venv_pin
PYTHON=${PYTHON:-python3}

echo "Creating virtualenv in ${VENV_DIR}..."
${PYTHON} -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Generating pinned requirements to requirements-pinned.txt"
pip freeze > requirements-pinned.txt
echo "Done. Review requirements-pinned.txt and commit if satisfied."
