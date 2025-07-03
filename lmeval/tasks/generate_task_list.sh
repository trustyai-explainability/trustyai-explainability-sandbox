#!/bin/bash

# Script to install lm-evaluation-harness from ODH fork and generate task list CSV
set -e

# Set default branch if not provided
BRANCH=${1:-release-0.4.8}

echo "Installing lm-evaluation-harness from OpenDataHub fork (branch: $BRANCH)..."
pip install git+https://github.com/opendatahub-io/lm-evaluation-harness.git@$BRANCH

echo "Generating task list with Python..."
python3 extract_tasks.py

echo "Script completed successfully!"
