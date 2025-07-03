# LM Evaluation Harness Task List Generator

This directory contains scripts to generate a comprehensive list of all available tasks from the OpenDataHub fork of lm-evaluation-harness and save them to a CSV file.

## Overview

The task list generator consists of two scripts:
- `generate_task_list.sh`: Main bash script that handles installation and orchestration
- `extract_tasks.py`: Python script that extracts tasks using the lm_eval API

## Prerequisites

- Python 3.7 or higher
- pip package manager
- Git (for installing from GitHub)

## Usage

### Quick Start

Run the main script:

```bash
./generate_task_list.sh
```

This will:
1. Install lm-evaluation-harness from the OpenDataHub fork (default: release-0.4.8 branch)
2. Extract all available tasks using the Python API
3. Save the results to `tasks.csv`

### Using a Different Branch

To use a different branch or tag:

```bash
./generate_task_list.sh main
./generate_task_list.sh v0.5.0
./generate_task_list.sh feature-branch
```

If no branch is specified, it defaults to `release-0.4.8`.

## Output

The script generates a CSV file called `tasks.csv` with just the task names:

### Example Output

```csv
Name
arc_challenge
arc_easy
boolq
hellaswag
winogrande
wmdp_bio
wmdp_chem
wmdp_cyber
```
