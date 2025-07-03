# LM Evaluation Harness Task List Generator

This directory contains scripts to generate a comprehensive list of all available tasks from the OpenDataHub fork of lm-evaluation-harness and save them to a CSV file.

## Overview

The task list generator consists of two scripts:
- `generate_task_list.sh`: Main bash script that handles installation and orchestration
- `extract_tasks.py`: Python script that extracts tasks using the lm_eval CLI and extracts dataset information from YAML configuration files

## Prerequisites

- Python 3.7 or higher
- pip package manager
- Git (for installing from GitHub)
- PyYAML library (`pip install pyyaml`)

## Usage

### Quick Start

Run the main script:

```bash
./generate_task_list.sh
```

This will:
1. Install lm-evaluation-harness from the OpenDataHub fork (default: release-0.4.8 branch)
2. Extract all available tasks using the lm_eval CLI
3. Read YAML configuration files to extract dataset information
4. Save the results to `tasks.csv` with task names and datasets
5. Create a combined CSV with original task list if available

### Using a Different Branch

To use a different branch or tag:

```bash
./generate_task_list.sh main
./generate_task_list.sh v0.5.0
./generate_task_list.sh feature-branch
```

If no branch is specified, it defaults to `release-0.4.8`.

## Output

The script generates the following files:

1. **`tasks.csv`**: Contains task names and their associated datasets
2. **`new_task_list.csv`**: Combined CSV with original task list data plus existence and dataset information (if `original_task_list.csv` exists)

### Example Output

**tasks.csv**:
```csv
Name,Dataset
arc_challenge,ai2_arc
arc_easy,ai2_arc
boolq,boolq
hellaswag,hellaswag
winogrande,winogrande
wmdp_bio,wmdp
wmdp_chem,wmdp
wmdp_cyber,wmdp
```

**new_task_list.csv** (if original_task_list.csv exists):
```csv
Name,HF dataset downloads,Exists,Dataset
arc_challenge,219310,true,ai2_arc
arc_easy,219310,true,ai2_arc
boolq,158151,true,boolq
hellaswag,186239,true,hellaswag
```
