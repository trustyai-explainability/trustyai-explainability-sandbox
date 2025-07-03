#!/usr/bin/env python3

import csv
import sys
import subprocess
import os

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

def get_dataset_from_yaml(yaml_path):
    """Extract dataset information from a YAML file"""
    try:
        if not os.path.exists(yaml_path):
            import importlib.util
            try:
                spec = importlib.util.find_spec('lm_eval')
                if spec and spec.origin:
                    package_dir = os.path.dirname(spec.origin)
                    potential_path = os.path.join(package_dir, yaml_path.replace('lm_eval/', ''))
                    if os.path.exists(potential_path):
                        yaml_path = potential_path
                    else:
                        # Try without the lm_eval prefix
                        yaml_relative = yaml_path.replace('lm_eval/tasks/', '')
                        potential_path = os.path.join(package_dir, 'tasks', yaml_relative)
                        if os.path.exists(potential_path):
                            yaml_path = potential_path
            except ImportError:
                pass
        
        if not os.path.exists(yaml_path):
            return "Unknown"
        
        with open(yaml_path, 'r', encoding='utf-8') as file:
            # Create a custom loader that handles unknown tags gracefully
            class CustomLoader(yaml.SafeLoader):
                def construct_unknown(self, node):
                    # Return a placeholder for unknown tags
                    if isinstance(node, yaml.ScalarNode):
                        return f"<{node.tag}:{node.value}>"
                    elif isinstance(node, yaml.SequenceNode):
                        return [self.construct_unknown(child) for child in node.value]
                    elif isinstance(node, yaml.MappingNode):
                        return {key.value: self.construct_unknown(value) for key, value in node.value}
                    return f"<{node.tag}>"
            
            CustomLoader.add_constructor(None, CustomLoader.construct_unknown)
            data = yaml.load(file, Loader=CustomLoader)
            
        if isinstance(data, dict):
            # Priority order for dataset fields
            dataset_fields = ['dataset_path', 'dataset', 'dataset_name', 'hf_dataset']
            
            # First check for include field that might reference another config
            if 'include' in data:
                include_path = data['include']
                if isinstance(include_path, str):
                    # Recursively check included file
                    base_dir = os.path.dirname(yaml_path)
                    included_yaml = os.path.join(base_dir, include_path)
                    included_dataset = get_dataset_from_yaml(included_yaml)
                    if included_dataset != "Unknown":
                        return included_dataset
            
            for field in dataset_fields:
                if field in data:
                    dataset_value = data[field]
                    if isinstance(dataset_value, str) and dataset_value.strip():
                        return dataset_value.strip()
                    elif isinstance(dataset_value, dict) and 'path' in dataset_value:
                        return dataset_value['path']
                    elif isinstance(dataset_value, dict) and 'name' in dataset_value:
                        return dataset_value['name']
            
            # Check nested task configuration
            if 'task' in data and isinstance(data['task'], dict):
                task_info = data['task']
                for field in dataset_fields:
                    if field in task_info:
                        dataset_value = task_info[field]
                        if isinstance(dataset_value, str) and dataset_value.strip():
                            return dataset_value.strip()
                        elif isinstance(dataset_value, dict) and 'path' in dataset_value:
                            return dataset_value['path']
                        elif isinstance(dataset_value, dict) and 'name' in dataset_value:
                            return dataset_value['name']
            
            # Check for tag field which sometimes contains dataset info
            if 'tag' in data:
                tag_value = data['tag']
                if isinstance(tag_value, list) and len(tag_value) > 0:
                    return tag_value[0]
                elif isinstance(tag_value, str) and tag_value.strip():
                    return tag_value.strip()
        
        return "Unknown"
        
    except Exception as e:
        print(f"Error reading YAML {yaml_path}: {e}")
        return "Unknown"

def get_tasks_from_cli():
    """Get tasks using the lm_eval CLI command and extract YAML paths"""
    try:
        print("Getting tasks using lm_eval CLI...")
        result = subprocess.run(['python3', '-m', 'lm_eval', '--tasks', 'list'],
                              capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        task_data = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            if all(c in '-=|+ ' for c in line) and len(line) > 5:
                continue
            
            if any(header in line.lower() for header in [
                'available tasks:', 'tasks:', 'groups:', 'usage:', 'warning', 
                'error', 'help', 'options:', 'arguments:', 'description:', 'tag'
            ]):
                continue
            
            if line.startswith(('#', '-', '=', '*', '>', '<')) and '|' not in line:
                continue
            
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 3:  # Should have at least empty, Group, Config Location, empty
                    task_name = parts[1] if len(parts) > 1 and parts[1] else None
                    yaml_path = parts[2] if len(parts) > 2 and parts[2] else None
                    eval_type = parts[3] if len(parts) > 3 and parts[3] else "Unknown"
                    
                    if (task_name and yaml_path and 
                        task_name != '' and yaml_path != '' and
                        not task_name.startswith(('-', '=', '*', '>', '<', '|')) and
                        yaml_path.endswith('.yaml') and
                        ':' not in task_name and
                        not task_name.lower() in ['available', 'tasks', 'groups', 'warning', 'error', 'usage', 'help', 'tag', 'name', 'group', 'config', 'location'] and
                        not task_name.isdigit() and
                        len(task_name) > 1 and
                        not all(c in '-|=+ ' for c in task_name) and
                        task_name != 'Group' and
                        not task_name.lower().startswith('task')):
                        
                        # Extract dataset from YAML
                        dataset = get_dataset_from_yaml(yaml_path)
                        task_data.append({
                            'name': task_name,
                            'yaml_path': yaml_path,
                            'dataset': dataset,
                            'eval_type': eval_type
                        })
        
        # Remove duplicates based on task name
        seen = set()
        unique_tasks = []
        for task in task_data:
            if task['name'] not in seen:
                seen.add(task['name'])
                unique_tasks.append(task)
        
        unique_tasks.sort(key=lambda x: x['name'])
        
        print(f"Extracted {len(unique_tasks)} unique tasks with dataset information")
        if len(unique_tasks) > 0:
            print(f"First few tasks: {[t['name'] for t in unique_tasks[:5]]}")
        
        return unique_tasks
        
    except subprocess.CalledProcessError as e:
        print(f"CLI method failed with exit code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"CLI method failed: {e}")
        return None

def save_simple_csv(task_data, filename='tasks.csv'):
    """Save task list to CSV file with Name, Dataset, and Evaluation Type columns"""
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Name', 'Dataset', 'Evaluation Type'])  # Header
            
            for task in task_data:
                writer.writerow([task['name'], task['dataset'], task['eval_type']])
        
        print(f"Successfully saved {len(task_data)} tasks to {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving simple CSV: {e}")
        return False

def read_original_task_list(filename='original_task_list.csv'):
    """Read the original task list CSV"""
    try:
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping combined CSV creation.")
            return None
        
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            original_tasks = list(reader)
        
        print(f"Read {len(original_tasks)} tasks from {filename}")
        return original_tasks
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def create_combined_csv(original_tasks, new_task_data, output_filename='new_task_list.csv'):
    """Create combined CSV with original columns plus Exists, Dataset, and Evaluation Type columns"""
    try:
        if not original_tasks:
            print("No original tasks to combine with")
            return False
        
        new_task_dict = {task['name']: task for task in new_task_data}
        
        fieldnames = list(original_tasks[0].keys()) + ['Exists', 'Dataset', 'Evaluation Type']
        
        matches = 0
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for task in original_tasks:
                # Copy all original fields
                new_row = task.copy()
                
                # Add Exists, Dataset, and Evaluation Type columns
                task_name = task.get('Name', '')
                if task_name in new_task_dict:
                    new_row['Exists'] = 'true'
                    new_row['Dataset'] = new_task_dict[task_name]['dataset']
                    new_row['Evaluation Type'] = new_task_dict[task_name]['eval_type']
                    matches += 1
                else:
                    new_row['Exists'] = 'false'
                    new_row['Dataset'] = 'Unknown'
                    new_row['Evaluation Type'] = 'Unknown'
                
                writer.writerow(new_row)
        
        print(f"Found {matches} matches out of {len(original_tasks)} original tasks")
        print(f"Successfully created combined CSV: {output_filename}")
        return True
        
    except Exception as e:
        print(f"Error creating combined CSV: {e}")
        return False

def main():
    """Main function to generate task list and create combined CSV"""
    print("Generating task list from lm-evaluation-harness...")
    
    # Get tasks using CLI method
    task_data = get_tasks_from_cli()
    
    if task_data is None or len(task_data) == 0:
        print("Error: Could not extract tasks from lm_eval CLI")
        print("Please ensure lm-evaluation-harness is properly installed and accessible")
        sys.exit(1)
    
    # Save simple CSV with task names and datasets
    if not save_simple_csv(task_data, 'tasks.csv'):
        print("Error: Failed to save simple task CSV")
        sys.exit(1)
    
    print(f"Found {len(task_data)} tasks")
    
    # Read original task list
    original_tasks = read_original_task_list('original_task_list.csv')
    
    if original_tasks is not None:
        # Create combined CSV
        if create_combined_csv(original_tasks, task_data, 'new_task_list.csv'):
            print("Combined task list created successfully!")
        else:
            print("Warning: Failed to create combined task list")
    
    print("Task list generation completed successfully!")

if __name__ == "__main__":
    main() 