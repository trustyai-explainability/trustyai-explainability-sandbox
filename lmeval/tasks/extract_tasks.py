#!/usr/bin/env python3

import csv
import sys
import subprocess
import os

def get_tasks_from_cli():
    """Get tasks using the lm_eval CLI command"""
    try:
        print("Getting tasks using lm_eval CLI...")
        result = subprocess.run(['lm_eval', '--tasks', 'list'], 
                              capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        task_list = []
        
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
            
            task_name = None
            
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:  # Ensure it's actually a table row
                    for part in parts:
                        part = part.strip()
                        if part and part != '':
                            task_name = part
                            break
            else:
                words = line.split()
                if words:
                    task_name = words[0]
            
            if (task_name and 
                task_name != '' and
                not task_name.startswith(('-', '=', '*', '>', '<', '|')) and
                ':' not in task_name and
                not task_name.lower() in ['available', 'tasks', 'groups', 'warning', 'error', 'usage', 'help', 'tag', 'name'] and
                not task_name.isdigit() and
                len(task_name) > 1 and
                not all(c in '-|=+ ' for c in task_name) and
                task_name != 'Name' and
                # Additional filters for table headers
                not task_name.lower().startswith('task') and
                not task_name.lower().startswith('group')):
                
                task_list.append(task_name)
        
        seen = set()
        unique_tasks = []
        for task in task_list:
            if task not in seen:
                seen.add(task)
                unique_tasks.append(task)
        
        unique_tasks.sort()
        
        print(f"Extracted {len(unique_tasks)} unique tasks")
        if len(unique_tasks) > 0:
            print(f"First few tasks: {unique_tasks[:5]}")
        
        return unique_tasks
        
    except Exception as e:
        print(f"CLI method failed: {e}")
        return None

def save_simple_csv(task_list, filename='tasks.csv'):
    """Save task list to simple CSV file with just Name column"""
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Name'])  # Header
            
            for task in task_list:
                writer.writerow([task])
        
        print(f"Successfully saved {len(task_list)} tasks to {filename}")
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

def create_combined_csv(original_tasks, new_task_names, output_filename='new_task_list.csv'):
    """Create combined CSV with original columns plus Exists column"""
    try:
        if not original_tasks:
            print("No original tasks to combine with")
            return False
        
        # Convert new task names to a set for faster lookup
        new_task_set = set(new_task_names)
        
        # Get the fieldnames from the original CSV
        fieldnames = list(original_tasks[0].keys()) + ['Exists']
        
        matches = 0
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for task in original_tasks:
                # Copy all original fields
                new_row = task.copy()
                
                # Add Exists column based on whether task name is in new list
                task_name = task.get('Name', '')
                exists = task_name in new_task_set
                new_row['Exists'] = 'true' if exists else 'false'
                
                if exists:
                    matches += 1
                
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
    task_list = get_tasks_from_cli()
    
    if task_list is None or len(task_list) == 0:
        print("Error: Could not extract tasks from lm_eval CLI")
        sys.exit(1)
    
    # Save simple CSV with just task names
    if not save_simple_csv(task_list, 'tasks.csv'):
        print("Error: Failed to save simple task CSV")
        sys.exit(1)
    
    print(f"Found {len(task_list)} tasks")
    
    # Read original task list
    original_tasks = read_original_task_list('original_task_list.csv')
    
    if original_tasks is not None:
        # Create combined CSV
        if create_combined_csv(original_tasks, task_list, 'new_task_list.csv'):
            print("Combined task list created successfully!")
        else:
            print("Warning: Failed to create combined task list")
    
    print("Task list generation completed successfully!")

if __name__ == "__main__":
    main() 