#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict

def read_task_list(filename='new_task_list.csv'):
    """Read the task list CSV file"""
    try:
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            tasks = list(reader)
        
        print(f"Read {len(tasks)} tasks from {filename}")
        return tasks
        
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run extract_tasks.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        sys.exit(1)

def filter_tasks(tasks):
    """Filter tasks that are Tier 1"""
    filtered_tasks = []
    
    for task in tasks:
        # Check if task is Tier 1
        if task.get('Tier', '').strip() == '1':
            filtered_tasks.append(task)
    
    print(f"Filtered to {len(filtered_tasks)} Tier 1 tasks")
    return filtered_tasks

def group_tasks_by_dataset(tasks):
    """Group tasks by their dataset"""
    dataset_groups = defaultdict(list)
    
    for task in tasks:
        dataset = task.get('Dataset', 'Unknown').strip()
        if dataset and dataset != 'Unknown':
            dataset_groups[dataset].append(task)
    
    # Sort datasets by number of tasks (descending)
    sorted_groups = dict(sorted(dataset_groups.items(), 
                               key=lambda x: len(x[1]), 
                               reverse=True))
    
    print(f"Grouped tasks into {len(sorted_groups)} datasets")
    return sorted_groups

def generate_grouping_table(dataset_groups):
    """Generate and display a table of dataset groupings"""
    print("\n" + "="*80)
    print("DATASET GROUPINGS (Tier 1 Tasks)")
    print("="*80)
    
    total_tasks = 0
    for i, (dataset, tasks) in enumerate(dataset_groups.items(), 1):
        task_count = len(tasks)
        total_tasks += task_count
        
        # Get download count (should be same for all tasks in group)
        downloads = tasks[0].get('HF dataset downloads', 'Unknown')
        
        print(f"\n{i}. Dataset: {dataset}")
        print(f"   Downloads: {int(downloads):,}" if str(downloads).isdigit() else f"   Downloads: {downloads}")
        print(f"   Task Count: {task_count}")
        print(f"   Tasks: {', '.join(task['Name'] for task in tasks[:10])}")
        if task_count > 10:
            print(f"          ... and {task_count - 10} more")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(dataset_groups)} datasets, {total_tasks} total tasks")
    print(f"{'='*80}")

def save_grouping_csv(dataset_groups, filename='dataset_groupings.csv'):
    """Save dataset groupings to CSV file"""
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Dataset', 'Downloads', 'Task Count', 'Task Names'])
            
            for dataset, tasks in dataset_groups.items():
                downloads = tasks[0].get('HF dataset downloads', 'Unknown')
                task_count = len(tasks)
                task_names = '; '.join(task['Name'] for task in tasks)
                
                writer.writerow([dataset, downloads, task_count, task_names])
        
        print(f"Saved dataset groupings to {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return False

def main():
    """Main function to group tasks by dataset"""
    print("Grouping tasks by dataset...")
    
    # Read task list
    tasks = read_task_list('new_task_list.csv')
    
    # Filter tasks that are Tier 1
    filtered_tasks = filter_tasks(tasks)
    
    if not filtered_tasks:
        print("No Tier 1 tasks found")
        sys.exit(1)
    
    # Group tasks by dataset
    dataset_groups = group_tasks_by_dataset(filtered_tasks)
    
    if not dataset_groups:
        print("No dataset groupings found")
        sys.exit(1)
    
    # Generate and display table
    generate_grouping_table(dataset_groups)
    
    # Save to CSV
    save_grouping_csv(dataset_groups, 'dataset_groupings.csv')
    
    print("\nDataset grouping completed successfully!")

if __name__ == "__main__":
    main() 