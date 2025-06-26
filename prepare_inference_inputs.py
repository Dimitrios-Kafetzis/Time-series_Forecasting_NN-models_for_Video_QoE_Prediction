#!/usr/bin/env python3
"""
Filename: prepare_inference_inputs_v2.py
Description:
    This script extracts files from your validation dataset to create 
    a set of inference inputs for testing with run_inference.py.
    
    It copies the files to the inference_inputs directory and REMOVES
    the QoE values from the copied files, since those are what the model
    will be predicting.
    
    The script also creates a reference file with the original QoE values
    so you can compare the model predictions against ground truth.

Usage:
    python3 prepare_inference_inputs_v2.py --input_folder ./final_complete_dataset/validation_set --output_folder ./inference_inputs [options]
    
Options:
    --num_files: Number of files to extract (default: 5)
    --start_index: Starting index in the sorted file list (default: 0)
    --sequential: Extract files in sequence rather than evenly spaced (default: False)
    --qoe_range: Extract files with diverse QoE values (default: False)
    --augmented: Indicate that files are in augmented format (default: False)
"""

import os
import json
import argparse
import shutil
import numpy as np
from datetime import datetime

def get_file_qoe(file_path):
    """Extract QoE value from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('QoE')
    except Exception as e:
        print(f"Error reading QoE from {file_path}: {str(e)}")
        return None

def extract_timestamp(filename):
    """Extract timestamp from filename."""
    try:
        # Extract timestamp from filename (assuming format like '20250402131554.json')
        timestamp = int(os.path.splitext(filename)[0])
        return timestamp
    except ValueError:
        # Return 0 for files without valid timestamps
        return 0

def select_files_by_qoe_range(files, input_folder, num_files):
    """Select files to cover a wide range of QoE values."""
    # Get QoE values for all files
    file_qoe_pairs = []
    for filename in files:
        file_path = os.path.join(input_folder, filename)
        qoe = get_file_qoe(file_path)
        if qoe is not None:
            file_qoe_pairs.append((filename, qoe))
    
    # Sort by QoE value
    file_qoe_pairs.sort(key=lambda x: x[1])
    
    # If fewer files than requested, return all
    if len(file_qoe_pairs) <= num_files:
        return [f[0] for f in file_qoe_pairs]
    
    # Select evenly spaced files across the QoE range
    indices = np.linspace(0, len(file_qoe_pairs) - 1, num_files, dtype=int)
    return [file_qoe_pairs[i][0] for i in indices]

def select_sequential_files(files, start_index, num_files):
    """Select files sequentially starting from start_index."""
    # Ensure start_index is valid
    if start_index >= len(files):
        start_index = 0
    
    # Get sequential files
    end_index = min(start_index + num_files, len(files))
    return files[start_index:end_index]

def select_evenly_spaced_files(files, num_files):
    """Select evenly spaced files from the full list."""
    # If fewer files than requested, return all
    if len(files) <= num_files:
        return files
    
    # Select evenly spaced files
    indices = np.linspace(0, len(files) - 1, num_files, dtype=int)
    return [files[i] for i in indices]

def copy_file_without_qoe(src_path, dst_path):
    """Copy a file to destination but remove the QoE value."""
    try:
        with open(src_path, 'r') as f:
            data = json.load(f)
        
        # Store the original QoE value
        original_qoe = data.get('QoE')
        
        # Remove QoE value (set to None)
        data['QoE'] = None
        
        # Write modified data to destination
        with open(dst_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        return original_qoe
    except Exception as e:
        print(f"Error copying and modifying file {src_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Prepare inference inputs from validation dataset.')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to folder containing validation JSON files.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to output folder for inference inputs.')
    parser.add_argument('--num_files', type=int, default=5,
                        help='Number of files to extract (default: 5).')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting index in the sorted file list (default: 0).')
    parser.add_argument('--sequential', action='store_true',
                        help='Extract files in sequence rather than evenly spaced.')
    parser.add_argument('--qoe_range', action='store_true',
                        help='Extract files with diverse QoE values.')
    parser.add_argument('--augmented', action='store_true',
                        help='Indicate that files are in augmented format.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder {args.input_folder} does not exist.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # Get all JSON files from the input folder
    all_files = [f for f in os.listdir(args.input_folder) if f.endswith('.json')]
    print(f"Found {len(all_files)} JSON files in input folder.")
    
    # Sort files by timestamp
    sorted_files = sorted(all_files, key=extract_timestamp)
    
    # Select files based on options
    if args.qoe_range:
        selected_files = select_files_by_qoe_range(sorted_files, args.input_folder, args.num_files)
        print(f"Selected {len(selected_files)} files across QoE range.")
    elif args.sequential:
        selected_files = select_sequential_files(sorted_files, args.start_index, args.num_files)
        print(f"Selected {len(selected_files)} sequential files starting at index {args.start_index}.")
    else:
        selected_files = select_evenly_spaced_files(sorted_files, args.num_files)
        print(f"Selected {len(selected_files)} evenly spaced files.")
    
    # Copy selected files to output directory (without QoE values)
    copied_count = 0
    ground_truth = {}
    
    for filename in selected_files:
        src_path = os.path.join(args.input_folder, filename)
        dst_path = os.path.join(args.output_folder, filename)
        try:
            original_qoe = copy_file_without_qoe(src_path, dst_path)
            copied_count += 1
            
            # Store ground truth QoE for reference
            if original_qoe is not None:
                ground_truth[filename] = original_qoe
                print(f"Copied {filename} - Original QoE: {original_qoe}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Save ground truth information to a reference file
    reference_path = os.path.join(args.output_folder, "ground_truth_reference.json")
    with open(reference_path, 'w') as f:
        json.dump({
            "files": ground_truth,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Ground truth QoE values for inference files"
        }, f, indent=4)
    
    print(f"\nSuccessfully copied {copied_count} files to {args.output_folder}")
    print(f"Ground truth QoE values saved to {reference_path}")

if __name__ == "__main__":
    main()
