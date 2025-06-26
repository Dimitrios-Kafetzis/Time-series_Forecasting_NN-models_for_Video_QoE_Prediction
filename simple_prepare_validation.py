#!/usr/bin/env python3
"""
Filename: simple_prepare_validation.py
Description:
    A simplified script to prepare a validation dataset from an existing validation set.
    It takes an input validation directory and creates a structured validation dataset
    with ground truth and inference files for model validation.

Usage:
    python3 simple_prepare_validation.py --input_folder ./final_complete_dataset/validation_set --output_folder ./validation_dataset
"""

import os
import json
import argparse
import shutil
from datetime import datetime
import sys

def setup_directories(output_folder):
    """Create the necessary directory structure for validation data."""
    # Create directory for files with ground truth
    ground_truth_dir = os.path.join(output_folder, "ground_truth")
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)
    
    # Create directory for files without QoE (for inference)
    inference_dir = os.path.join(output_folder, "inference")
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
        
    return ground_truth_dir, inference_dir

def create_inference_copy(file_path, target_path):
    """
    Create a copy of the JSON file with QoE values removed.
    Returns the original QoE value for reference.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Get the original QoE value before removing
        original_qoe = data.get('QoE')
        
        # Replace QoE with None (representing unknown)
        data['QoE'] = None
        
        # Write the modified JSON to the target path
        with open(target_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        return original_qoe
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def get_file_qoe(file_path):
    """Extract QoE value from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('QoE')
    except Exception as e:
        print(f"Error reading QoE from {file_path}: {str(e)}")
        return None

def create_sequences(files, input_folder, seq_length=5):
    """
    Group files into chronological sequences of specified length.
    
    Args:
        files: List of JSON filenames
        input_folder: Folder containing the files
        seq_length: Number of files in each sequence (default: 5)
    
    Returns:
        List of sequences, where each sequence is a list of filenames
    """
    # Extract timestamps from filenames
    timestamped_files = []
    for filename in files:
        try:
            # Extract timestamp from filename (assuming format like '20250402131554.json')
            timestamp = int(os.path.splitext(filename)[0])
            timestamped_files.append((filename, timestamp))
        except ValueError:
            # Skip files without valid timestamps
            continue
    
    # Sort files by timestamp
    timestamped_files.sort(key=lambda x: x[1])
    sorted_files = [item[0] for item in timestamped_files]
    
    # Group into sequences
    sequences = []
    for i in range(len(sorted_files) - seq_length + 1):
        sequence = sorted_files[i:i+seq_length]
        sequences.append(sequence)
    
    return sequences

def main():
    parser = argparse.ArgumentParser(description='Prepare a validation dataset from an existing validation set.')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to folder containing original validation JSON files.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to output folder for validation dataset.')
    parser.add_argument('--seq_length', type=int, default=5,
                        help='Sequence length used by the models (default: 5).')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder {args.input_folder} does not exist.")
        sys.exit(1)
    
    # Create output directories
    ground_truth_dir, inference_dir = setup_directories(args.output_folder)
    
    # Create a metadata file to store ground truth QoE values
    metadata_path = os.path.join(args.output_folder, "validation_metadata.json")
    metadata = {
        "creation_timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "files": {},
        "format": "new",  # Using new format (augmented)
        "sample_ratio": 1.0,  # Using 100% of files
        "sequence_based": True,
        "sequence_length": args.seq_length
    }
    
    # Get all JSON files from the input folder
    all_files = [f for f in os.listdir(args.input_folder) if f.endswith('.json')]
    print(f"Found {len(all_files)} JSON files in input folder.")
    
    # Create sequences
    sequences = create_sequences(all_files, args.input_folder, args.seq_length)
    print(f"Created {len(sequences)} sequences of length {args.seq_length}.")
    
    # Store sequence information in metadata
    metadata["sequences"] = [{"files": sequence} for sequence in sequences]
    
    # Process each file
    valid_files = 0
    invalid_files = 0
    
    for filename in all_files:
        src_path = os.path.join(args.input_folder, filename)
        
        # Get QoE value
        qoe_value = get_file_qoe(src_path)
        
        if qoe_value is not None:
            # Copy original file to ground truth directory
            gt_path = os.path.join(ground_truth_dir, filename)
            shutil.copy2(src_path, gt_path)
            
            # Create modified version with QoE removed
            inf_path = os.path.join(inference_dir, filename)
            create_inference_copy(src_path, inf_path)
            
            # Store original QoE value in metadata
            metadata["files"][filename] = {
                "ground_truth_qoe": qoe_value
            }
            valid_files += 1
        else:
            invalid_files += 1
    
    # Update sequence QoE values in metadata
    for i, seq_info in enumerate(metadata["sequences"]):
        sequence = seq_info["files"]
        last_file = sequence[-1]
        if last_file in metadata["files"]:
            qoe = metadata["files"][last_file]["ground_truth_qoe"]
            metadata["sequences"][i]["ground_truth_qoe"] = qoe
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Print summary
    print(f"\nValidation dataset preparation complete:")
    print(f"  - Files processed: {valid_files}")
    print(f"  - Files skipped: {invalid_files}")
    print(f"  - Sequences created: {len(sequences)}")
    print(f"  - Ground truth files saved to: {ground_truth_dir}")
    print(f"  - Inference files saved to: {inference_dir}")
    print(f"  - Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()
