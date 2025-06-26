#!/usr/bin/env python3
"""
Filename: prepare_validation_data.py
Description:
    This script prepares a validation dataset by creating pairs of files:
    1. Original files containing the ground truth QoE values
    2. Modified files with QoE values removed (for inference)
    
    The script supports the new augmented dataset format (10-second windows with 
    2-second intervals) as well as the legacy format.

Usage Examples:
    Basic usage:
      $ python3 prepare_validation_data.py --input_folder ./real_dataset --output_folder ./validation_dataset

    Specify a specific subset (percentage) of files to process:
      $ python3 prepare_validation_data.py --input_folder ./real_dataset --output_folder ./validation_dataset --sample_ratio 0.2

    Create samples using a specified seed for reproducibility:
      $ python3 prepare_validation_data.py --input_folder ./real_dataset --output_folder ./validation_dataset --sample_ratio 0.3 --random_seed 42

    Filter files by QoE range:
      $ python3 prepare_validation_data.py --input_folder ./real_dataset --output_folder ./validation_dataset --qoe_min 80 --qoe_max 100

    Select a specific number of files:
      $ python3 prepare_validation_data.py --input_folder ./real_dataset --output_folder ./validation_dataset --num_files 50

    Create a validation dataset from legacy format files:
      $ python3 prepare_validation_data.py --input_folder ./old_dataset --output_folder ./validation_dataset --legacy_format
"""

import os
import json
import argparse
import random
import shutil
from datetime import datetime
import sys

def setup_directories(base_output_folder):
    """Create the necessary directory structure for validation data."""
    # Create directory for files with ground truth
    ground_truth_dir = os.path.join(base_output_folder, "ground_truth")
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)
    
    # Create directory for files without QoE (for inference)
    inference_dir = os.path.join(base_output_folder, "inference")
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
        
    return ground_truth_dir, inference_dir

def create_inference_copy(file_path, target_path, legacy_format=False):
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

def validate_json_structure(file_path, legacy_format=False):
    """
    Validate if the JSON file has the expected structure for validation.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check for required fields
        if 'QoE' not in data or 'timestamp' not in data:
            return False
        
        # If QoE is null or None, file is not valid for validation
        if data['QoE'] is None:
            return False
            
        # For new format (default), check for timestamp data structure
        if not legacy_format:
            # Check that there are timestamp keys with nested data
            timestamp_keys = [k for k in data.keys() if k not in ['QoE', 'timestamp']]
            if not timestamp_keys:
                return False
                
            # Check nested structure on first timestamp
            first_ts = timestamp_keys[0]
            required_fields = ['throughput', 'packets_lost', 'packet_loss_rate', 'jitter', 'speed']
            for field in required_fields:
                if field not in data[first_ts]:
                    return False
                    
        return True
    except Exception as e:
        print(f"Error validating file {file_path}: {str(e)}")
        return False

def get_file_qoe(file_path):
    """Extract QoE value from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('QoE')
    except Exception as e:
        print(f"Error reading QoE from {file_path}: {str(e)}")
        return None

def filter_files_by_qoe_range(files, input_folder, qoe_min=None, qoe_max=None):
    """
    Filter files based on their QoE values falling within a specified range.
    
    Args:
        files: List of all JSON filenames
        input_folder: Folder containing the files
        qoe_min: Minimum QoE value to include (inclusive)
        qoe_max: Maximum QoE value to include (inclusive)
        
    Returns:
        List of filenames that have QoE values within the specified range
    """
    # If no range filters are set, return all files
    if qoe_min is None and qoe_max is None:
        return files
    
    filtered_files = []
    for filename in files:
        file_path = os.path.join(input_folder, filename)
        qoe = get_file_qoe(file_path)
        
        # Skip files with no QoE value
        if qoe is None:
            continue
        
        # Check if QoE is within range
        if qoe_min is not None and qoe < qoe_min:
            continue
        if qoe_max is not None and qoe > qoe_max:
            continue
        
        # If we get here, QoE is within range
        filtered_files.append(filename)
    
    print(f"QoE range filter: {qoe_min if qoe_min is not None else 'min'}-{qoe_max if qoe_max is not None else 'max'}")
    print(f"Filtered from {len(files)} to {len(filtered_files)} files based on QoE range")
    return filtered_files

def stratified_sampling(files, input_folder, sample_count=None, sample_ratio=1.0, num_bins=10):
    """
    Sample files to ensure representation across the entire QoE range.
    
    Args:
        files: List of all JSON filenames
        input_folder: Folder containing the files
        sample_count: Exact number of files to select (overrides sample_ratio if set)
        sample_ratio: Fraction of files to include (used if sample_count is None)
        num_bins: Number of QoE range bins (default: 10)
        
    Returns:
        List of selected filenames
    """
    # First pass: gather all QoE values
    file_qoe_pairs = []
    for filename in files:
        file_path = os.path.join(input_folder, filename)
        qoe = get_file_qoe(file_path)
        if qoe is not None:
            file_qoe_pairs.append((filename, qoe))
    
    # Sort by QoE
    file_qoe_pairs.sort(key=lambda x: x[1])
    
    if not file_qoe_pairs:
        print("No valid QoE values found!")
        return []
    
    # Calculate number of files to select
    if sample_count is not None:
        num_files_to_select = min(sample_count, len(file_qoe_pairs))
    else:
        num_files_to_select = int(len(file_qoe_pairs) * sample_ratio)
    
    # Divide into bins
    total_files = len(file_qoe_pairs)
    bin_size = total_files // num_bins
    if bin_size == 0:  # If we have fewer files than bins
        bin_size = 1
        actual_bins = min(total_files, num_bins)
    else:
        actual_bins = num_bins
    
    # Calculate how many files to select from each bin
    files_per_bin = num_files_to_select // actual_bins
    remainder = num_files_to_select % actual_bins
    
    # Select files from each bin
    selected_files = []
    for i in range(actual_bins):
        bin_start = i * bin_size
        bin_end = bin_start + bin_size if i < actual_bins - 1 else total_files
        bin_files = [pair[0] for pair in file_qoe_pairs[bin_start:bin_end]]
        
        # Add extra file from first few bins if we have remainder
        extra_file = 1 if i < remainder else 0
        num_to_select = min(files_per_bin + extra_file, len(bin_files))
        
        # Randomly select files from this bin
        if num_to_select > 0:
            bin_selected = random.sample(bin_files, num_to_select)
            selected_files.extend(bin_selected)
    
    print(f"Stratified sampling selected {len(selected_files)} files across {actual_bins} QoE ranges")
    return selected_files

def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename assuming format like '20250402122207.json'"""
    try:
        # Remove file extension and convert to integer
        return int(os.path.splitext(filename)[0])
    except ValueError:
        # If filename doesn't follow expected pattern
        return None

def group_files_into_sequences(files, sequence_length=5):
    """
    Group files into chronological sequences of specified length.
    
    Args:
        files: List of JSON filenames
        sequence_length: Number of files in each sequence (default: 5)
    
    Returns:
        List of sequences, where each sequence is a list of filenames
    """
    # Extract timestamps from filenames
    timestamped_files = []
    for filename in files:
        timestamp = extract_timestamp_from_filename(filename)
        if timestamp is not None:
            timestamped_files.append((filename, timestamp))
    
    # Sort files by timestamp
    timestamped_files.sort(key=lambda x: x[1])
    sorted_files = [item[0] for item in timestamped_files]
    
    # Group into sequences
    sequences = []
    for i in range(len(sorted_files) - sequence_length + 1):
        sequence = sorted_files[i:i+sequence_length]
        sequences.append(sequence)
    
    return sequences

def get_sequence_qoe(sequence, input_folder):
    """
    Calculate representative QoE for a sequence of files.
    Uses the QoE of the last file in the sequence.
    
    Args:
        sequence: List of filenames in the sequence
        input_folder: Path to folder containing the files
    
    Returns:
        QoE value (float) or None if not available
    """
    # Get the last file in the sequence
    last_file = sequence[-1]
    file_path = os.path.join(input_folder, last_file)
    
    return get_file_qoe(file_path)

def stratified_sequence_sampling(files, input_folder, sample_count=None, sample_ratio=1.0, num_bins=10, sequence_length=5):
    """
    Sample sequences to ensure representation across the entire QoE range.
    
    Args:
        files: List of all JSON filenames
        input_folder: Folder containing the files
        sample_count: Exact number of sequences to select (overrides sample_ratio if set)
        sample_ratio: Fraction of sequences to include (used if sample_count is None)
        num_bins: Number of QoE range bins (default: 10)
        sequence_length: Number of files in each sequence (default: 5)
        
    Returns:
        List of selected sequences, each a list of filenames
    """
    # Group files into sequences
    sequences = group_files_into_sequences(files, sequence_length)
    print(f"Found {len(sequences)} sequences of length {sequence_length}")
    
    if not sequences:
        print("No valid sequences found!")
        return []
    
    # Calculate QoE for each sequence
    sequence_qoe_pairs = []
    for sequence in sequences:
        qoe = get_sequence_qoe(sequence, input_folder)
        if qoe is not None:
            sequence_qoe_pairs.append((sequence, qoe))
    
    # Sort by QoE
    sequence_qoe_pairs.sort(key=lambda x: x[1])
    
    if not sequence_qoe_pairs:
        print("No valid QoE values found for sequences!")
        return []
    
    # Calculate number of sequences to select
    if sample_count is not None:
        num_sequences_to_select = min(sample_count, len(sequence_qoe_pairs))
    else:
        num_sequences_to_select = int(len(sequence_qoe_pairs) * sample_ratio)
    
    # Divide into bins
    total_sequences = len(sequence_qoe_pairs)
    bin_size = total_sequences // num_bins
    if bin_size == 0:  # If we have fewer sequences than bins
        bin_size = 1
        actual_bins = min(total_sequences, num_bins)
    else:
        actual_bins = num_bins
    
    # Calculate how many sequences to select from each bin
    sequences_per_bin = num_sequences_to_select // actual_bins
    remainder = num_sequences_to_select % actual_bins
    
    # Select sequences from each bin
    selected_sequences = []
    for i in range(actual_bins):
        bin_start = i * bin_size
        bin_end = bin_start + bin_size if i < actual_bins - 1 else total_sequences
        bin_sequences = [pair[0] for pair in sequence_qoe_pairs[bin_start:bin_end]]
        
        # Add extra sequence from first few bins if we have remainder
        extra_sequence = 1 if i < remainder else 0
        num_to_select = min(sequences_per_bin + extra_sequence, len(bin_sequences))
        
        # Randomly select sequences from this bin
        if num_to_select > 0:
            bin_selected = random.sample(bin_sequences, num_to_select)
            selected_sequences.extend(bin_selected)
    
    print(f"Stratified sampling selected {len(selected_sequences)} sequences across {actual_bins} QoE ranges")
    return selected_sequences

def print_qoe_distribution(files, input_folder):
    """
    Print distribution of QoE values in the selected files.
    
    Args:
        files: List of filenames
        input_folder: Path to the folder containing the files
    """
    qoe_values = []
    for filename in files:
        file_path = os.path.join(input_folder, filename)
        qoe = get_file_qoe(file_path)
        if qoe is not None:
            qoe_values.append(qoe)
    
    if not qoe_values:
        print("No QoE values found in selected files.")
        return
    
    # Calculate basic stats
    qoe_min = min(qoe_values)
    qoe_max = max(qoe_values)
    qoe_avg = sum(qoe_values) / len(qoe_values)
    
    # Create histogram-like distribution (5 bins)
    bins = 5
    bin_size = (qoe_max - qoe_min) / bins if qoe_max > qoe_min else 1
    bin_counts = [0] * bins
    
    for qoe in qoe_values:
        bin_idx = min(int((qoe - qoe_min) / bin_size), bins - 1)
        bin_counts[bin_idx] += 1
    
    print("\nQoE Distribution of Selected Files:")
    print(f"  - Range: {qoe_min:.2f} to {qoe_max:.2f}")
    print(f"  - Average: {qoe_avg:.2f}")
    print(f"  - Total Files: {len(qoe_values)}")
    print("  - Distribution:")
    
    for i in range(bins):
        bin_start = qoe_min + i * bin_size
        bin_end = qoe_min + (i + 1) * bin_size if i < bins - 1 else qoe_max
        percentage = (bin_counts[i] / len(qoe_values)) * 100
        print(f"    {bin_start:.2f}-{bin_end:.2f}: {bin_counts[i]} files ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Prepare validation dataset by creating pairs of files.')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to folder containing original JSON files.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to output folder for validation dataset.')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Fraction of files to include in validation set (default: 1.0 = all files).')
    parser.add_argument('--num_files', type=int, default=None,
                        help='Exact number of files to include in validation set (overrides sample_ratio).')
    parser.add_argument('--qoe_min', type=float, default=None,
                        help='Minimum QoE value to include (inclusive).')
    parser.add_argument('--qoe_max', type=float, default=None,
                        help='Maximum QoE value to include (inclusive).')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed for reproducible sampling (default: None).')
    parser.add_argument('--legacy_format', action='store_true',
                        help='Indicate that the dataset is in legacy format.')
    parser.add_argument('--stratified_sampling', action='store_true',
                        help='Enable stratified sampling across the QoE range.')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='Number of QoE range bins for stratified sampling (default: 10).')
    parser.add_argument('--sequence_based', action='store_true',
                        help='Create validation dataset based on sequences of chronological files.')
    parser.add_argument('--sequence_length', type=int, default=5,
                        help='Number of files in each sequence when using sequence-based validation (default: 5).')
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.random_seed is not None:
        random.seed(args.random_seed)
    
    # Create output directories
    ground_truth_dir, inference_dir = setup_directories(args.output_folder)
    
    # Create a metadata file to store ground truth QoE values
    metadata_path = os.path.join(args.output_folder, "validation_metadata.json")
    metadata = {
        "creation_timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "files": {},
        "format": "legacy" if args.legacy_format else "new",
        "sample_ratio": args.sample_ratio if args.num_files is None else None,
        "num_files": args.num_files,
        "qoe_min": args.qoe_min,
        "qoe_max": args.qoe_max,
        "random_seed": args.random_seed,
        "sampling_method": "stratified" if args.stratified_sampling else "random",
        "sequence_based": args.sequence_based,
        "sequence_length": args.sequence_length if args.sequence_based else None
    }
    
    # Get list of all JSON files
    all_files = [f for f in os.listdir(args.input_folder) if f.endswith('.json')]
    
    # First apply QoE range filtering if specified
    if args.qoe_min is not None or args.qoe_max is not None:
        all_files = filter_files_by_qoe_range(all_files, args.input_folder, args.qoe_min, args.qoe_max)
        if not all_files:
            print("No files match the specified QoE range criteria.")
            return
    
    # Determine files to process
    if args.sequence_based:
        if args.stratified_sampling:
            sequences = stratified_sequence_sampling(
                all_files, args.input_folder, args.num_files, args.sample_ratio,
                args.num_bins, args.sequence_length
            )
        else:
            # Create sequences then randomly sample them
            sequences = group_files_into_sequences(all_files, args.sequence_length)
            
            if args.num_files is not None:
                num_sequences = min(args.num_files, len(sequences))
            else:
                num_sequences = int(len(sequences) * args.sample_ratio)
                
            sequences = random.sample(sequences, num_sequences) if num_sequences < len(sequences) else sequences
        
        # Flatten sequences to get list of unique files to process
        selected_files = list(set([file for sequence in sequences for file in sequence]))
        
        # Store sequence information in metadata
        metadata["sequences"] = [{"files": sequence} for sequence in sequences]
    else:
        # Original single-file selection logic
        if args.num_files is not None or args.sample_ratio < 1.0:
            if args.stratified_sampling:
                selected_files = stratified_sampling(
                    all_files, args.input_folder, args.num_files, args.sample_ratio, args.num_bins
                )
            else:
                if args.num_files is not None:
                    num_files = min(args.num_files, len(all_files))
                else:
                    num_files = int(len(all_files) * args.sample_ratio)
                selected_files = random.sample(all_files, num_files)
        else:
            selected_files = all_files
    
    print(f"Preparing validation dataset with {len(selected_files)} files")
    
    # Print distribution of QoE values in selected files
    print_qoe_distribution(selected_files, args.input_folder)
    
    valid_file_count = 0
    invalid_file_count = 0
    
    # Process each file
    for filename in selected_files:
        src_path = os.path.join(args.input_folder, filename)
        
        # Check if file has valid structure for validation
        if validate_json_structure(src_path, args.legacy_format):
            # Copy original file to ground truth directory
            gt_path = os.path.join(ground_truth_dir, filename)
            shutil.copy2(src_path, gt_path)
            
            # Create modified version with QoE removed
            inf_path = os.path.join(inference_dir, filename)
            qoe_value = create_inference_copy(src_path, inf_path, args.legacy_format)
            
            # Store original QoE value in metadata
            if qoe_value is not None:
                metadata["files"][filename] = {
                    "ground_truth_qoe": qoe_value
                }
                valid_file_count += 1
        else:
            invalid_file_count += 1
    
    # If using sequences, calculate and store sequence QoE in metadata
    if args.sequence_based:
        for i, sequence in enumerate(sequences):
            last_file = sequence[-1]
            if last_file in metadata["files"]:
                qoe = metadata["files"][last_file]["ground_truth_qoe"]
                metadata["sequences"][i]["ground_truth_qoe"] = qoe
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Print summary
    print(f"Validation dataset preparation complete:")
    print(f"  - Valid files processed: {valid_file_count}")
    print(f"  - Invalid files skipped: {invalid_file_count}")
    if args.sequence_based:
        print(f"  - Sequences created: {len(sequences)}")
    print(f"  - Ground truth files saved to: {ground_truth_dir}")
    print(f"  - Inference files saved to: {inference_dir}")
    print(f"  - Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()