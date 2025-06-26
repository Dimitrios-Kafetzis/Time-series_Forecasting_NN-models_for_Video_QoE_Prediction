#!/usr/bin/env python3
"""Validation dataset test script"""

import os
import sys
import json

def main():
    validation_folder = "./validation_dataset"
    
    # Check if the validation folder exists
    print(f"Checking if validation folder exists: {validation_folder}")
    if not os.path.exists(validation_folder):
        print(f"FAILED: Validation folder {validation_folder} does not exist")
        return
    print("SUCCESS: Validation folder exists")
    
    # Check for required subdirectories
    ground_truth_dir = os.path.join(validation_folder, "ground_truth")
    inference_dir = os.path.join(validation_folder, "inference")
    metadata_path = os.path.join(validation_folder, "validation_metadata.json")
    
    print(f"Checking ground_truth directory: {ground_truth_dir}")
    if not os.path.exists(ground_truth_dir):
        print(f"FAILED: ground_truth directory not found")
        return
    print("SUCCESS: ground_truth directory exists")
    
    print(f"Checking inference directory: {inference_dir}")
    if not os.path.exists(inference_dir):
        print(f"FAILED: inference directory not found")
        return
    print("SUCCESS: inference directory exists")
    
    print(f"Checking metadata file: {metadata_path}")
    if not os.path.exists(metadata_path):
        print(f"FAILED: validation_metadata.json not found")
        return
    print("SUCCESS: metadata file exists")
    
    # Try to load the metadata file
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"SUCCESS: Metadata loaded with {len(metadata.get('files', {}))} files")
    except Exception as e:
        print(f"FAILED: Error loading metadata: {str(e)}")
        return
    
    # Check if there are any files in the directories
    ground_truth_files = [f for f in os.listdir(ground_truth_dir) if f.endswith('.json')]
    inference_files = [f for f in os.listdir(inference_dir) if f.endswith('.json')]
    
    print(f"Found {len(ground_truth_files)} files in ground_truth directory")
    print(f"Found {len(inference_files)} files in inference directory")
    
    # Check if the first few files in metadata actually exist
    print("\nChecking files referenced in metadata:")
    count = 0
    for filename in list(metadata.get('files', {}).keys())[:5]:
        gt_path = os.path.join(ground_truth_dir, filename)
        inf_path = os.path.join(inference_dir, filename)
        
        print(f"File: {filename}")
        print(f"  Ground truth exists: {os.path.exists(gt_path)}")
        print(f"  Inference exists: {os.path.exists(inf_path)}")
        count += 1
        
    if count == 0:
        print("No files found in metadata")

if __name__ == "__main__":
    main()