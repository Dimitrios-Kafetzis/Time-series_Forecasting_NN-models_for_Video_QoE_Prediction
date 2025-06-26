#!/usr/bin/env python3
"""Debug script for validation system"""

import os
import sys
import json
import traceback
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# Import the specific functions we need
try:
    from timeseries_forecasting_models_v5 import (
        TransformerBlock, SelfAttention, 
        load_augmented_dataset_from_folder,
        preprocess_dataframe
    )
    print("Successfully imported from timeseries_forecasting_models_v5")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def step1_test_imports():
    print("\n=== STEP 1: Testing imports ===")
    print("All imports loaded successfully")
    return True

def step2_test_file_loading(validation_folder):
    print("\n=== STEP 2: Testing validation data loading ===")
    inference_dir = os.path.join(validation_folder, "inference")
    
    # Load metadata
    try:
        metadata_path = os.path.join(validation_folder, "validation_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata loaded with {len(metadata.get('files', {}))} files")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None, None
    
    # Determine legacy format
    legacy_format = metadata.get("format", "new") == "legacy"
    use_stats = True  # We'll try with stats enabled
    
    # Try to load the first dataset
    try:
        print(f"Loading data from {inference_dir} (legacy_format={legacy_format}, use_stats={use_stats})")
        df = load_augmented_dataset_from_folder(inference_dir, use_stats=use_stats, new_format=not legacy_format)
        print(f"Data loaded successfully, shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df, metadata
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        return None, None

def step3_test_preprocessing(df):
    print("\n=== STEP 3: Testing preprocessing ===")
    if df is None:
        print("Error: No dataframe to preprocess")
        return None
        
    try:
        df_processed = preprocess_dataframe(df)
        print(f"Preprocessing successful, shape: {df_processed.shape}")
        return df_processed
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        traceback.print_exc()
        return None

def step4_test_scaler(df, scaler_path):
    print("\n=== STEP 4: Testing scaler ===")
    if df is None:
        print("Error: No dataframe to scale")
        return None, None, None
        
    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in ["QoE", "timestamp"]]
        print(f"Feature columns: {len(feature_cols)} columns")
        
        # Define normalization columns
        norm_cols = feature_cols + ["QoE"]
        
        # Apply scaler transformation
        print(f"Applying scaler to {len(norm_cols)} columns")
        print(f"First few norm_cols: {norm_cols[:5]}")
        print(f"Column dtypes before scaling: {df[norm_cols].dtypes}")
        
        df[norm_cols] = scaler.transform(df[norm_cols])
        print("Scaling successful")
        return df, feature_cols, norm_cols
    except Exception as e:
        print(f"Error in scaling: {e}")
        traceback.print_exc()
        return None, None, None

def step5_test_model_loading(model_path):
    print("\n=== STEP 5: Testing model loading ===")
    custom_objects = {"TransformerBlock": TransformerBlock, "SelfAttention": SelfAttention}
    
    try:
        print(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully")
        print(f"Model type: {model.__class__.__name__}")
        model.summary()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def main():
    # Parameters
    validation_folder = "./validation_dataset"
    model_dir = "./forecasting_models_v5"
    scaler_path = "./forecasting_models_v5/scaler.save"
    
    # Test imports
    if not step1_test_imports():
        return
    
    # Test file loading
    df, metadata = step2_test_file_loading(validation_folder)
    if df is None:
        print("File loading failed")
        return
    
    # Test preprocessing
    df_processed = step3_test_preprocessing(df)
    if df_processed is None:
        print("Preprocessing failed")
        return
    
    # Test scaler
    df_scaled, feature_cols, norm_cols = step4_test_scaler(df_processed, scaler_path)
    if df_scaled is None:
        print("Scaling failed")
        return
    
    # Find first model to test
    model_file = None
    for filename in os.listdir(model_dir):
        if filename.endswith('.h5'):
            model_file = os.path.join(model_dir, filename)
            break
    
    if not model_file:
        print(f"No model files found in {model_dir}")
        return
    
    # Test model loading
    model = step5_test_model_loading(model_file)
    if model is None:
        print("Model loading failed")
        return
    
    # Test sequence generation
    print("\n=== STEP 6: Testing sequence generation ===")
    try:
        # Create a list of filenames ordered by timestamp
        filenames = [f for f in os.listdir(os.path.join(validation_folder, "inference")) if f.endswith('.json')]
        filenames = sorted(filenames)
        print(f"Found {len(filenames)} files")
        
        # Extract timestamps for first few files
        print("First few filenames:")
        for f in filenames[:5]:
            print(f"  {f}")
            
        # Create sequences for testing
        from timeseries_forecasting_models_v5 import create_sequences
        seq_length = 5
        
        print(f"Testing create_sequences function with seq_length={seq_length}")
        X, y = create_sequences(df_scaled, seq_length=seq_length, 
                               feature_cols=feature_cols, target_col='QoE')
        print(f"Created sequences, X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        print(f"Error in sequence generation: {e}")
        traceback.print_exc()
        return
    
    print("\n=== ALL TESTS PASSED ===")
    print("Each component of the validation system is working correctly.")
    print("Continue with running the full validate_models.py script.")

if __name__ == "__main__":
    main()