#!/usr/bin/env python3
"""
A simple script to test the linear model and generate feature importance data.
"""
import os
import sys
import json
import traceback
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from time import time

# Get paths from environment variables
models_dir = os.path.expanduser(os.environ.get('MODELS_DIR', '~/Impact-xG_prediction_model/forecasting_models_v5'))
data_folder = os.path.expanduser(os.environ.get('DATA_FOLDER', '~/Impact-xG_prediction_model/real_dataset'))
scaler_file = os.path.expanduser(os.environ.get('SCALER_FILE', os.path.join(models_dir, 'scaler.save')))
model_file = os.path.expanduser(os.environ.get('MODEL_FILE', os.path.join(models_dir, 'linear_basic.h5')))
feature_importance_dir = os.path.expanduser(os.environ.get('FEATURE_IMPORTANCE_DIR', os.path.join(models_dir, 'feature_importance')))
results_csv = os.path.expanduser(os.environ.get('RESULTS_CSV', os.path.join(models_dir, 'linear_model_results.csv')))

# Function to load augmented dataset
def load_augmented_dataset_from_folder(folder_path, use_stats=False, new_format=True):
    """
    Load all JSON files from the folder (augmented format) and return a DataFrame.
    """
    data = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            inner_keys = [k for k in json_data.keys() if k not in ["QoE", "timestamp"]]
            inner_keys = sorted(inner_keys)
            
            flat_features = []
            if use_stats:
                stats_features = {
                    "packet_loss_rate": [], "jitter": [], "throughput": [], 
                    "speed": [], "packets_lost": []
                }
            
            for key in inner_keys:
                entry = json_data[key]
                if new_format:
                    # New format includes "packets_lost"
                    flat_features.extend([
                        entry["throughput"], entry["packets_lost"], entry["packet_loss_rate"], 
                        entry["jitter"], entry["speed"]
                    ])
                    if use_stats:
                        stats_features["throughput"].append(entry["throughput"])
                        stats_features["packets_lost"].append(entry["packets_lost"])
                        stats_features["packet_loss_rate"].append(entry["packet_loss_rate"])
                        stats_features["jitter"].append(entry["jitter"])
                        stats_features["speed"].append(entry["speed"])
            
            sample = {}
            for i, val in enumerate(flat_features):
                sample[f"f{i}"] = val
            
            if use_stats:
                for feature in stats_features.keys():
                    arr = np.array(stats_features[feature])
                    sample[f"{feature}_mean"] = float(np.mean(arr))
                    sample[f"{feature}_std"] = float(np.std(arr))
                    sample[f"{feature}_min"] = float(np.min(arr))
                    sample[f"{feature}_max"] = float(np.max(arr))
            
            sample["QoE"] = json_data["QoE"]
            sample["timestamp"] = json_data["timestamp"]
            data.append(sample)
    
    data_sorted = sorted(data, key=lambda x: x["timestamp"])
    df = pd.DataFrame(data_sorted)
    return df

def preprocess_dataframe(df):
    """
    Convert columns to numeric types and convert 'timestamp' to a datetime object.
    """
    numeric_cols = [col for col in df.columns if col != 'timestamp']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle timestamp which might be an integer or string
    if df['timestamp'].dtype == object:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')
    else:
        # Convert integer timestamp to string first
        df['timestamp'] = df['timestamp'].astype(str)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')
    
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def create_sequences(df, seq_length=5, feature_cols=None, target_col='QoE'):
    """
    Build sequences of shape (seq_length, number_of_features) and corresponding targets.
    """
    X, y = [], []
    for i in range(len(df) - seq_length):
        seq_X = df.iloc[i:i+seq_length][feature_cols].values
        seq_y = df.iloc[i+seq_length][target_col]
        X.append(seq_X)
        y.append(seq_y)
    return np.array(X), np.array(y)

def measure_inference_latency(model, sample_input, num_runs=50):
    """
    Measure the average inference latency of the model.
    """
    # Warm-up run
    model.predict(sample_input)
    times = []
    for _ in range(num_runs):
        start_time = time()
        model.predict(sample_input)
        end_time = time()
        times.append(end_time - start_time)
    avg_latency = np.mean(times)
    return avg_latency * 1000  # Convert to milliseconds

def main():
    try:
        # Create feature importance directory if it doesn't exist
        os.makedirs(feature_importance_dir, exist_ok=True)
        
        print(f"Loading dataset from {data_folder}...")
        df = load_augmented_dataset_from_folder(data_folder, use_stats=True, new_format=True)
        
        # Get feature columns (all columns except QoE and timestamp)
        feature_cols = [col for col in df.columns if col not in ["QoE", "timestamp"]]
        print(f"Dataset loaded with {len(feature_cols)} features")
        
        # Preprocess the dataframe
        df = preprocess_dataframe(df)
        
        # Load the scaler and apply scaling
        print(f"Loading scaler from {scaler_file}...")
        scaler = joblib.load(scaler_file)
        
        # Define normalization columns (features + target)
        norm_cols = feature_cols + ["QoE"]
        df[norm_cols] = scaler.transform(df[norm_cols])
        
        # Create sequences
        print("Creating sequences...")
        X, y = create_sequences(df, seq_length=5, feature_cols=feature_cols, target_col='QoE')
        
        # Reserve the last 20% for testing
        test_size = int(len(X) * 0.2)
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        print(f"Test set size: {len(X_test)}")
        
        # Load the model
        print(f"Loading model from {model_file}...")
        model = tf.keras.models.load_model(model_file)
        
        # Evaluate the model
        print("Evaluating model...")
        test_loss = model.evaluate(X_test, y_test, verbose=1)
        
        # Make predictions
        predictions = model.predict(X_test, verbose=1)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Measure inference latency
        sample_input = X_test[0:1]
        latency = measure_inference_latency(model, sample_input)
        
        # Print metrics
        print("\nEvaluation Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R^2 Score: {r2:.4f}")
        print(f"Average Inference Latency: {latency:.2f} ms")
        
        # Extract feature importance for linear model
        print("Extracting feature importance...")
        weights = model.layers[-1].get_weights()[0]
        bias = model.layers[-1].get_weights()[1]
        
        # Create feature names
        flattened_feature_names = []
        for t in range(5):  # seq_length
            for feature in feature_cols:
                flattened_feature_names.append(f"{feature}_t-{5-t}")
        
        # Create feature importance data
        feature_importance = sorted(zip(flattened_feature_names, weights.flatten()), 
                                  key=lambda x: abs(x[1]), reverse=True)
        
        # Print top features
        print("\nTop 10 Features by Importance:")
        for feature, weight in feature_importance[:10]:
            print(f"{feature}: {weight:.4f}")
        
        # Save feature importance to file
        model_name = os.path.basename(model_file)
        feature_importance_data = {
            "model_name": model_name,
            "model_type": "Linear Regressor",
            "feature_importance": [{"feature": f, "weight": float(w)} for f, w in feature_importance],
            "bias": float(bias[0]),
            "feature_count": len(feature_cols),
            "sequence_length": 5
        }
        
        output_file = os.path.join(feature_importance_dir, f"{model_name}.importance.json")
        with open(output_file, 'w') as f:
            json.dump(feature_importance_data, f, indent=2)
        
        print(f"Feature importance saved to: {output_file}")
        
        # Save results to CSV
        with open(results_csv, 'w') as f:
            f.write("Model,Model Type,MSE,MAE,R2,Inference Latency (ms)\n")
            f.write(f"{model_name},Linear Regressor,{mse:.4f},{mae:.4f},{r2:.4f},{latency:.2f}\n")
        
        print(f"Results saved to: {results_csv}")
        print("Test completed successfully!")
        
        return 0
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
