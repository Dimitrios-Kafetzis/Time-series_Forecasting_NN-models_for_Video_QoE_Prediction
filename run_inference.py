#!/usr/bin/env python3
"""
Filename: run_inference_v2.py
Author: Dimitrios Kafetzis (Modified)
Creation Date: 2025-05-21
Description:
    This script loads a saved trained model and scaler, loads a new inference JSON file,
    and performs inference by constructing the required input sequence.
    It supports two dataset formats:
      - Standard: one JSON file per 10‑second timepoint with flat structure.
      - Augmented: one JSON file per 5‑second window containing per‑second sub‐records and
                   an aggregated QoE (the overall timestamp is that of the last second in the window).
    
    This version properly handles both TransformerBlock and SelfAttention custom layers.

Usage Examples:
    Standard mode:
      $ python3 run_inference_v2.py --inference_file ./inference_inputs/20250204123000.json \
          --data_folder ./mock_dataset --seq_length 5 --model_file model_transformer.h5 --scaler_file scaler.save

    Augmented mode:
      $ python3 run_inference_v2.py --inference_file ./inference_inputs/20250204123000.json \
          --data_folder ./augmented_dataset --seq_length 5 --model_file model_transformer.h5 --scaler_file scaler.save --augmented
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# Import the custom TransformerBlock and SelfAttention (needed when loading the model)
try:
    # Try to import from timeseries_forecasting_models_v6 first
    from timeseries_forecasting_models_v6 import TransformerBlock, SelfAttention
    print("Using custom layers from timeseries_forecasting_models_v6")
except ImportError:
    try:
        # Fall back to timeseries_forecasting_models_v5
        from timeseries_forecasting_models_v5 import TransformerBlock, SelfAttention
        print("Using custom layers from timeseries_forecasting_models_v5")
    except ImportError:
        # Define the layers directly here as fallback
        print("Could not import custom layers from models - defining them locally")
        
        # TransformerBlock implementation
        class TransformerBlock(tf.keras.layers.Layer):
            def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
                super(TransformerBlock, self).__init__(**kwargs)
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.ff_dim = ff_dim
                self.rate = rate
                self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
                self.ffn = tf.keras.Sequential([
                    tf.keras.layers.Dense(ff_dim, activation="relu"),
                    tf.keras.layers.Dense(embed_dim),
                ])
                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = tf.keras.layers.Dropout(rate)
                self.dropout2 = tf.keras.layers.Dropout(rate)

            def call(self, inputs, training=False):
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)
            
            def get_config(self):
                config = super(TransformerBlock, self).get_config()
                config.update({
                    "embed_dim": self.embed_dim,
                    "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim,
                    "rate": self.rate,
                })
                return config
                
        # SelfAttention implementation
        class SelfAttention(tf.keras.layers.Layer):
            def __init__(self, attention_units=128, **kwargs):
                self.attention_units = attention_units
                super(SelfAttention, self).__init__(**kwargs)
                
            def build(self, input_shape):
                # input_shape = (batch_size, time_steps, features)
                self.time_steps = input_shape[1]
                self.features = input_shape[2]
                
                # Dense layer to compute attention scores
                self.W_attention = self.add_weight(name='W_attention',
                                                shape=(self.features, self.attention_units),
                                                initializer='glorot_uniform',
                                                trainable=True)
                
                self.b_attention = self.add_weight(name='b_attention',
                                                shape=(self.attention_units,),
                                                initializer='zeros',
                                                trainable=True)
                
                # Context vector to compute attention weights
                self.u_attention = self.add_weight(name='u_attention',
                                                shape=(self.attention_units, 1),
                                                initializer='glorot_uniform',
                                                trainable=True)
                
                super(SelfAttention, self).build(input_shape)
            
            def call(self, inputs):
                # inputs shape: (batch_size, time_steps, features)
                
                # Step 1: Compute attention scores
                # (batch_size, time_steps, features) @ (features, attention_units) = (batch_size, time_steps, attention_units)
                score = tf.tanh(tf.tensordot(inputs, self.W_attention, axes=[[2], [0]]) + self.b_attention)
                
                # Step 2: Compute attention weights
                # (batch_size, time_steps, attention_units) @ (attention_units, 1) = (batch_size, time_steps, 1)
                attention_weights = tf.nn.softmax(tf.tensordot(score, self.u_attention, axes=[[2], [0]]), axis=1)
                
                # Step 3: Apply attention weights to input sequence
                # (batch_size, time_steps, 1) * (batch_size, time_steps, features) = (batch_size, time_steps, features)
                context_vector = attention_weights * inputs
                
                # Step 4: Sum over time dimension to get weighted representation
                # (batch_size, features)
                context_vector = tf.reduce_sum(context_vector, axis=1)
                
                return context_vector
            
            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[2])
            
            def get_config(self):
                config = super(SelfAttention, self).get_config()
                config.update({
                    'attention_units': self.attention_units,
                })
                return config

def load_dataset_from_folder(folder_path):
    """
    Load all JSON files from the folder assuming standard format:
      {
          "packet_loss_rate": <float>,
          "jitter": <float>,
          "throughput": <float>,
          "speed": <float>,
          "QoE": <float>,
          "timestamp": "YYYYMMDDHHMMSS"
      }
    Returns a DataFrame.
    """
    data = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                data.append(json_data)
    data_sorted = sorted(data, key=lambda x: x['timestamp'])
    df = pd.DataFrame(data_sorted)
    return df

def load_augmented_dataset_from_folder(folder_path):
    """
    Load all JSON files from the folder assuming augmented format.
    Each JSON file is expected to have a structure like:
    {
      "<timestamp1>": { "packet_loss_rate": <float>, "jitter": <float>, "throughput": <float>, "speed": <float> },
      "<timestamp2>": { ... },
      ...
      "<timestampN>": { ... },
      "QoE": <float>,         # aggregated QoE for the window
      "timestamp": "YYYYMMDDHHMMSS"  # overall timestamp 
    }
    For each file, this function flattens features into f0, f1, f2, ... format.
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
            for key in inner_keys:
                entry = json_data[key]
                # New format includes "packets_lost"
                if isinstance(entry, dict):
                    if "packets_lost" in entry:
                        flat_features.extend([
                            entry["throughput"], entry["packets_lost"], entry["packet_loss_rate"], 
                            entry["jitter"], entry["speed"]
                        ])
                    else:
                        flat_features.extend([
                            entry["packet_loss_rate"], entry["jitter"], entry["throughput"], entry["speed"]
                        ])
            
            sample = {}
            for i, val in enumerate(flat_features):
                sample[f"f{i}"] = val
            
            sample["QoE"] = json_data["QoE"]
            sample["timestamp"] = json_data["timestamp"]
            data.append(sample)
    
    data_sorted = sorted(data, key=lambda x: x["timestamp"])
    df = pd.DataFrame(data_sorted)
    return df

def load_inference_file(file_path, augmented=False, use_stats=False):
    """
    Loads a new inference JSON file and parses it according to the format.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not augmented:
        # Standard mode: simply return the 4 feature values as a dict
        return {
            "packet_loss_rate": data["packet_loss_rate"],
            "jitter": data["jitter"],
            "throughput": data["throughput"],
            "speed": data["speed"]
        }
    else:
        # Augmented mode: flatten into f0, f1, f2, ... format
        inner_keys = [k for k in data.keys() if k not in ["QoE", "timestamp"]]
        inner_keys = sorted(inner_keys)
        
        flat_features = []
        stats_features = None
        
        if use_stats:
            # If stats are used, track metric values
            stats_features = {
                "packet_loss_rate": [], "jitter": [], "throughput": [], 
                "speed": [], "packets_lost": []
            }
        
        for key in inner_keys:
            entry = data[key]
            # Check if entry has the packets_lost field (new format)
            if isinstance(entry, dict):
                if "packets_lost" in entry:
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
                else:
                    flat_features.extend([
                        entry["packet_loss_rate"], entry["jitter"], entry["throughput"], entry["speed"]
                    ])
                    if use_stats:
                        stats_features["packet_loss_rate"].append(entry["packet_loss_rate"])
                        stats_features["jitter"].append(entry["jitter"])
                        stats_features["throughput"].append(entry["throughput"])
                        stats_features["speed"].append(entry["speed"])
        
        # Create the features dictionary
        result = {}
        for i, val in enumerate(flat_features):
            result[f"f{i}"] = val
        
        # Add statistics if needed
        if use_stats and stats_features:
            for feature, values in stats_features.items():
                if values:
                    arr = np.array(values)
                    result[f"{feature}_mean"] = float(np.mean(arr))
                    result[f"{feature}_std"] = float(np.std(arr))
                    result[f"{feature}_min"] = float(np.min(arr))
                    result[f"{feature}_max"] = float(np.max(arr))
        
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_file", type=str, required=True,
                        help="Path to the new inference JSON file.")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to folder containing the training dataset (to retrieve previous records).")
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Sequence length (number of time steps used as input).")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the saved trained model file.")
    parser.add_argument("--scaler_file", type=str, required=True,
                        help="Path to the saved scaler file.")
    parser.add_argument("--augmented", action="store_true",
                        help="Flag to indicate that the dataset is in augmented mode.")
    parser.add_argument("--use_stats", action="store_true",
                        help="Include extra statistical features (must match model training configuration).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print additional debug information.")
    
    args = parser.parse_args()

    if args.verbose:
        print(f"Loading model: {args.model_file}")
        print(f"Loading scaler: {args.scaler_file}")
        print(f"Inference file: {args.inference_file}")
        print(f"Data folder: {args.data_folder}")
        print(f"Sequence length: {args.seq_length}")
        print(f"Augmented mode: {args.augmented}")
        print(f"Using stats: {args.use_stats}")

    # Load the saved model and scaler with proper custom objects
    try:
        # Define custom objects for model loading
        custom_objects = {
            "TransformerBlock": TransformerBlock,
            "SelfAttention": SelfAttention
        }
        
        model = tf.keras.models.load_model(
            args.model_file,
            custom_objects=custom_objects
        )
        
        if args.verbose:
            print("Successfully loaded model")
            model.summary()
            
        scaler = joblib.load(args.scaler_file)
        if args.verbose:
            print("Successfully loaded scaler")
    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")
        return

    # Depending on the mode, load the training dataset appropriately.
    try:
        if args.augmented:
            df = load_augmented_dataset_from_folder(args.data_folder)
        else:
            df = load_dataset_from_folder(args.data_folder)
        
        # Ensure the dataframe is sorted by timestamp.
        df.sort_values("timestamp", inplace=True)
        
        if args.verbose:
            print(f"Loaded dataset with {len(df)} records")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Define feature columns based on the mode
    if args.augmented:
        feature_cols = [col for col in df.columns if col not in ["QoE", "timestamp"]]
    else:
        feature_cols = ['packet_loss_rate', 'jitter', 'throughput', 'speed']
    
    if args.verbose:
        print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # For inference, we need the last (seq_length - 1) records from the training dataset.
    if len(df) < args.seq_length - 1:
        print(f"Error: Not enough records in dataset. Need at least {args.seq_length - 1} records but found {len(df)}")
        return
        
    last_records = df.iloc[-(args.seq_length - 1):][feature_cols].values
    if args.verbose:
        print(f"Using {len(last_records)} previous records for sequence construction")

    # Load the new inference file
    try:
        new_record_features = load_inference_file(args.inference_file, augmented=args.augmented, use_stats=args.use_stats)
        
        if args.verbose:
            print("Loaded inference file")
            print(f"Features: {new_record_features}")
        
        # Extract features in the right order
        new_data = []
        for feature in feature_cols:
            new_data.append(new_record_features[feature])
        
        new_data = np.array([new_data])
        
        # The scaler was fitted on features + QoE; add a dummy for QoE.
        dummy = np.zeros((new_data.shape[0], 1))
        new_record_full = np.hstack([new_data, dummy])
        
        # Transform using the scaler
        new_record_scaled = scaler.transform(new_record_full)[:, :len(feature_cols)]
        
        if args.verbose:
            print("Successfully scaled the inference data")
    except Exception as e:
        print(f"Error processing inference file: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Form the full input sequence by stacking the last records and the new record.
    try:
        sequence = np.vstack([last_records, new_record_scaled])
        sequence = sequence.reshape(1, args.seq_length, len(feature_cols))
        
        if args.verbose:
            print(f"Input sequence shape: {sequence.shape}")
    except Exception as e:
        print(f"Error creating input sequence: {str(e)}")
        return

    # Predict the QoE (prediction is in normalized scale).
    try:
        predicted_qoe_scaled = model.predict(sequence)
        
        # To convert the predicted QoE back to the original scale, create a dummy array.
        dummy_array = np.zeros((1, len(feature_cols) + 1))  # +1 for QoE
        dummy_array[0, -1] = predicted_qoe_scaled[0, 0]
        predicted_qoe = scaler.inverse_transform(dummy_array)[0, -1]
        
        print(f"Predicted QoE: {predicted_qoe:.4f}")
        # Print the prediction again in a consistent format that's easier to parse
        print(f"RESULT_MARKER:{predicted_qoe:.6f}")
        return predicted_qoe
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return

if __name__ == "__main__":
    main()
