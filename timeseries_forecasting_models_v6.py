#!/usr/bin/env python3
"""
Filename: timeseries_forecasting_models_v6.py
Author: Dimitrios Kafetzis (Modified)
Creation Date: 2025-05-19
Description:
    This script implements deep learning models for time series forecasting using TensorFlow.
    It includes five model architectures:
        - Linear Regressor
        - Simple DNN
        - LSTM
        - GRU
        - Transformer
    The purpose is to predict the QoE (Quality of Experience) value for network data.
    The input dataset is composed of JSON files. There are two supported formats:
    
      1. Regular (Legacy):
         Each file corresponds to a 10-second interval and contains the fields:
             "packet_loss_rate", "jitter", "throughput", "speed", "QoE", "timestamp",
             and additional temporal features: "hour", "minute", "day_of_week".
      2. Augmented (Default):
         Each file corresponds to a 10-second window with 2-second interval measurements:
         {
             "QoE": <float>,
             "timestamp": <int>,
             "<timestamp_1>": {
                 "throughput": <float>,
                 "packets_lost": <float>,
                 "packet_loss_rate": <float>,
                 "jitter": <float>,
                 "speed": <float>
             },
             "<timestamp_2>": { ... },
             ...
         }
    
    Modifications in v6:
    - Added support for loading data from separate train, validation, and test directories
    - Made the output directory configurable via command line argument
    - Adapted the training process to use dedicated validation data
    - Preserved all existing functionality including different model architectures, 
      hyperparameter tuning, and data format support
    
    Usage Examples:
      1. Train a Linear Regressor model on a regular dataset:
         $ python3 timeseries_forecasting_models_v6.py --train_dir ./train_set --val_dir ./validation_set --test_dir ./test_set --model_type linear --seq_length 5 --epochs 20 --batch_size 16 --output_dir ./model_outputs

      2. Train a Simple DNN model on the new augmented dataset (default format):
         $ python3 timeseries_forecasting_models_v6.py --train_dir ./train_set --val_dir ./validation_set --test_dir ./test_set --model_type dnn --seq_length 5 --epochs 20 --batch_size 16 --augmented --output_dir ./model_outputs

      3. Train an LSTM model with self-attention on a regular dataset:
         $ python3 timeseries_forecasting_models_v6.py --train_dir ./train_set --val_dir ./validation_set --test_dir ./test_set --model_type lstm --seq_length 5 --epochs 20 --batch_size 16 --attention_units 128 --output_dir ./model_outputs

      4. Train a GRU model with self-attention on the new augmented dataset:
         $ python3 timeseries_forecasting_models_v6.py --train_dir ./train_set --val_dir ./validation_set --test_dir ./test_set --model_type gru --seq_length 5 --epochs 20 --batch_size 16 --augmented --attention_units 128 --output_dir ./model_outputs

      5. Train a model on legacy format data (if needed):
         $ python3 timeseries_forecasting_models_v6.py --train_dir ./train_set --val_dir ./validation_set --test_dir ./test_set --model_type lstm --seq_length 5 --epochs 20 --batch_size 16 --augmented --legacy_format --output_dir ./model_outputs
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (LSTM, GRU, Dense, Input, Flatten, Dropout,
                                     LayerNormalization, MultiHeadAttention, Bidirectional,
                                     Attention, GlobalAveragePooling1D, Reshape, Permute,
                                     TimeDistributed, Lambda, Activation, RepeatVector, 
                                     Concatenate, multiply)
import tensorflow.keras.backend as K

# For automated tuning
import keras_tuner as kt

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================

def load_dataset_from_folder(folder_path):
    """
    Load all JSON files from the folder (regular format) and return a DataFrame.
    Each JSON file is expected to have the keys:
      "packet_loss_rate", "jitter", "throughput", "speed", "QoE", "timestamp",
      and (optionally) additional temporal fields ("hour", "minute", "day_of_week").
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

def load_augmented_dataset_from_folder(folder_path, use_stats=False, new_format=False):
    """
    Load all JSON files from the folder (augmented format) and return a DataFrame.
    
    For legacy augmented format:
      In each JSON file, the keys that are not "QoE" or "timestamp" represent 1-second measurements.
      These are sorted and flattened into feature columns f0, f1, ..., f19 (for 5 seconds × 4 features).
    
    For new augmented format:
      Each file corresponds to a 10-second window with 2-second interval measurements:
      {
          "QoE": <float>,
          "timestamp": <int>,
          "<timestamp_1>": {
              "throughput": <float>,
              "packets_lost": <float>,
              "packet_loss_rate": <float>,
              "jitter": <float>,
              "speed": <float>
          },
          "<timestamp_2>": { ... },
          ...
      }
      These are flattened into feature columns f0, f1, ..., f24 (for 5 timestamps × 5 features).
      
    If use_stats is True, additional statistics (mean, std, min, max for each feature) are computed.
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
                # For the new format, track metrics differently
                if new_format:
                    stats_features = {
                        "packet_loss_rate": [], "jitter": [], "throughput": [], 
                        "speed": [], "packets_lost": []
                    }
                else:
                    stats_features = {
                        "packet_loss_rate": [], "jitter": [], "throughput": [], "speed": []
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
                else:
                    # Legacy format
                    flat_features.extend([
                        entry["packet_loss_rate"], entry["jitter"], entry["throughput"], entry["speed"]
                    ])
                    if use_stats:
                        stats_features["packet_loss_rate"].append(entry["packet_loss_rate"])
                        stats_features["jitter"].append(entry["jitter"])
                        stats_features["throughput"].append(entry["throughput"])
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
    The target is the QoE value at the time step immediately after the sequence.
    """
    X, y = [], []
    for i in range(len(df) - seq_length):
        seq_X = df.iloc[i:i+seq_length][feature_cols].values
        seq_y = df.iloc[i+seq_length][target_col]
        X.append(seq_X)
        y.append(seq_y)
    return np.array(X), np.array(y)

# =============================================================================
# 2. Self-Attention Layer Implementation
# =============================================================================

class SelfAttention(tf.keras.layers.Layer):
    """
    Custom Self-Attention Layer
    This layer applies attention over the time steps of a sequence, allowing the model
    to focus on the most relevant parts of the time series for prediction.
    """
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

# =============================================================================
# 3. Model Definitions (Including New Linear and DNN Models)
# =============================================================================

def build_linear_model(seq_length, feature_dim, learning_rate=0.001, l1_reg=0.0, l2_reg=0.0):
    """
    Build and compile a Linear Regression model.
    The model flattens the sequence input and applies a single Dense layer with no activation.
    Regularization can be applied to prevent overfitting.
    """
    # Create the model
    model = Sequential()
    model.add(Input(shape=(seq_length, feature_dim)))
    # Flatten the sequence to treat it as a single feature vector
    model.add(Flatten())
    # Add a single dense layer with no activation (linear regression)
    # Apply L1 and L2 regularization if specified
    model.add(Dense(1, activation=None, 
                  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
    
    # Compile the model with MSE loss (standard for regression)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.MeanSquaredError())
    return model

def build_dnn_model(seq_length, feature_dim, hidden_layers=[64, 32], dropout_rate=0.2, 
                   activation='relu', learning_rate=0.001, l2_reg=0.0):
    """
    Build and compile a simple DNN model.
    The model flattens the sequence input and applies multiple Dense layers.
    
    Parameters:
    - seq_length: Length of input sequence
    - feature_dim: Number of features per time step
    - hidden_layers: List of neurons in each hidden layer
    - dropout_rate: Dropout rate for regularization
    - activation: Activation function for hidden layers
    - learning_rate: Learning rate for optimizer
    - l2_reg: L2 regularization factor
    """
    # Create the model
    model = Sequential()
    model.add(Input(shape=(seq_length, feature_dim)))
    # Flatten the sequence
    model.add(Flatten())
    
    # Add hidden layers with the specified number of neurons
    for i, units in enumerate(hidden_layers):
        # Add dense layer with L2 regularization
        model.add(Dense(units, activation=activation, 
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        # Add dropout after each hidden layer
        model.add(Dropout(dropout_rate))
    
    # Output layer (single neuron for regression)
    model.add(Dense(1))
    
    # Compile the model with log-cosh loss (same as other models for consistency)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.log_cosh)
    return model

def build_lstm_model(seq_length, feature_dim, hidden_units=50, num_layers=2, dropout_rate=0.2,
                     bidirectional=False, learning_rate=0.001, attention_units=128):
    """
    Build and compile an LSTM model with self-attention.
    Modifications:
      - All recurrent layers output sequences (return_sequences=True).
      - Self-attention is used instead of GlobalAveragePooling1D to aggregate temporal information.
      - log-cosh loss is used.
    """
    model = Sequential()
    for i in range(num_layers):
        # Force return_sequences=True for all layers
        return_seq = True
        if i == 0:
            if bidirectional:
                lstm_layer = LSTM(hidden_units, return_sequences=return_seq,
                                  dropout=dropout_rate, recurrent_dropout=dropout_rate,
                                  input_shape=(seq_length, feature_dim))
                lstm_layer = Bidirectional(lstm_layer)
            else:
                lstm_layer = LSTM(hidden_units, return_sequences=return_seq,
                                  dropout=dropout_rate, recurrent_dropout=dropout_rate,
                                  input_shape=(seq_length, feature_dim))
        else:
            rnn = LSTM(hidden_units, return_sequences=return_seq,
                       dropout=dropout_rate, recurrent_dropout=dropout_rate)
            lstm_layer = Bidirectional(rnn) if bidirectional else rnn
        model.add(lstm_layer)
    
    # Replace GlobalAveragePooling1D with self-attention
    model.add(SelfAttention(attention_units=attention_units))
    
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.log_cosh)
    return model

def build_gru_model(seq_length, feature_dim, hidden_units=50, num_layers=2, dropout_rate=0.2,
                    bidirectional=False, learning_rate=0.001, attention_units=128):
    """
    Build and compile a GRU model with self-attention.
    Modifications:
      - All recurrent layers output sequences (return_sequences=True).
      - Self-attention is used instead of GlobalAveragePooling1D to aggregate temporal information.
      - log-cosh loss is used.
    """
    model = Sequential()
    for i in range(num_layers):
        return_seq = True
        if i == 0:
            if bidirectional:
                gru_layer = GRU(hidden_units, return_sequences=return_seq,
                                dropout=dropout_rate, recurrent_dropout=dropout_rate,
                                input_shape=(seq_length, feature_dim))
                gru_layer = Bidirectional(gru_layer)
            else:
                gru_layer = GRU(hidden_units, return_sequences=return_seq,
                                dropout=dropout_rate, recurrent_dropout=dropout_rate,
                                input_shape=(seq_length, feature_dim))
        else:
            rnn = GRU(hidden_units, return_sequences=return_seq,
                      dropout=dropout_rate, recurrent_dropout=dropout_rate)
            gru_layer = Bidirectional(rnn) if bidirectional else rnn
        model.add(gru_layer)
    
    # Replace GlobalAveragePooling1D with self-attention
    model.add(SelfAttention(attention_units=attention_units))
    
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.log_cosh)
    return model

# TransformerBlock class is defined below.
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

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

def build_transformer_model(seq_length, feature_dim, num_heads=2, ff_dim=64, dropout_rate=0.1, learning_rate=0.001):
    """
    Build and compile a Transformer model for time series forecasting.
    We use log-cosh loss here as well.
    """
    inputs = Input(shape=(seq_length, feature_dim))
    transformer_block = TransformerBlock(embed_dim=feature_dim, num_heads=num_heads, ff_dim=ff_dim, rate=dropout_rate)
    x = transformer_block(inputs)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.log_cosh)
    return model

# =============================================================================
# 4. Main Routine: Training, Evaluation, Automated Tuning, and Inference
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    # Dataset directory arguments (new)
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path to folder containing JSON files for training.")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to folder containing JSON files for validation.")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to folder containing JSON files for testing.")
    parser.add_argument("--output_dir", type=str, default="forecasting_models_v6",
                        help="Directory to save trained models and scalers.")
    
    # Model and training parameters
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["linear", "dnn", "lstm", "gru", "transformer"],
                        help="Type of model to train: linear, dnn, lstm, gru, or transformer.")
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Sequence length (number of time steps used as input).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer.")
    
    # Parameters specific to Linear model
    parser.add_argument("--l1_reg", type=float, default=0.0, help="L1 regularization for linear model.")
    parser.add_argument("--l2_reg", type=float, default=0.0, help="L2 regularization for linear and DNN models.")
    
    # Parameters specific to DNN model
    parser.add_argument("--hidden_layers", type=str, default="64,32", 
                        help="Comma-separated list of neurons in each hidden layer (DNN model).")
    parser.add_argument("--activation", type=str, default="relu", 
                        help="Activation function for hidden layers (DNN model).")
    
    # Parameters specific to RNN models
    parser.add_argument("--hidden_units", type=int, default=50, help="Number of hidden units in RNN layers.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of stacked recurrent layers.")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for recurrent layers.")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional RNN layers.")
    parser.add_argument("--attention_units", type=int, default=128, help="Number of units in self-attention layer.")
    
    # Parameters specific to Transformer model
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads for Transformer.")
    parser.add_argument("--ff_dim", type=int, default=64, help="Feed-forward dimension for Transformer.")
    
    # Hyperparameter tuning options
    parser.add_argument("--tune", action="store_true", help="Enable automated hyperparameter tuning using Keras Tuner.")
    parser.add_argument("--max_trials", type=int, default=10, help="Maximum number of trials for tuning.")
    parser.add_argument("--tune_epochs", type=int, default=20, help="Number of epochs to train each trial during tuning.")
    
    # Dataset format options
    parser.add_argument("--augmented", action="store_true", help="Indicate that the dataset is in augmented mode.")
    parser.add_argument("--use_stats", action="store_true", help="Include extra statistical features from the augmented data.")
    parser.add_argument("--legacy_format", action="store_true", 
                       help="Use the legacy dataset format (5-second windows with 1-second intervals)")
    parser.add_argument("--new_format", action="store_true", 
                       help="DEPRECATED: New format is now the default. This flag has no effect.")
    
    args = parser.parse_args()
    
    # Parse the hidden layers argument for DNN
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
    
    # Function to load and process data from a directory
    def load_data_from_directory(directory):
        if args.augmented:
            print(f"Loading augmented dataset from: {directory}")
            print(f"Using {'legacy' if args.legacy_format else 'new'} format")
            df = load_augmented_dataset_from_folder(directory, use_stats=args.use_stats, new_format=not args.legacy_format)
        else:
            print(f"Loading regular dataset from: {directory}")
            df = load_dataset_from_folder(directory)
        
        df = preprocess_dataframe(df)
        return df
    
    # Load datasets from each directory
    train_df = load_data_from_directory(args.train_dir)
    val_df = load_data_from_directory(args.val_dir)
    test_df = load_data_from_directory(args.test_dir)
    
    # Combine train and validation data for feature scaling
    # This ensures consistent scaling across all datasets
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Print data info
    print(f"Training set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Get feature columns (excluding QoE and timestamp)
    feature_cols = [col for col in train_df.columns if col not in ["QoE", "timestamp"]]
    print(f"Number of features: {len(feature_cols)}")
    
    # Normalize the data using only training data statistics
    norm_cols = feature_cols + ["QoE"]
    scaler = MinMaxScaler()
    # Fit scaler on combined training and validation data
    scaler.fit(combined_df[norm_cols])
    
    # Apply scaler to all datasets
    train_df[norm_cols] = scaler.transform(train_df[norm_cols])
    val_df[norm_cols] = scaler.transform(val_df[norm_cols])
    test_df[norm_cols] = scaler.transform(test_df[norm_cols])
    
    # Create sequences for each dataset
    X_train, y_train = create_sequences(train_df, seq_length=args.seq_length, 
                                         feature_cols=feature_cols, target_col='QoE')
    X_val, y_val = create_sequences(val_df, seq_length=args.seq_length, 
                                     feature_cols=feature_cols, target_col='QoE')
    X_test, y_test = create_sequences(test_df, seq_length=args.seq_length, 
                                      feature_cols=feature_cols, target_col='QoE')
    
    print("Training sequences:", X_train.shape[0])
    print("Validation sequences:", X_val.shape[0])
    print("Test sequences:", X_test.shape[0])
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
    callbacks_list = [early_stop, reduce_lr]

    feature_dim = X_train.shape[2]
    
    if args.tune:
        print("Starting automated hyperparameter tuning...")
        def hypermodel_builder(hp):
            model_type = args.model_type
            learning_rate = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4], default=args.learning_rate)
            
            if model_type == "linear":
                # Hyperparameters for linear model
                l1_reg = hp.Float("l1_reg", min_value=0.0, max_value=0.1, step=0.01, default=args.l1_reg)
                l2_reg = hp.Float("l2_reg", min_value=0.0, max_value=0.1, step=0.01, default=args.l2_reg)
                model = build_linear_model(seq_length=args.seq_length,
                                           feature_dim=feature_dim,
                                           learning_rate=learning_rate,
                                           l1_reg=l1_reg,
                                           l2_reg=l2_reg)
            elif model_type == "dnn":
                # Hyperparameters for DNN model
                units_1 = hp.Int("units_1", min_value=32, max_value=256, step=32, default=64)
                units_2 = hp.Int("units_2", min_value=16, max_value=128, step=16, default=32)
                dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1, default=args.dropout_rate)
                l2_reg = hp.Float("l2_reg", min_value=0.0, max_value=0.1, step=0.01, default=args.l2_reg)
                activation = hp.Choice("activation", values=["relu", "elu", "tanh"], default=args.activation)
                model = build_dnn_model(seq_length=args.seq_length,
                                       feature_dim=feature_dim,
                                       hidden_layers=[units_1, units_2],
                                       dropout_rate=dropout_rate,
                                       activation=activation,
                                       learning_rate=learning_rate,
                                       l2_reg=l2_reg)
            elif model_type == "lstm":
                # Hyperparameters for LSTM model
                hidden_units = hp.Int("hidden_units", min_value=32, max_value=128, step=32, default=args.hidden_units)
                num_layers = hp.Int("num_layers", min_value=1, max_value=4, step=1, default=args.num_layers)
                dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1, default=args.dropout_rate)
                attention_units = hp.Int("attention_units", min_value=64, max_value=256, step=64, default=args.attention_units)
                model = build_lstm_model(seq_length=args.seq_length,
                                         feature_dim=feature_dim,
                                         hidden_units=hidden_units,
                                         num_layers=num_layers,
                                         dropout_rate=dropout_rate,
                                         bidirectional=args.bidirectional,
                                         learning_rate=learning_rate,
                                         attention_units=attention_units)
            elif model_type == "gru":
                # Hyperparameters for GRU model
                hidden_units = hp.Int("hidden_units", min_value=32, max_value=128, step=32, default=args.hidden_units)
                num_layers = hp.Int("num_layers", min_value=1, max_value=4, step=1, default=args.num_layers)
                dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1, default=args.dropout_rate)
                attention_units = hp.Int("attention_units", min_value=64, max_value=256, step=64, default=args.attention_units)
                model = build_gru_model(seq_length=args.seq_length,
                                        feature_dim=feature_dim,
                                        hidden_units=hidden_units,
                                        num_layers=num_layers,
                                        dropout_rate=dropout_rate,
                                        bidirectional=args.bidirectional,
                                        learning_rate=learning_rate,
                                        attention_units=attention_units)
            elif model_type == "transformer":
                # Hyperparameters for Transformer model
                num_heads = hp.Int("num_heads", min_value=1, max_value=4, step=1, default=args.num_heads)
                ff_dim = hp.Int("ff_dim", min_value=32, max_value=128, step=32, default=args.ff_dim)
                dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1, default=args.dropout_rate)
                model = build_transformer_model(seq_length=args.seq_length,
                                                feature_dim=feature_dim,
                                                num_heads=num_heads,
                                                ff_dim=ff_dim,
                                                dropout_rate=dropout_rate,
                                                learning_rate=learning_rate)
            return model

        tuner = kt.Hyperband(
            hypermodel_builder,
            objective='val_loss',
            max_epochs=args.tune_epochs,
            factor=3,
            directory='tuner_dir',
            project_name='qoe_tuning'
        )
        
        # Use validation data for tuning
        tuner.search(X_train, y_train, validation_data=(X_val, y_val), 
                    epochs=args.tune_epochs, callbacks=callbacks_list)
        
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters found:")
        print(best_hp.values)
        model = hypermodel_builder(best_hp)
        model.summary()
        
        # Train the final model with the best hyperparameters
        history = model.fit(X_train, y_train, 
                           validation_data=(X_val, y_val),
                           epochs=args.epochs, 
                           batch_size=args.batch_size,
                           callbacks=callbacks_list)
    else:
        if args.model_type == "linear":
            print("Building Linear Regression model...")
            model = build_linear_model(seq_length=args.seq_length,
                                      feature_dim=feature_dim,
                                      learning_rate=args.learning_rate,
                                      l1_reg=args.l1_reg,
                                      l2_reg=args.l2_reg)
        elif args.model_type == "dnn":
            print("Building Simple DNN model...")
            model = build_dnn_model(seq_length=args.seq_length,
                                   feature_dim=feature_dim,
                                   hidden_layers=hidden_layers,
                                   dropout_rate=args.dropout_rate,
                                   activation=args.activation,
                                   learning_rate=args.learning_rate,
                                   l2_reg=args.l2_reg)
        elif args.model_type == "lstm":
            print("Building LSTM model with self-attention...")
            model = build_lstm_model(seq_length=args.seq_length,
                                     feature_dim=feature_dim,
                                     hidden_units=args.hidden_units,
                                     num_layers=args.num_layers,
                                     dropout_rate=args.dropout_rate,
                                     bidirectional=args.bidirectional,
                                     learning_rate=args.learning_rate,
                                     attention_units=args.attention_units)
        elif args.model_type == "gru":
            print("Building GRU model with self-attention...")
            model = build_gru_model(seq_length=args.seq_length,
                                    feature_dim=feature_dim,
                                    hidden_units=args.hidden_units,
                                    num_layers=args.num_layers,
                                    dropout_rate=args.dropout_rate,
                                    bidirectional=args.bidirectional,
                                    learning_rate=args.learning_rate,
                                    attention_units=args.attention_units)
        elif args.model_type == "transformer":
            print("Building Transformer model...")
            model = build_transformer_model(seq_length=args.seq_length,
                                            feature_dim=feature_dim,
                                            num_heads=args.num_heads,
                                            ff_dim=args.ff_dim,
                                            dropout_rate=args.dropout_rate,
                                            learning_rate=args.learning_rate)
        model.summary()
        
        # Train using validation data
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            callbacks=callbacks_list)
    
    # Evaluate on test data
    test_loss = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")
    
    # Save model and scaler
    model_filename = os.path.join(args.output_dir, f"model_{args.model_type}_with_attention.h5")
    model.save(model_filename)
    print("Saved model as", model_filename)
    
    scaler_filename = os.path.join(args.output_dir, "scaler.save")
    joblib.dump(scaler, scaler_filename)
    print("Saved scaler as", scaler_filename)
    
    # Inference Example:
    print(f"\nPreparing sample inference example for {args.model_type.upper()} model...")
    
    if args.augmented and not args.legacy_format:
        # New format (now the default for augmented mode)
        sample_data = {}
        base_timestamp = 20250204123000
        
        for i in range(5):
            timestamp = base_timestamp + i*2
            sample_data[str(timestamp)] = {
                "throughput": 1200.0,
                "packets_lost": 0.0,
                "packet_loss_rate": 0.0,
                "jitter": 15.0,
                "speed": 0.0
            }
        
        sample_data["timestamp"] = base_timestamp + 8
        sample_data["QoE"] = None
        
        # Flatten the sample data into feature vector
        sample_inference_record = {}
        flat_features = []
        
        for key in sorted([k for k in sample_data.keys() if k not in ["QoE", "timestamp"]]):
            entry = sample_data[key]
            flat_features.extend([
                entry["throughput"], entry["packets_lost"], entry["packet_loss_rate"], 
                entry["jitter"], entry["speed"]
            ])
        
        for i, val in enumerate(flat_features):
            sample_inference_record[f"f{i}"] = val
            
        if args.use_stats:
            # Add statistics
            sample_inference_record["throughput_mean"] = 1200.0
            sample_inference_record["throughput_std"] = 0.0
            sample_inference_record["throughput_min"] = 1200.0
            sample_inference_record["throughput_max"] = 1200.0
            
            sample_inference_record["packets_lost_mean"] = 0.0
            sample_inference_record["packets_lost_std"] = 0.0
            sample_inference_record["packets_lost_min"] = 0.0
            sample_inference_record["packets_lost_max"] = 0.0
            
            sample_inference_record["packet_loss_rate_mean"] = 0.0
            sample_inference_record["packet_loss_rate_std"] = 0.0
            sample_inference_record["packet_loss_rate_min"] = 0.0
            sample_inference_record["packet_loss_rate_max"] = 0.0
            
            sample_inference_record["jitter_mean"] = 15.0
            sample_inference_record["jitter_std"] = 0.0
            sample_inference_record["jitter_min"] = 15.0
            sample_inference_record["jitter_max"] = 15.0
            
            sample_inference_record["speed_mean"] = 0.0
            sample_inference_record["speed_std"] = 0.0
            sample_inference_record["speed_min"] = 0.0
            sample_inference_record["speed_max"] = 0.0
        
        sample_inference_record["QoE"] = None
        sample_inference_record["timestamp"] = base_timestamp + 8
    
    elif args.augmented:
        # Legacy augmented format
        f_values = []
        for i in range(5):
            f_values.extend([0.0, 15.0, 1200.0, 0.0])
        sample_inference_record = {}
        for i, val in enumerate(f_values):
            sample_inference_record[f"f{i}"] = val
        if args.use_stats:
            sample_inference_record["packet_loss_rate_mean"] = 0.0
            sample_inference_record["packet_loss_rate_std"] = 0.0
            sample_inference_record["packet_loss_rate_min"] = 0.0
            sample_inference_record["packet_loss_rate_max"] = 0.0
            sample_inference_record["jitter_mean"] = 15.0
            sample_inference_record["jitter_std"] = 0.0
            sample_inference_record["jitter_min"] = 15.0
            sample_inference_record["jitter_max"] = 15.0
            sample_inference_record["throughput_mean"] = 1200.0
            sample_inference_record["throughput_std"] = 0.0
            sample_inference_record["throughput_min"] = 1200.0
            sample_inference_record["throughput_max"] = 1200.0
            sample_inference_record["speed_mean"] = 0.0
            sample_inference_record["speed_std"] = 0.0
            sample_inference_record["speed_min"] = 0.0
            sample_inference_record["speed_max"] = 0.0
        sample_inference_record["QoE"] = None
        sample_inference_record["timestamp"] = "20250204123000"
    else:
        # Regular (legacy) format
        sample_inference_record = {
            "packet_loss_rate": 0.0,
            "jitter": 15.0,
            "throughput": 1200.0,
            "speed": 0.0,
            "QoE": None,
            "timestamp": "20250204123000"
        }
    
    inference_feature_cols = feature_cols
    
    # Get the last seq_length-1 records from the test dataset
    last_records = test_df.iloc[-(args.seq_length - 1):][inference_feature_cols].values
    
    # Create a DataFrame with the sample inference record
    sample_inference_df = pd.DataFrame([sample_inference_record])
    # Normalize using the same scaler
    inference_norm_cols = inference_feature_cols
    sample_inference_df[inference_norm_cols] = scaler.transform(sample_inference_df[inference_norm_cols])
    sample_inference_features = sample_inference_df[inference_feature_cols].values
    
    # Combine the last records with the sample inference record
    sequence_for_inference = np.vstack([last_records, sample_inference_features])
    sequence_for_inference = sequence_for_inference.reshape(1, args.seq_length, len(inference_feature_cols))
    
    # Predict QoE
    predicted_qoe_scaled = model.predict(sequence_for_inference)
    
    # Inverse transform to get the actual QoE value
    dummy_array = np.zeros((1, len(norm_cols)))
    dummy_array[0, -1] = predicted_qoe_scaled[0, 0]
    inverted = scaler.inverse_transform(dummy_array)
    predicted_qoe = inverted[0, -1]
    
    print("Predicted QoE for the next time step:", predicted_qoe)
    
    # If we're using one of the new models, print interpretable information
    if args.model_type == "linear" and not args.tune:
        # For linear model, extract weights to show feature importance
        weights = model.layers[-1].get_weights()[0]
        flattened_feature_names = []
        for t in range(args.seq_length):
            for feature in inference_feature_cols:
                flattened_feature_names.append(f"{feature}_t-{args.seq_length-t}")
        
        # Sort weights by absolute value to find most important features
        feature_importance = sorted(zip(flattened_feature_names, weights.flatten()), 
                                  key=lambda x: abs(x[1]), reverse=True)
        
        print("\nLinear Model Feature Importance:")
        for feature, weight in feature_importance[:10]:  # Show top 10 features
            print(f"{feature}: {weight[0]:.4f}")

if __name__ == "__main__":
    main()
