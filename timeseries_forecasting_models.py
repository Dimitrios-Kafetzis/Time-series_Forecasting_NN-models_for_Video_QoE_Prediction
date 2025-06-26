#!/usr/bin/env python3
"""
Filename: timeseries_forecasting_models.py
Author: Dimitrios Kafetzis
Creation Date: 2025-02-04
Description:
    This script implements deep learning models for time series forecasting using TensorFlow.
    It includes three model architectures:
        - LSTM
        - GRU
        - Transformer
    The purpose is to predict the QoE (Quality of Experience) value for network data.
    The input dataset is composed of JSON files (one per 10 seconds) with the following structure:
        {
            "packet_loss_rate": <float>,
            "jitter": <float>,
            "throughput": <float>,
            "speed": <float>,
            "QoE": <float>,
            "timestamp": "YYYYMMDDHHMMSS"
        }
    The script performs the following steps:
        - Loads and sorts the JSON files based on their timestamp.
        - Preprocesses the data (including normalization and sequence creation).
        - Splits the dataset into training and testing sets.
        - Builds, trains, and evaluates one of the selected models (LSTM, GRU, or Transformer).
        - Optionally performs automated hyperparameter tuning using Keras Tuner.
        - Monitors training using early stopping and learning rate reduction callbacks.
        - Provides an example of how to perform inference on new JSON input data.
Usage Examples:
    1. Train an LSTM model normally:
       $ python timeseries_forecasting_models.py --data_folder ./mock_dataset --model_type lstm --seq_length 5 --epochs 20 --batch_size 16
    2. Train a GRU model normally:
       $ python timeseries_forecasting_models.py --data_folder ./mock_dataset --model_type gru --seq_length 5 --epochs 20 --batch_size 16
    3. Train a Transformer model normally:
       $ python timeseries_forecasting_models.py --data_folder ./mock_dataset --model_type transformer --seq_length 5 --epochs 20 --batch_size 16
    4. Perform automated hyperparameter tuning:
       $ python timeseries_forecasting_models.py --data_folder ./mock_dataset --model_type lstm --seq_length 5 --epochs 20 --batch_size 16 --tune --max_trials 10 --tune_epochs 20
    Additionally, hyperparameters like learning rate, hidden units, number of layers, dropout rate,
    and bidirectionality (for RNNs) can be tuned via command-line arguments.
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
                                     LayerNormalization, MultiHeadAttention, Bidirectional)

# For automated tuning
import keras_tuner as kt

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================

def load_dataset_from_folder(folder_path):
    """
    Load all JSON files from the folder and return a DataFrame.
    Each JSON file is expected to have the keys:
      "packet_loss_rate", "jitter", "throughput", "speed", "QoE", "timestamp"
    """
    data = []
    # List all files and sort them by filename (which is the timestamp)
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                data.append(json_data)
    # Sort by the timestamp field to be safe (format: YYYYMMDDHHMMSS)
    data_sorted = sorted(data, key=lambda x: x['timestamp'])
    df = pd.DataFrame(data_sorted)
    return df

def preprocess_dataframe(df):
    """
    Convert columns to proper numeric types and convert timestamps.
    """
    # Convert the feature columns and target to numeric
    numeric_cols = ['packet_loss_rate', 'jitter', 'throughput', 'speed', 'QoE']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Convert timestamp string to datetime (assuming format YYYYMMDDHHMMSS)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def create_sequences(df, seq_length=5, feature_cols=['packet_loss_rate', 'jitter', 'throughput', 'speed'], target_col='QoE'):
    """
    Build sequences of shape (seq_length, number_of_features) and their corresponding target.
    The target is the QoE value at the time step immediately after the sequence.
    """
    X, y = [], []
    for i in range(len(df) - seq_length):
        seq_X = df.iloc[i:i+seq_length][feature_cols].values
        seq_y = df.iloc[i+seq_length][target_col]
        X.append(seq_X)
        y.append(seq_y)
    return np.array(X), np.array(y)

def train_test_split(X, y, train_ratio=0.8):
    """
    Split the data sequentially into training and testing sets.
    """
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

# =============================================================================
# 2. Model Definitions
# =============================================================================

def build_lstm_model(seq_length, feature_dim, hidden_units=50, num_layers=2, dropout_rate=0.2,
                     bidirectional=False, learning_rate=0.001):
    """Build and compile an LSTM model with optional bidirectionality and multiple layers."""
    model = Sequential()
    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        # For the first layer, we need to supply the input shape.
        if i == 0:
            if bidirectional:
                # Create the inner LSTM layer without the input shape then wrap it,
                # passing the input shape to the Bidirectional wrapper.
                lstm_layer = LSTM(hidden_units, return_sequences=return_seq,
                                  dropout=dropout_rate, recurrent_dropout=dropout_rate)
                lstm_layer = Bidirectional(lstm_layer, input_shape=(seq_length, feature_dim))
            else:
                lstm_layer = LSTM(hidden_units, return_sequences=return_seq,
                                  dropout=dropout_rate, recurrent_dropout=dropout_rate,
                                  input_shape=(seq_length, feature_dim))
        else:
            rnn = LSTM(hidden_units, return_sequences=return_seq,
                       dropout=dropout_rate, recurrent_dropout=dropout_rate)
            lstm_layer = Bidirectional(rnn) if bidirectional else rnn
        model.add(lstm_layer)
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


def build_gru_model(seq_length, feature_dim, hidden_units=50, num_layers=2, dropout_rate=0.2,
                    bidirectional=False, learning_rate=0.001):
    """Build and compile a GRU model with optional bidirectionality and multiple layers."""
    model = Sequential()
    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        # For the first layer, supply the input shape appropriately.
        if i == 0:
            if bidirectional:
                gru_layer = GRU(hidden_units, return_sequences=return_seq,
                                dropout=dropout_rate, recurrent_dropout=dropout_rate)
                gru_layer = Bidirectional(gru_layer, input_shape=(seq_length, feature_dim))
            else:
                gru_layer = GRU(hidden_units, return_sequences=return_seq,
                                dropout=dropout_rate, recurrent_dropout=dropout_rate,
                                input_shape=(seq_length, feature_dim))
        else:
            rnn = GRU(hidden_units, return_sequences=return_seq,
                      dropout=dropout_rate, recurrent_dropout=dropout_rate)
            gru_layer = Bidirectional(rnn) if bidirectional else rnn
        model.add(gru_layer)
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Transformer block as a custom layer
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
    """Build and compile a Transformer model for time series forecasting."""
    inputs = Input(shape=(seq_length, feature_dim))
    transformer_block = TransformerBlock(embed_dim=feature_dim, num_heads=num_heads, ff_dim=ff_dim, rate=dropout_rate)
    x = transformer_block(inputs)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# =============================================================================
# 3. Main Routine: Training, Evaluation, Automated Tuning, and Inference
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to folder containing JSON files for training/testing.")
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["lstm", "gru", "transformer"],
                        help="Type of model to train: lstm, gru, or transformer.")
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Sequence length (number of time steps used as input).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    # Hyperparameter arguments for RNN-based models
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--hidden_units", type=int, default=50, help="Number of hidden units in RNN layers.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of stacked recurrent layers.")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for recurrent layers.")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional RNN layers.")
    # Transformer-specific arguments
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads for Transformer.")
    parser.add_argument("--ff_dim", type=int, default=64, help="Feed-forward dimension for Transformer.")
    # Automated tuning parameters
    parser.add_argument("--tune", action="store_true", help="Enable automated hyperparameter tuning using Keras Tuner.")
    parser.add_argument("--max_trials", type=int, default=10, help="Maximum number of trials for tuning.")
    parser.add_argument("--tune_epochs", type=int, default=20, help="Number of epochs to train each trial during tuning.")
    args = parser.parse_args()

    # Load and preprocess data
    print("Loading data from:", args.data_folder)
    df = load_dataset_from_folder(args.data_folder)
    df = preprocess_dataframe(df)

    # For normalization we use all 5 columns: 4 features and QoE
    feature_and_target_cols = ['packet_loss_rate', 'jitter', 'throughput', 'speed', 'QoE']
    scaler = MinMaxScaler()
    df[feature_and_target_cols] = scaler.fit_transform(df[feature_and_target_cols])
    
    # Create sequences: each sequence consists of `seq_length` time steps and the target is QoE at the next step.
    X, y = create_sequences(df, seq_length=args.seq_length)
    print("Total sequences:", X.shape[0])
    
    # Split the dataset into training and test sets (maintaining sequential order)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("Training samples:", X_train.shape[0], "Test samples:", X_test.shape[0])
    
    # Define callbacks for monitoring and regularization
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
    callbacks_list = [early_stop, reduce_lr]

    feature_dim = X.shape[2]  # should be 4: packet_loss_rate, jitter, throughput, speed

    # If automated tuning is enabled, use Keras Tuner to search for optimal hyperparameters.
    if args.tune:
        print("Starting automated hyperparameter tuning...")
        def hypermodel_builder(hp):
            model_type = args.model_type
            # Set tunable hyperparameters; use defaults from args as starting point.
            hidden_units = hp.Int("hidden_units", min_value=32, max_value=128, step=32, default=args.hidden_units)
            num_layers = hp.Int("num_layers", min_value=1, max_value=4, step=1, default=args.num_layers)
            dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1, default=args.dropout_rate)
            learning_rate = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4], default=args.learning_rate)
            if model_type == "lstm":
                model = build_lstm_model(seq_length=args.seq_length,
                                         feature_dim=feature_dim,
                                         hidden_units=hidden_units,
                                         num_layers=num_layers,
                                         dropout_rate=dropout_rate,
                                         bidirectional=args.bidirectional,
                                         learning_rate=learning_rate)
            elif model_type == "gru":
                model = build_gru_model(seq_length=args.seq_length,
                                        feature_dim=feature_dim,
                                        hidden_units=hidden_units,
                                        num_layers=num_layers,
                                        dropout_rate=dropout_rate,
                                        bidirectional=args.bidirectional,
                                        learning_rate=learning_rate)
            elif model_type == "transformer":
                num_heads = hp.Int("num_heads", min_value=1, max_value=4, step=1, default=args.num_heads)
                ff_dim = hp.Int("ff_dim", min_value=32, max_value=128, step=32, default=args.ff_dim)
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
        tuner.search(X_train, y_train, validation_split=0.2, epochs=args.tune_epochs, callbacks=callbacks_list)
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters found:")
        print(best_hp.values)
        # Build the model with the best hyperparameters and train again.
        model = hypermodel_builder(best_hp)
        model.summary()
        history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                            validation_split=0.2, callbacks=callbacks_list)
    else:
        # No tuning: build the model using command-line arguments
        if args.model_type == "lstm":
            print("Building LSTM model...")
            model = build_lstm_model(seq_length=args.seq_length, 
                                     feature_dim=feature_dim, 
                                     hidden_units=args.hidden_units, 
                                     num_layers=args.num_layers, 
                                     dropout_rate=args.dropout_rate, 
                                     bidirectional=args.bidirectional,
                                     learning_rate=args.learning_rate)
        elif args.model_type == "gru":
            print("Building GRU model...")
            model = build_gru_model(seq_length=args.seq_length, 
                                    feature_dim=feature_dim, 
                                    hidden_units=args.hidden_units, 
                                    num_layers=args.num_layers, 
                                    dropout_rate=args.dropout_rate, 
                                    bidirectional=args.bidirectional,
                                    learning_rate=args.learning_rate)
        elif args.model_type == "transformer":
            print("Building Transformer model...")
            model = build_transformer_model(seq_length=args.seq_length, 
                                            feature_dim=feature_dim, 
                                            num_heads=args.num_heads, 
                                            ff_dim=args.ff_dim, 
                                            dropout_rate=args.dropout_rate,
                                            learning_rate=args.learning_rate)
        model.summary()
        history = model.fit(X_train, y_train,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            validation_split=0.2,
                            callbacks=callbacks_list)
    
    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    
    # Save the model and scaler for later inference
    model_filename = f"model_{args.model_type}.h5"
    model.save(model_filename)
    print("Saved model as", model_filename)
    joblib.dump(scaler, "scaler.save")
    print("Saved scaler as scaler.save")
    
    # =============================================================================
    # 4. Inference Example
    # =============================================================================
    #
    # For inference, we assume that a new JSON file arrives every 10 seconds.
    # To predict QoE for the next time step, we need to form an input sequence.
    # Here, we take the last (seq_length - 1) records from our dataset and add the new record.
    
    sample_inference_json = {
        "packet_loss_rate": 2.50,
        "jitter": 0.190,
        "throughput": 105.00,
        "speed": 45.20,
        "QoE": None,
        "timestamp": "20250204123000"
    }
    
    # Extract the last (seq_length - 1) rows (only the 4 features) from the current (normalized) dataframe.
    last_records = df.iloc[-(args.seq_length - 1):][['packet_loss_rate', 'jitter', 'throughput', 'speed']].values
    
    # Prepare the new record.
    # Our scaler was fitted on 5 columns, so to transform the 4 features we add a dummy value for QoE.
    new_record = np.array([[sample_inference_json["packet_loss_rate"],
                             sample_inference_json["jitter"],
                             sample_inference_json["throughput"],
                             sample_inference_json["speed"]]])
    dummy = np.zeros((new_record.shape[0], 1))
    new_record_full = np.hstack([new_record, dummy])
    new_record_scaled = scaler.transform(new_record_full)[:, :4]  # keep only the 4 features
    
    # Form the full input sequence by stacking the last records with the new record.
    sequence_for_inference = np.vstack([last_records, new_record_scaled])
    # Reshape to (1, seq_length, 4) for prediction.
    sequence_for_inference = sequence_for_inference.reshape(1, args.seq_length, feature_dim)
    
    # Predict (the output is in normalized scale)
    predicted_qoe_scaled = model.predict(sequence_for_inference)
    
    # To convert the predicted QoE back to the original scale, we need to invert the normalization.
    # We create a dummy array with the same number of columns as the scaler was fitted on.
    dummy_array = np.zeros((1, 5))
    dummy_array[0, -1] = predicted_qoe_scaled[0, 0]
    inverted = scaler.inverse_transform(dummy_array)
    predicted_qoe = inverted[0, -1]
    
    print("Predicted QoE for the next time step:", predicted_qoe)

if __name__ == "__main__":
    main()
