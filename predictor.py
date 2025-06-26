import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
from datetime import datetime

# Import the custom attention layer from the training script
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


class Predictor:
    def __init__(self, model_path="model_gru_with_attention.h5", scaler_path="scaler.save", seq_length=5, use_stats=False):
        """
        Initialize the Predictor with the trained model and scaler.
        
        Args:
            model_path: Path to the saved model (.h5 file)
            scaler_path: Path to the saved scaler (.save file)
            seq_length: Sequence length used during training (default: 5)
            use_stats: Whether statistical features were used during training (default: False)
        """
        print("Predictor initializing...")
        
        self.seq_length = seq_length
        self.use_stats = use_stats
        self.model = None
        self.scaler = None
        self.feature_cols = None
        
        # Load the model with custom objects
        try:
            # Load the model with the custom SelfAttention layer
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'SelfAttention': SelfAttention}
            )
            print(f"Model loaded successfully from {model_path}")
            
            # Load the scaler
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded successfully from {scaler_path}")
            
            # Determine feature columns based on the new format (5 timestamps × 5 features = 25 features)
            self.feature_cols = [f"f{i}" for i in range(25)]
            
            # Get the expected feature names from the scaler
            if hasattr(self.scaler, 'feature_names_in_'):
                self.scaler_features = list(self.scaler.feature_names_in_)
                print(f"Scaler expects {len(self.scaler_features)} features: {self.scaler_features[:5]}...")
            else:
                # If scaler doesn't have feature names, we'll need to figure it out
                self.scaler_features = None
                print("Warning: Scaler doesn't have feature_names_in_ attribute")
            
            print("Predictor initialized successfully.")
            
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Looking for model at: {model_path}")
            print(f"Looking for scaler at: {scaler_path}")
            raise

    def _extract_features_from_trace(self, trace_dict):
        """
        Extract features from a single trace dictionary.
        
        Args:
            trace_dict: Dictionary containing timestamp data
            
        Returns:
            Dictionary with flattened features
        """
        # Get all timestamp keys (excluding the main "timestamp" key)
        inner_keys = [k for k in trace_dict.keys() if k not in ["timestamp"]]
        inner_keys = sorted(inner_keys)
        
        # We expect 5 timestamps for a sequence length of 5
        if len(inner_keys) != self.seq_length:
            raise ValueError(f"Expected {self.seq_length} timestamps, but got {len(inner_keys)}")
        
        # Flatten the features in the same order as during training
        flat_features = []
        
        # For statistics calculation if needed
        if self.use_stats:
            stats_features = {
                "packet_loss_rate": [], "jitter": [], "throughput": [], 
                "speed": [], "packets_lost": []
            }
        
        for key in inner_keys:
            entry = trace_dict[key]
            # Order: throughput, packets_lost, packet_loss_rate, jitter, speed
            flat_features.extend([
                entry["throughput"],
                entry["packets_lost"],
                entry["packet_loss_rate"],
                entry["jitter"],
                float(entry["speed"])  # Ensure speed is float
            ])
            
            if self.use_stats:
                stats_features["throughput"].append(entry["throughput"])
                stats_features["packets_lost"].append(entry["packets_lost"])
                stats_features["packet_loss_rate"].append(entry["packet_loss_rate"])
                stats_features["jitter"].append(entry["jitter"])
                stats_features["speed"].append(float(entry["speed"]))
        
        # Create feature dictionary
        feature_dict = {}
        for i, val in enumerate(flat_features):
            feature_dict[f"f{i}"] = val
        
        # Add statistical features if needed
        if self.use_stats:
            for feature in stats_features.keys():
                arr = np.array(stats_features[feature])
                feature_dict[f"{feature}_mean"] = float(np.mean(arr))
                feature_dict[f"{feature}_std"] = float(np.std(arr))
                feature_dict[f"{feature}_min"] = float(np.min(arr))
                feature_dict[f"{feature}_max"] = float(np.max(arr))
        
        # Add QoE as None (will be filled with 0 for scaling)
        feature_dict["QoE"] = 0.0
        
        # Add timestamp if it's in the scaler features
        if self.scaler_features and "timestamp" in self.scaler_features:
            # Use the timestamp from the trace_dict
            feature_dict["timestamp"] = trace_dict.get("timestamp", 0)
        
        return feature_dict

    def infer(self, trace_data):
        """
        Perform inference on the provided trace data.
        
        Args:
            trace_data: List of dictionaries containing network trace data
            
        Returns:
            List of predicted QoE values
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not loaded properly")
        
        print(f"Processing {len(trace_data)} trace records...")
        
        predictions = []
        
        # Process each trace independently
        for i, trace in enumerate(trace_data):
            try:
                # Extract features from the trace
                features = self._extract_features_from_trace(trace)
                
                # Create a DataFrame with the features
                df = pd.DataFrame([features])
                
                # Get the columns that the scaler expects
                if self.scaler_features:
                    # Ensure we have all required columns
                    for col in self.scaler_features:
                        if col not in df.columns:
                            # Add missing columns with default values
                            if col == "QoE":
                                df[col] = 0.0
                            elif col == "timestamp":
                                df[col] = trace.get("timestamp", 0)
                            else:
                                # For any other missing columns, add as 0
                                df[col] = 0.0
                    
                    # Reorder columns to match scaler's expected order
                    df = df[self.scaler_features]
                
                # Normalize all features using the scaler
                df_normalized = self.scaler.transform(df)
                df_normalized = pd.DataFrame(df_normalized, columns=df.columns)
                
                # For this model, we need ALL features (not just the base features) at each timestep
                # The model expects (batch_size, seq_length, 45) where 45 = all features except QoE
                
                # Get all feature columns except QoE
                all_feature_cols = [col for col in df_normalized.columns if col != 'QoE']
                
                # Extract all features for the model
                model_features = df_normalized[all_feature_cols].values
                
                # The model expects the same features repeated for each timestep in the sequence
                # Since we only have one sample, we'll repeat it for each timestep
                sequence = np.tile(model_features, (1, self.seq_length, 1))
                
                # Make prediction
                predicted_qoe_scaled = self.model.predict(sequence, verbose=0)
                
                # Inverse transform to get the actual QoE value
                # Create a dummy array with the same shape as what the scaler expects
                dummy_array = np.zeros((1, len(df.columns)))
                
                # Find the index of QoE column
                qoe_index = list(df.columns).index("QoE")
                dummy_array[0, qoe_index] = predicted_qoe_scaled[0, 0]
                
                # Inverse transform
                inverted = self.scaler.inverse_transform(dummy_array)
                predicted_qoe = inverted[0, qoe_index]
                
                predictions.append(float(predicted_qoe))
                
                print(f"Trace {i}: Predicted QoE = {predicted_qoe:.4f}")
                
            except Exception as e:
                print(f"Error processing trace {i}: {e}")
                import traceback
                traceback.print_exc()
                predictions.append(-1)  # Return -1 for errors as in the original
        
        # Return single value if only one trace, otherwise return list
        if len(predictions) == 1:
            return predictions[0]
        else:
            return predictions