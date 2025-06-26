# debug_test.py
import sys
import os
import traceback

# Set debugging
debug = True

try:
    print("Starting test with debugging...")
    # Get the data folder path
    data_folder = os.path.expanduser('~/Impact-xG_prediction_model/real_dataset')
    model_file = os.path.expanduser('~/Impact-xG_prediction_model/forecasting_models_v5/linear_basic.h5')
    scaler_file = os.path.expanduser('~/Impact-xG_prediction_model/forecasting_models_v5/scaler.save')
    output_dir = os.path.expanduser('~/Impact-xG_prediction_model/forecasting_models_v5')
    
    print(f"Data folder: {data_folder}")
    print(f"Model file: {model_file}")
    print(f"Scaler file: {scaler_file}")
    print(f"Output directory: {output_dir}")
    
    # Import necessary libraries
    print("Importing libraries...")
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from timeseries_forecasting_models_v5 import load_augmented_dataset_from_folder, preprocess_dataframe, create_sequences

    # Load the dataset
    print("Loading dataset...")
    df = load_augmented_dataset_from_folder(data_folder, use_stats=True, new_format=True)
    print(f"Dataset loaded, shape: {df.shape}")
    
    # Get features
    feature_cols = [col for col in df.columns if col not in ["QoE", "timestamp"]]
    print(f"Number of features: {len(feature_cols)}")
    
    # Preprocess
    print("Preprocessing dataframe...")
    df = preprocess_dataframe(df)
    
    # Load scaler
    print("Loading scaler...")
    scaler = joblib.load(scaler_file)
    
    # Apply scaling
    norm_cols = feature_cols + ["QoE"]
    print(f"Applying scaling to {len(norm_cols)} columns...")
    df[norm_cols] = scaler.transform(df[norm_cols])
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(df, seq_length=5, feature_cols=feature_cols, target_col='QoE')
    print(f"Created sequences, X shape: {X.shape}, y shape: {y.shape}")
    
    # Split for testing
    test_size = int(len(X) * 0.2)
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    print(f"Test set size: {len(X_test)}")
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(model_file)
    print("Model loaded successfully")
    
    # Evaluate model
    print("Evaluating model...")
    test_loss = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {test_loss}")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test, verbose=1)
    
    # Calculate metrics
    print("Calculating metrics...")
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\nEvaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    
    # Extract weights (feature importance) for linear model
    print("Extracting feature importance...")
    weights = model.layers[-1].get_weights()[0]
    bias = model.layers[-1].get_weights()[1]
    
    # Create feature names based on sequence structure
    flattened_feature_names = []
    for t in range(5):  # seq_length
        for feature in feature_cols:
            flattened_feature_names.append(f"{feature}_t-{5-t}")
    
    # Sort weights by absolute magnitude
    feature_importance = sorted(zip(flattened_feature_names, weights.flatten()), 
                              key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop 10 Features by Importance:")
    for feature, weight in feature_importance[:10]:
        print(f"{feature}: {weight:.4f}")
    
    print(f"Bias term: {bias[0]:.4f}")
    
    # Save feature importance to file
    print("Saving feature importance to file...")
    import json
    feature_importance_dir = os.path.join(output_dir, "feature_importance")
    os.makedirs(feature_importance_dir, exist_ok=True)
    
    feature_importance_data = {
        "model_name": os.path.basename(model_file),
        "model_type": "Linear Regressor",
        "feature_importance": [{"feature": f, "weight": float(w)} for f, w in feature_importance],
        "bias": float(bias[0]),
        "feature_count": len(feature_cols),
        "sequence_length": 5
    }
    
    output_file = os.path.join(feature_importance_dir, f"{os.path.basename(model_file)}.importance.json")
    with open(output_file, 'w') as f:
        json.dump(feature_importance_data, f, indent=2)
    
    print(f"Feature importance saved to: {output_file}")
    print("Test completed successfully!")
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    traceback.print_exc()
    sys.exit(1)