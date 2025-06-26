# Time Series Forecasting Models for Video QoE Prediction

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating Mock Datasets](#generating-mock-datasets)
  - [Generating Network Datasets](#generating-network-datasets)
  - [Running Experiments](#running-experiments)
  - [Training All Models](#training-all-models)
  - [Testing All Models](#testing-all-models)
  - [Inference](#inference)
  - [Testing and Evaluation](#testing-and-evaluation)
  - [Automated Hyperparameter Tuning](#automated-hyperparameter-tuning)
- [Models](#models)
- [Distinction Between Timeseries Forecasting Model Versions](#distinction-between-timeseries-forecasting-model-versions)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Video streaming applications require constant monitoring of network conditions to maintain high user Quality of Experience (QoE). This project implements time series forecasting models using TensorFlow and Keras to predict QoE based on network metrics like packet loss, jitter, throughput, and speed. The project supports both standard and augmented dataset formats, enabling robust experimentation and real-world inference scenarios.

## Features

- **Multiple Model Architectures:**
  - Linear Regressor (Baseline)
  - Simple DNN (Deep Neural Network)
  - LSTM (with and without self-attention)
  - GRU (with and without self-attention)
  - Transformer
 
- **Experimentation Pipelines:**
  Two experiment runners (experiment_runner.py and experiment_runner_v2.py) automate hyperparameter grid search and log results into CSV files for analysis.

- **Automated Model Training:**
  The train_all_models_v5.sh script automates training of multiple model variants with different configurations.

- **Comprehensive Model Evaluation:**
  The test_all_models.sh script evaluates all trained models and generates comparative performance reports.

- **Synthetic Dataset Generation:**
  The generate_mock_dataset.py script creates JSON datasets in three modes:
  - **Dataset Mode:** Complete training/testing data with QoE values.
  - **Inference Mode:** JSON files with QoE set to null for inference testing.
  - **Augmented Mode:** 5‑second windows with per‑second measurements and aggregated QoE, with optional statistical features.

- **Network Dataset Generation:**
  The network_dataset_generator.py script creates realistic network datasets by reusing existing JSON files and organizing them into predictable urban network patterns.

- **Inference Support:**
  The run_inference.py script loads a saved model and scaler, processes new JSON input (supporting both standard and augmented formats), and outputs a predicted QoE value.

- **Model Evaluation and Testing:**
  The test_models.py script evaluates models using regression metrics (MSE, MAE, R²) and measures inference latency, with options to simulate performance on different hardware (e.g., Xeon, Jetson).

- **Automated Hyperparameter Tuning:**
  The project supports automated tuning for all model types via Keras Tuner.

- **Version 6 Enhancements:**
  Support for separate train/validation/test directories, enhanced validation capabilities, and improved model evaluation pipelines.

## Installation

1. **System Requirements:**
   - Ubuntu 20.04+ (or compatible)
   - Python 3.8+
   - TensorFlow 2.x
   - CUDA-compatible GPU recommended for faster training

2. **Set Up a Virtual Environment (Recommended):**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies:**
```bash
pip install --upgrade pip
pip install numpy pandas tensorflow joblib matplotlib scikit-learn keras_tuner tqdm seaborn
```

## Usage

### Generating Mock Datasets

Generate synthetic JSON datasets using:
```bash
python3 generate_mock_dataset.py --output_folder <folder_path> --mode <mode> --num_points <N> --start_timestamp <YYYYMMDDHHMMSS>
```

**Examples:**

- **Dataset Mode (Training/Testing):**
```bash
python3 generate_mock_dataset.py --output_folder ./mock_dataset --mode dataset --num_points 100 --start_timestamp 20250130114158
```

- **Inference Mode:**
```bash
python3 generate_mock_dataset.py --output_folder ./inference_inputs --mode inference --num_points 1 --start_timestamp 20250204123000
```

- **Augmented Mode (with per‑second granularity):**
```bash
python3 generate_mock_dataset.py --output_folder ./augmented_dataset --mode augmented --num_points 100 --start_timestamp 20250130114158
```

### Generating Network Datasets

The `network_dataset_generator.py` script creates realistic network datasets by reusing existing JSON files and organizing them into predictable patterns that simulate urban network behavior throughout the week.

**Features:**
- Categorizes existing files by QoE levels (A: Excellent, B: Good, C: Average, D: Fair, E: Poor)
- Creates weekly patterns simulating network quality variations throughout different times of day
- Generates datasets for any specified time period by intelligently cycling through available files
- Maintains realistic transitions between network quality states

**Usage:**
```bash
python3 network_dataset_generator.py --input-dir <source_folder> --output-dir <output_folder> --weeks <num_weeks> [--start-date YYYY-MM-DD]
```

**Example:**
```bash
# Generate 2 weeks of network data starting from today
python3 network_dataset_generator.py --input-dir ./original_dataset --output-dir ./generated_dataset --weeks 2

# Generate 4 weeks starting from a specific date
python3 network_dataset_generator.py --input-dir ./original_dataset --output-dir ./generated_dataset --weeks 4 --start-date 2025-01-01
```

The generator creates patterns such as:
- Excellent quality during night hours (low network usage)
- Degraded quality during peak hours (5-9 PM)
- Different patterns for weekdays vs weekends
- Smooth transitions between quality categories

### Running Experiments

Two experiment runners are provided:

- **Standard Experiment Runner:**
```bash
python3 experiment_runner.py --data_folder ./mock_dataset --epochs 20 --batch_size 16
```

- **Enhanced Experiment Runner (Augmented Dataset Support):**
```bash
python3 experiment_runner_v2.py --data_folder ./augmented_dataset --epochs 20 --batch_size 16 --augmented
```

Both scripts iterate over a grid of hyperparameter configurations, log the results to CSV files, and analyze them to determine the best configuration based on test loss.

### Training All Models

To train a comprehensive set of model variants with different architectures and configurations, use the train_all_models_v5.sh script:

```bash
./train_all_models_v5.sh
```

For version 6 models with separate train/validation/test directories:

```bash
python3 timeseries_forecasting_models_v6.py --train_dir ./final_complete_dataset/train_set --val_dir ./final_complete_dataset/validation_set --test_dir ./final_complete_dataset/test_set --model_type lstm --seq_length 5 --epochs 20 --batch_size 16 --output_dir ./forecasting_models_v6 --augmented --use_stats
```

This script trains 21 different model configurations:
- 4 Linear Regressor variants (basic, L1 regularization, L2 regularization, ElasticNet)
- 4 Simple DNN variants (basic, deep, with ELU activation, with high dropout)
- 5 LSTM variants (basic, deep, wide, bidirectional, with stats)
- 5 GRU variants (basic, deep, wide, bidirectional, with stats)
- 5 Transformer variants (basic, more heads, large feed-forward, low dropout, with stats)

All models are saved to the specified output directory with appropriate naming conventions.

### Testing All Models

To evaluate and compare all trained models, use the test_all_models.sh script:

```bash
./test_all_models.sh
```

For version 6 models with enhanced capabilities:

```bash
./test_all_models_v6.sh --models-dir ./forecasting_models_v6 --test-dir ./final_complete_dataset/test_set --scaler-file ./forecasting_models_v6/scaler.save --seq-length 5 --use-stats --validate --validation-folder ./validation_dataset
```

Options for v6 testing:
- `--models-dir`: Directory containing model files
- `--test-dir`: Test dataset path
- `--scaler-file`: Path to scaler file
- `--seq-length`: Sequence length (default: 5)
- `--use-stats`: Enable statistical features
- `--validate`: Enable validation testing
- `--validation-folder`: Validation dataset folder

This script:
1. Tests each model in the specified directory
2. Computes evaluation metrics (MSE, MAE, R² Score)
3. Measures inference latency for each model
4. Generates a comprehensive evaluation report
5. Creates a CSV file with all results
6. Optionally runs validation experiments

The report includes model rankings by each metric and identifies the best overall model based on a weighted combination of all metrics.

### Inference

Run the inference script to predict QoE for new input data:

- **Standard Mode:**
```bash
python3 run_inference.py --inference_file ./inference_inputs/20250204123000.json --data_folder ./mock_dataset --seq_length 5 --model_file forecasting_models_v5/lstm_with_stats.h5 --scaler_file forecasting_models_v5/scaler.save
```

- **Augmented Mode with Statistical Features:**
```bash
python3 run_inference.py --inference_file ./inference_inputs/20250204123000.json --data_folder ./augmented_dataset --seq_length 5 --model_file forecasting_models_v5/lstm_with_stats.h5 --scaler_file forecasting_models_v5/scaler.save --augmented --use_stats
```

### Testing and Evaluation

Evaluate your trained models using:
```bash
python3 test_models.py --data_folder <folder_path> --model_file <model_file> --seq_length 5 --scaler_file forecasting_models_v5/scaler.save [--augmented] [--use_stats] [--simulate_device <device>]
```

For version 6 models with enhanced testing:
```bash
python3 test_models_v6.py --data_folder ./final_complete_dataset/test_set --model_file ./forecasting_models_v6/gru_basic.h5 --seq_length 5 --scaler_file ./forecasting_models_v6/scaler.save --augmented --use_stats --verbose
```

**Example:**
```bash
python3 test_models.py --data_folder ./augmented_dataset --model_file forecasting_models_v5/gru_with_stats.h5 --seq_length 5 --scaler_file forecasting_models_v5/scaler.save --augmented --use_stats --simulate_device jetson
```

This script reports evaluation metrics such as MSE, MAE, R², and measures inference latency.

### Automated Hyperparameter Tuning

The timeseries_forecasting_models_v5.py supports automated hyperparameter tuning for all model types via Keras Tuner:

```bash
python3 timeseries_forecasting_models_v5.py --data_folder ./augmented_dataset --model_type lstm --seq_length 5 --epochs 20 --batch_size 16 --augmented --use_stats --tune --max_trials 10 --tune_epochs 20
```

For version 6 with separate directories:
```bash
python3 timeseries_forecasting_models_v6.py --train_dir ./final_complete_dataset/train_set --val_dir ./final_complete_dataset/validation_set --test_dir ./final_complete_dataset/test_set --model_type lstm --seq_length 5 --epochs 20 --batch_size 16 --augmented --use_stats --tune --max_trials 10 --tune_epochs 20
```

This allows you to find optimal hyperparameters for any model type (linear, dnn, lstm, gru, transformer).

## Models

The latest version (v5/v6) includes the following model types:

1. **Linear Regressor Models**: 
   - Basic implementation flattens input sequences and applies a single Dense layer
   - Variants with L1, L2, and ElasticNet regularization

2. **Simple DNN Models**:
   - Flatten input sequence and apply multiple Dense layers
   - Configurable activation functions, depth, and dropout rates

3. **LSTM Models with Self-Attention**:
   - Replace traditional pooling with self-attention mechanism
   - Variants with different depths, widths, and bidirectional configurations

4. **GRU Models with Self-Attention**:
   - Similar to LSTM but using GRU cells
   - Also implements self-attention for temporal feature aggregation

5. **Transformer Models**:
   - Implements transformer architecture for sequence modeling
   - Configurable attention heads, feed-forward dimensions, and dropout

## Distinction Between Timeseries Forecasting Model Versions

There are six versions of the model definition scripts:

- **timeseries_forecasting_models_v2.py:**
  - Implements LSTM, GRU, Transformer, and a Linear Regressor baseline.
  - Uses Mean Squared Error (MSE) as the loss function.
  - Contains a simpler architecture without explicit temporal pooling.
  - Provides support for automated hyperparameter tuning via Keras Tuner.

- **timeseries_forecasting_models_v3.py:**
  - Focuses on LSTM, GRU, and Transformer models (linear regressor is not included).
  - Implements architectural enhancements such as forcing all recurrent layers to output sequences and applying GlobalAveragePooling1D to aggregate temporal information.
  - Uses the log-cosh loss function for potentially smoother convergence.
  - Also supports hyperparameter tuning via Keras Tuner.

- **timeseries_forecasting_models_v4.py:**
  - Builds upon v3 by replacing GlobalAveragePooling1D with a custom SelfAttention mechanism in the LSTM and GRU models.
  - Implements the SelfAttention layer class for enhanced temporal feature aggregation.
  - Maintains the three core model architectures: LSTM, GRU, and Transformer.
  - Uses log-cosh loss function and supports automated hyperparameter tuning.
  - Allows for bidirectional RNNs combined with the self-attention mechanism.
  
- **timeseries_forecasting_models_v5.py:**
  - Most comprehensive implementation with five model types: Linear Regressor, Simple DNN, LSTM, GRU, and Transformer.
  - Extends the SelfAttention approach from v4 to all recurrent models.
  - Adds multiple linear regressor variants with different regularization strategies (L1, L2, ElasticNet).
  - Implements configurable DNN models with various architectures, activations, and dropout rates.
  - Uses log-cosh loss for most models (except linear regressors which use MSE).
  - Enhanced hyperparameter tuning with model-specific parameter spaces.
  - Full support for statistical features with the `--use_stats` flag.

- **timeseries_forecasting_models_v6.py:**
  - Enhanced version of v5 with support for separate train/validation/test directories.
  - Designed for more rigorous model evaluation with dedicated validation sets.
  - Maintains all model architectures from v5 (Linear, DNN, LSTM, GRU, Transformer).
  - Configurable output directory for saving trained models and scalers.
  - Uses validation data during training for better generalization.
  - Key enhancements:
    - `--train_dir`, `--val_dir`, `--test_dir` arguments for data separation
    - `--output_dir` for customizable model save location
    - Consistent scaler fitting across train/validation data
    - Improved data loading for multi-directory workflows
  
Choose the version that best suits your experimental needs or to compare performance differences.

### Model Validation

The repository includes a validation system to perform controlled experiments on trained models using known ground truth data. This allows for more realistic assessment of model performance beyond traditional test set evaluation.

#### Preparing a Validation Dataset

First, create a validation dataset with pairs of files (with/without QoE values) using:

```bash
python3 prepare_validation_data.py --input_folder ./real_dataset --output_folder ./validation_dataset --sample_ratio 0.2 --random_seed 42
```

Options:
- `--input_folder`: Path to original dataset with ground truth QoE values
- `--output_folder`: Where to save the validation dataset
- `--sample_ratio`: Fraction of files to include in validation set (default: 1.0)
- `--random_seed`: Random seed for reproducible sampling
- `--legacy_format`: Use if your dataset is in legacy format

#### Validating Models

Run controlled validation experiments on one or more models:

```bash
python3 validate_models.py --validation_folder ./validation_dataset --model_dir ./forecasting_models_v5 --scaler_file ./forecasting_models_v5/scaler.save --output_dir ./validation_results --seq_length 5 --use_stats
```

For version 6 models with enhanced validation:
```bash
python3 validate_models_v6.py --validation_folder ./validation_dataset --model_dir ./forecasting_models_v6 --scaler_file ./forecasting_models_v6/scaler.save --output_dir ./validation_results --seq_length 5 --use_stats --plots
```

For a single model:
```bash
python3 validate_models.py --validation_folder ./validation_dataset --model_file ./forecasting_models_v5/model_lstm.h5 --scaler_file ./forecasting_models_v5/scaler.save --output_dir ./validation_results_lstm
```

Options:
- `--validation_folder`: Path to validation dataset created by prepare_validation_data.py
- `--model_dir`: Directory containing all models to validate
- `--model_file`: Path to a single model file for individual validation
- `--scaler_file`: Path to the scaler file
- `--output_dir`: Where to save validation results and visualizations
- `--seq_length`: Sequence length used by the model(s)
- `--use_stats`: Include if models were trained with statistical features
- `--legacy_format`: Use if validation dataset is in legacy format
- `--plots`: (v6 only) Generate publication-ready plots

#### Integrated Testing and Validation

Use the enhanced test_all_models.sh script to run both testing and validation:

```bash
./test_all_models.sh --validate --prepare-validation --validation-sample 0.2
```

For version 6:
```bash
./test_all_models_v6.sh --models-dir ./forecasting_models_v6 --test-dir ./final_complete_dataset/test_set --validate --validation-folder ./validation_dataset --use-stats
```

Options:
- `--validate`: Enable validation after regular testing
- `--prepare-validation`: Create validation dataset before testing
- `--validation-sample <ratio>`: Sampling ratio for validation dataset
- `--validation-seed <num>`: Random seed for reproducible validation
- `--validation-folder <path>`: Custom validation dataset location
- `--validation-output <path>`: Custom validation results location

#### Validation Outputs

The validation process generates:

1. **Validation report** (validation_report.txt):
   - Comprehensive metrics for each model
   - Rankings based on various metrics (RMSE, MAE, R², bias)
   - Detailed analysis of largest errors

2. **Visualizations**:
   - Scatter plots of predictions vs ground truth
   - Error distribution histograms
   - Error vs ground truth plots
   - Top files with largest errors
   - Comparative visualizations across models
   - (v6) Publication-ready figures including KDE plots and comprehensive error bars

3. **CSV files** for further analysis:
   - validation_summary.csv: Summary metrics for all models
   - model_detailed_results.csv: File-by-file predictions and errors

## Contributing

We welcome contributions! Please submit issues and pull requests on GitHub.

## License

This project is licensed under the MIT License.

## Contact

For questions, feedback, or collaboration, please contact:
Dimitrios Kafetzis  (dimitrioskafetzis@gmail.com and kafetzis@aueb.gr)