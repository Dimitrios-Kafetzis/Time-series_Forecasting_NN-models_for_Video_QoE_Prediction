#!/bin/bash
# deployment.sh - Script to deploy QoE Prediction Model to a Linux machine
# Author: Based on Dimitrios Kafetzis's QoE prediction project

# Create the base directory structure
echo "Creating directory structure..."
mkdir -p qoe_prediction_model/{models,data/{mock_dataset,augmented_dataset,inference_inputs},scripts}

# Copy the Python scripts
echo "Copying Python scripts..."
cp run_inference.py qoe_prediction_model/scripts/
cp timeseries_forecasting_models_v5.py qoe_prediction_model/scripts/

# Create requirements.txt
echo "Creating requirements.txt..."
cat > qoe_prediction_model/requirements.txt << EOL
numpy>=1.19.2
pandas>=1.1.3
tensorflow==2.10.0
joblib>=0.17.0
scikit-learn>=0.24.0
keras-tuner>=1.0.4
EOL

# Create a sample README.md
echo "Creating README.md..."
cat > qoe_prediction_model/README.md << EOL
# QoE Prediction Model

This package contains a trained model for Quality of Experience (QoE) prediction based on network metrics.

## Deployment Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
1. Clone or extract this package to your Linux machine
2. Navigate to the project directory:
   \`\`\`
   cd qoe_prediction_model
   \`\`\`
3. Create a virtual environment (recommended):
   \`\`\`
   python3 -m venv venv
   source venv/bin/activate
   \`\`\`
4. Install the required dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

## Usage

### Standard Mode
For standard inference (one JSON file per 10-second timepoint):

\`\`\`bash
python3 scripts/run_inference.py --inference_file ./data/inference_inputs/20250204123000.json \\
    --data_folder ./data/mock_dataset --seq_length 5 \\
    --model_file ./models/model_transformer.h5 \\
    --scaler_file ./models/scaler.save
\`\`\`

### Augmented Mode
For augmented inference (one JSON file per 5-second window with per-second sub-records):

\`\`\`bash
python3 scripts/run_inference.py --inference_file ./data/inference_inputs/20250402122044.json \\
    --data_folder ./data/augmented_dataset --seq_length 5 \\
    --model_file ./models/model_transformer.h5 \\
    --scaler_file ./models/scaler.save --augmented
\`\`\`

See README_FULL.md for complete documentation.
EOL

# Create a sample JSON file for inference
echo "Creating sample JSON files..."
mkdir -p qoe_prediction_model/data/inference_inputs

cat > qoe_prediction_model/data/inference_inputs/20250402122044.json << EOL
{
    "QoE": 43.132408,
    "timestamp": 20250402122044,
    "20250402122036": {
        "throughput": 1517.7,
        "packets_lost": 4.0,
        "packet_loss_rate": 1.2,
        "jitter": 15.201,
        "speed": 0
    },
    "20250402122038": {
        "throughput": 1427.9,
        "packets_lost": 0.0,
        "packet_loss_rate": 0.0,
        "jitter": 15.92,
        "speed": 0
    },
    "20250402122040": {
        "throughput": 605.2,
        "packets_lost": 1.0,
        "packet_loss_rate": 0.8,
        "jitter": 20.682,
        "speed": 0
    },
    "20250402122042": {
        "throughput": 969.2,
        "packets_lost": 1.0,
        "packet_loss_rate": 0.5,
        "jitter": 22.78,
        "speed": 0
    },
    "20250402122044": {
        "throughput": 1262.4,
        "packets_lost": 0.0,
        "packet_loss_rate": 0.0,
        "jitter": 17.601,
        "speed": 0
    }
}
EOL

# Create model placeholder messages
echo "IMPORTANT: Add your model files to the models directory" > qoe_prediction_model/models/README.txt
echo "Add your model_transformer.h5 and scaler.save files here" >> qoe_prediction_model/models/README.txt

# Create a compressed archive
echo "Creating tarball..."
tar -czf qoe_prediction_model.tar.gz qoe_prediction_model

echo "Deployment package created: qoe_prediction_model.tar.gz"
echo "Before deployment, add your model files to the models directory."
