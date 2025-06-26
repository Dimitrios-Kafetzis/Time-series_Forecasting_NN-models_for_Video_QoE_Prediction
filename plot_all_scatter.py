#!/usr/bin/env python3
"""
Filename: plot_all_scatter.py
Description:
    This script generates high-quality scatter plots of Predicted QoE vs Ground Truth QoE
    for all models in a validation results directory.
    
    It reads the detailed CSV files created by validate_models_v6.py and creates
    publication-ready scatter plots for each model, saving them to a figures directory.

Usage:
    python3 plot_all_scatter.py --results_dir /path/to/validation_results --output_dir /path/to/output
    
Example:
    python3 plot_all_scatter.py \
        --results_dir ~/Impact-xG_prediction_model/forecasting_models_v6_complete_dataset/validation_results \
        --output_dir ~/Impact-xG_prediction_model/forecasting_models_v6_complete_dataset/validation_plots
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def create_scatter_plot(data_df, model_name, output_dir, model_type=None):
    """Create a high-quality scatter plot for a single model."""
    plt.figure(figsize=(10, 8))
    
    # Set a consistent style
    sns.set_style("whitegrid")
    
    # Create the scatter plot
    scatter = sns.scatterplot(
        x='Ground Truth', 
        y='Prediction', 
        data=data_df, 
        alpha=0.7,
        s=60,  # Slightly larger points
        edgecolor='k',  # Black edge for better visibility
        linewidth=0.5,
    )
    
    # Add perfect prediction line
    min_val = min(data_df['Ground Truth'].min(), data_df['Prediction'].min())
    max_val = max(data_df['Ground Truth'].max(), data_df['Prediction'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # Calculate metrics for the plot title
    mse = ((data_df['Prediction'] - data_df['Ground Truth'])**2).mean()
    rmse = np.sqrt(mse)
    r2 = 1 - (((data_df['Prediction'] - data_df['Ground Truth'])**2).sum() / 
             ((data_df['Ground Truth'] - data_df['Ground Truth'].mean())**2).sum())
    
    # Create title with metrics
    model_display = model_name.replace('.h5', '')
    if model_type:
        title = f"{model_display} ({model_type}) - Predicted vs Ground Truth QoE\nRMSE: {rmse:.4f}, R²: {r2:.4f}"
    else:
        title = f"{model_display} - Predicted vs Ground Truth QoE\nRMSE: {rmse:.4f}, R²: {r2:.4f}"
    
    plt.title(title, fontsize=14)
    plt.xlabel('Ground Truth QoE', fontsize=12)
    plt.ylabel('Predicted QoE', fontsize=12)
    
    # Make axes equal for better visual comparison
    plt.axis('equal')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Tighten layout and save
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(output_dir, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # Save the figure with high resolution
    clean_name = model_name.replace('.h5', '').replace(' ', '_')
    plt.savefig(os.path.join(figures_dir, f"{clean_name}_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created scatter plot for {model_name}")

def create_combined_scatter_plot(results_data, output_dir, max_models=5):
    """Create a combined scatter plot with the top performing models."""
    # Sort models by R2 score
    models_by_r2 = sorted(
        [(name, df) for name, df in results_data.items()],
        key=lambda x: 1 - (((x[1]['Prediction'] - x[1]['Ground Truth'])**2).sum() / 
                         ((x[1]['Ground Truth'] - x[1]['Ground Truth'].mean())**2).sum()),
        reverse=True
    )
    
    # Select top models (to avoid overcrowding the plot)
    top_models = models_by_r2[:max_models]
    
    plt.figure(figsize=(12, 10))
    
    # Set a consistent style
    sns.set_style("whitegrid")
    
    # Use a different color for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Find global min and max for consistent axes
    all_min = min([min(df['Ground Truth'].min(), df['Prediction'].min()) for _, df in top_models])
    all_max = max([max(df['Ground Truth'].max(), df['Prediction'].max()) for _, df in top_models])
    
    # Plot data for each model
    for i, (model_name, df) in enumerate(top_models):
        model_display = model_name.replace('.h5', '')
        plt.scatter(
            df['Ground Truth'], 
            df['Prediction'],
            alpha=0.7,
            s=50,
            edgecolor='k',
            linewidth=0.5,
            color=colors[i % len(colors)],
            label=model_display
        )
    
    # Add perfect prediction line
    plt.plot([all_min, all_max], [all_min, all_max], 'k--', linewidth=2, label='Perfect Prediction')
    
    plt.title("Top Models: Predicted vs Ground Truth QoE", fontsize=15)
    plt.xlabel('Ground Truth QoE', fontsize=13)
    plt.ylabel('Predicted QoE', fontsize=13)
    
    # Make axes equal for better visual comparison
    plt.axis('equal')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Tighten layout and save
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(output_dir, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # Save the figure with high resolution
    plt.savefig(os.path.join(figures_dir, "combined_top_models_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created combined scatter plot for top {len(top_models)} models")

def extract_model_type(results_dir):
    """Try to extract model types from the validation summary file if available."""
    model_types = {}
    summary_path = os.path.join(results_dir, "validation_summary.csv")
    
    if os.path.exists(summary_path):
        try:
            summary_df = pd.read_csv(summary_path)
            if "Model" in summary_df.columns and "Type" in summary_df.columns:
                for _, row in summary_df.iterrows():
                    model_types[row["Model"]] = row["Type"]
        except Exception as e:
            print(f"Warning: Could not read model types from summary file: {e}")
    
    return model_types

def main():
    parser = argparse.ArgumentParser(description='Generate high-quality scatter plots for all models in validation results.')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing validation results (CSV files)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save output plots (default: same as results_dir)')
    parser.add_argument('--max_models', type=int, default=5,
                       help='Maximum number of models to include in the combined plot (default: 5)')
    
    args = parser.parse_args()
    
    # Set output directory to results_dir if not specified
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Find all detailed results CSV files
    csv_files = glob(os.path.join(args.results_dir, "*_detailed_results.csv"))
    
    if not csv_files:
        print(f"No detailed results CSV files found in {args.results_dir}")
        print("Please run validate_models_v6.py first to generate these files.")
        sys.exit(1)
    
    # Try to get model types from summary file
    model_types = extract_model_type(args.results_dir)
    
    # Read all data files
    results_data = {}
    for csv_file in csv_files:
        try:
            model_name = os.path.basename(csv_file).replace("_detailed_results.csv", "")
            df = pd.read_csv(csv_file)
            results_data[model_name] = df
            
            # Create individual scatter plot
            model_type = model_types.get(model_name)
            create_scatter_plot(df, model_name, args.output_dir, model_type)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    # Create combined plot with top models
    if len(results_data) > 1:
        create_combined_scatter_plot(results_data, args.output_dir, args.max_models)
    
    print(f"All scatter plots have been saved to {os.path.join(args.output_dir, 'figures')}")

if __name__ == "__main__":
    main()
