#!/usr/bin/env python3
"""
Filename: generate_model_report_v6.py
Description:
    This script generates a comprehensive summary report for time series forecasting models.
    It reads from a CSV file containing evaluation metrics for different models and produces
    a report with rankings, comparative analysis, and performance insights.
    
    Version 6 fixes an issue with model type calculations and handles potential data type issues.

Usage:
    python3 generate_model_report_v6.py input_metrics.csv output_report.txt
    
Example:
    python3 generate_model_report_v6.py model_evaluation_results.csv model_evaluation_report.txt
"""

import sys
import os
import pandas as pd
import numpy as np
from collections import Counter

def clean_model_type(model_type):
    """Clean model type to ensure it's a simple string (not concatenated model names)"""
    # Handle common model type concatenation errors
    if 'LinearRegressor' in model_type:
        return 'LinearRegressor'
    elif 'SimpleDNN' in model_type:
        return 'SimpleDNN'
    elif 'LSTM' in model_type:
        return 'LSTM'
    elif 'GRU' in model_type:
        return 'GRU'
    elif 'Transformer' in model_type:
        return 'Transformer'
    else:
        return model_type.split('.')[0]  # Take the first part if still problematic

def generate_report(csv_path, report_path):
    """
    Generate a comprehensive evaluation report from model metrics.
    
    Parameters:
        csv_path (str): Path to the CSV file containing model metrics.
        report_path (str): Path to write the output report.
    """
    try:
        # Read CSV file with metrics
        df = pd.read_csv(csv_path)
        
        # Handle missing or malformed data
        if df.empty:
            with open(report_path, 'a') as f:
                f.write("\n\nSUMMARY REPORT\n")
                f.write("=============\n\n")
                f.write("No valid model data found. Please check the input CSV file.\n")
            print("No data found in CSV file.")
            return
        
        # Clean the model type column to ensure it's properly formatted
        if 'Model Type' in df.columns:
            df['Model Type'] = df['Model Type'].astype(str).apply(clean_model_type)
        
        # Filter out rows with invalid/error metrics
        valid_metrics_df = df.copy()
        for col in ['MSE', 'MAE', 'R2', 'Inference Latency (ms)']:
            if col in valid_metrics_df.columns:
                # Replace 'null' strings with actual NaN
                valid_metrics_df[col] = pd.to_numeric(valid_metrics_df[col], errors='coerce')
        
        # Drop rows with NaN values in numeric columns
        valid_metrics_df = valid_metrics_df.dropna(subset=['MSE', 'MAE', 'R2'])
        
        # Count model types
        if 'Model Type' in valid_metrics_df.columns:
            model_types = Counter(valid_metrics_df['Model Type'])
            model_type_summary = "\nModels by Type:\n"
            for model_type, count in model_types.items():
                model_type_summary += f"- {model_type}: {count} models\n"
        else:
            model_type_summary = "\nModel Type information not available.\n"
        
        # Create rankings
        mse_ranking = valid_metrics_df.sort_values('MSE').head(5)
        mae_ranking = valid_metrics_df.sort_values('MAE').head(5)
        r2_ranking = valid_metrics_df.sort_values('R2', ascending=False).head(5)
        
        # Latency ranking if available
        if 'Inference Latency (ms)' in valid_metrics_df.columns:
            latency_ranking = valid_metrics_df.sort_values('Inference Latency (ms)').head(5)
        else:
            latency_ranking = None
        
        # Compute a composite score (lower MSE & MAE, higher R2, lower latency)
        # Normalize each metric to 0-1 range so they have equal weight
        valid_metrics_df['MSE_norm'] = (valid_metrics_df['MSE'] - valid_metrics_df['MSE'].min()) / (valid_metrics_df['MSE'].max() - valid_metrics_df['MSE'].min() + 1e-10)
        valid_metrics_df['MAE_norm'] = (valid_metrics_df['MAE'] - valid_metrics_df['MAE'].min()) / (valid_metrics_df['MAE'].max() - valid_metrics_df['MAE'].min() + 1e-10)
        valid_metrics_df['R2_norm'] = 1 - (valid_metrics_df['R2'] - valid_metrics_df['R2'].min()) / (valid_metrics_df['R2'].max() - valid_metrics_df['R2'].min() + 1e-10)
        
        if 'Inference Latency (ms)' in valid_metrics_df.columns:
            valid_metrics_df['Latency_norm'] = (valid_metrics_df['Inference Latency (ms)'] - valid_metrics_df['Inference Latency (ms)'].min()) / (valid_metrics_df['Inference Latency (ms)'].max() - valid_metrics_df['Inference Latency (ms)'].min() + 1e-10)
            # Composite score: lower is better for MSE, MAE, latency; higher is better for R2
            valid_metrics_df['Composite'] = 1 - (valid_metrics_df['MSE_norm'] * 0.3 + 
                                               valid_metrics_df['MAE_norm'] * 0.3 + 
                                               valid_metrics_df['R2_norm'] * 0.3 + 
                                               valid_metrics_df['Latency_norm'] * 0.1)
        else:
            # Without latency data
            valid_metrics_df['Composite'] = 1 - (valid_metrics_df['MSE_norm'] * 0.33 + 
                                              valid_metrics_df['MAE_norm'] * 0.33 + 
                                              valid_metrics_df['R2_norm'] * 0.34)
        
        composite_ranking = valid_metrics_df.sort_values('Composite', ascending=False).head(5)
        
        # Safely calculate average performance by model type
        model_type_stats = None
        if 'Model Type' in valid_metrics_df.columns and len(valid_metrics_df) > 0:
            try:
                # Explicitly convert numeric columns to float for aggregation
                for col in ['MSE', 'MAE', 'R2', 'Inference Latency (ms)']:
                    if col in valid_metrics_df.columns:
                        valid_metrics_df[col] = valid_metrics_df[col].astype(float)
                
                # Calculate mean by model type
                model_type_stats = valid_metrics_df.groupby('Model Type').agg({
                    'MSE': 'mean',
                    'MAE': 'mean',
                    'R2': 'mean',
                    'Inference Latency (ms)': 'mean' if 'Inference Latency (ms)' in valid_metrics_df.columns else 'count'
                })
            except Exception as e:
                print(f"Error calculating model type statistics: {str(e)}")
                print("Continuing with report generation without this section...")
        
        # Create the summary report section
        with open(report_path, 'a') as f:
            f.write("\n\nSUMMARY REPORT\n")
            f.write("=============\n")
            
            # Model type summary
            f.write(model_type_summary)
            
            f.write("\nModel Rankings:\n")
            
            # MSE Ranking
            f.write("\nTop 5 Models by MSE (lower is better):\n")
            for i, (idx, row) in enumerate(mse_ranking.iterrows()):
                model_type_str = f" ({row['Model Type']})" if 'Model Type' in row else ""
                f.write(f"{i+1}. {row['Model']}{model_type_str} - MSE: {row['MSE']:.6f}\n")
            
            # MAE Ranking
            f.write("\nTop 5 Models by MAE (lower is better):\n")
            for i, (idx, row) in enumerate(mae_ranking.iterrows()):
                model_type_str = f" ({row['Model Type']})" if 'Model Type' in row else ""
                f.write(f"{i+1}. {row['Model']}{model_type_str} - MAE: {row['MAE']:.6f}\n")
            
            # R2 Ranking
            f.write("\nTop 5 Models by R² Score (higher is better):\n")
            for i, (idx, row) in enumerate(r2_ranking.iterrows()):
                model_type_str = f" ({row['Model Type']})" if 'Model Type' in row else ""
                f.write(f"{i+1}. {row['Model']}{model_type_str} - R²: {row['R2']:.6f}\n")
            
            # Latency Ranking
            if latency_ranking is not None:
                f.write("\nTop 5 Models by Inference Latency (lower is better):\n")
                for i, (idx, row) in enumerate(latency_ranking.iterrows()):
                    model_type_str = f" ({row['Model Type']})" if 'Model Type' in row else ""
                    f.write(f"{i+1}. {row['Model']}{model_type_str} - Latency: {row['Inference Latency (ms)']:.2f} ms\n")
            
            # Composite Ranking
            f.write("\nBest Overall Models (ranked by composite score):\n")
            f.write("(Composite score based on normalized: MSE, MAE, R2, Inference Latency (ms))\n")
            for i, (idx, row) in enumerate(composite_ranking.iterrows()):
                model_type_str = f" ({row['Model Type']})" if 'Model Type' in row else ""
                details = f"MSE: {row['MSE']:.6f}, MAE: {row['MAE']:.6f}, R²: {row['R2']:.6f}"
                if 'Inference Latency (ms)' in row:
                    details += f", Latency: {row['Inference Latency (ms)']:.2f} ms"
                f.write(f"{i+1}. {row['Model']}{model_type_str} - Score: {row['Composite']:.4f} ({details})\n")
            
            # Model Type Performance Averages
            f.write("\nAverage Performance by Model Type:\n")
            if model_type_stats is not None and len(model_type_stats) > 0:
                for model_type, stats in model_type_stats.iterrows():
                    f.write(f"- {model_type}:\n")
                    for metric, value in stats.items():
                        if metric in ['MSE', 'MAE', 'R2']:
                            f.write(f"  {metric}: {value:.6f}\n")
                        elif metric == 'Inference Latency (ms)':
                            f.write(f"  Inference Latency: {value:.2f} ms\n")
            else:
                f.write("  (Not available or insufficient data)\n")
        
        print("Report generation complete!")
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        # Add error information to the report
        with open(report_path, 'a') as f:
            f.write("\n\nERROR IN REPORT GENERATION\n")
            f.write(f"An error occurred during report generation: {str(e)}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 generate_model_report_v6.py input_metrics.csv output_report.txt")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    report_path = sys.argv[2]
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    generate_report(csv_path, report_path)
