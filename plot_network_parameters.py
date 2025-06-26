#!/usr/bin/env python3
"""
Network Parameters Time Series Visualization Script

This script processes a directory of JSON files containing network metrics data,
extracts timestamps and parameters, and generates time series bar graphs
with gaps explicitly marked.

Usage:
    python plot_network_parameters.py --input_dir /path/to/json/files --output_dir /path/to/save/plots

Parameters:
    --input_dir: Directory containing JSON files
    --output_dir: Directory to save generated plots
    --gap_threshold: Threshold in seconds to consider a gap significant (default: 60)
    --date_format: Format string for date labels (default: '%Y-%m-%d %H:%M')
    --figsize: Figure size in inches as width,height (default: 15,6)
    --dpi: Resolution of saved plots (default: 150)
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.ticker import MaxNLocator

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate time series plots for network parameters')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save plots')
    parser.add_argument('--gap_threshold', type=int, default=60, 
                        help='Threshold in seconds to consider a gap significant (default: 60)')
    parser.add_argument('--date_format', type=str, default='%Y-%m-%d %H:%M', 
                        help='Format string for date labels')
    parser.add_argument('--figsize', type=str, default='15,6', 
                        help='Figure size in inches as width,height')
    parser.add_argument('--dpi', type=int, default=150, 
                        help='Resolution of saved plots')
    return parser.parse_args()

def load_json_files(directory):
    """
    Load all JSON files from the directory and extract their data.
    
    Returns:
        A dictionary with parameter names as keys and lists of (timestamp, value) tuples as values.
    """
    # Initialize data structures
    data = {
        'timestamp': [],
        'QoE': []
    }
    
    # Parameters to extract (we'll gather them from the JSON files)
    params = set()
    
    # Find all JSON files
    json_files = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {directory}")
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    # Process each JSON file
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                file_data = json.load(f)
            
            # Check if the file has the expected structure
            if 'timestamp' not in file_data or 'QoE' not in file_data:
                print(f"Warning: File {file_path} missing required fields. Skipping.")
                continue
            
            # Extract the main timestamp and QoE
            main_timestamp = int(file_data['timestamp'])
            dt = datetime.strptime(str(main_timestamp), '%Y%m%d%H%M%S')
            data['timestamp'].append(dt)
            data['QoE'].append(float(file_data['QoE']))
            
            # Extract all parameters from the most recent measurement within the file
            # (usually the one with the same timestamp as the main timestamp)
            measurement_keys = [k for k in file_data.keys() 
                               if k.isdigit() and k != 'timestamp']
            
            if measurement_keys:
                # Use the last measurement in each file (usually the most recent)
                last_measurement_key = sorted(measurement_keys)[-1]
                last_measurement = file_data[last_measurement_key]
                
                # Add all parameters to the data dictionary
                for param, value in last_measurement.items():
                    if param not in data:
                        data[param] = []
                        params.add(param)
                    data[param].append(float(value))
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"Extracted data for parameters: QoE, {', '.join(params)}")
    return data, params

def identify_gaps(timestamps, threshold_seconds):
    """
    Identify time gaps larger than the threshold.
    
    Args:
        timestamps: List of datetime objects
        threshold_seconds: Threshold in seconds to consider a gap
        
    Returns:
        List of (timestamp, gap_size) tuples where gaps were detected
    """
    timestamps_sorted = sorted(timestamps)
    gaps = []
    
    for i in range(1, len(timestamps_sorted)):
        time_diff = (timestamps_sorted[i] - timestamps_sorted[i-1]).total_seconds()
        if time_diff > threshold_seconds:
            # Store the timestamp where the gap ends and the gap size
            gaps.append((timestamps_sorted[i], time_diff))
    
    return gaps

def format_time_diff(seconds):
    """Format time difference in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"

def plot_parameter(data, param, output_path, gaps, args):
    """
    Create a bar plot for a specific parameter.
    
    Args:
        data: Dictionary containing the data
        param: Parameter name to plot
        output_path: Path to save the plot
        gaps: List of (timestamp, gap_size) tuples where gaps were detected
        args: Command-line arguments
    """
    figsize = tuple(map(float, args.figsize.split(',')))
    plt.figure(figsize=figsize)
    
    # Create the bar plot
    timestamps = data['timestamp']
    values = data[param]
    
    # Determine bar width based on typical time difference
    time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                  for i in range(len(timestamps)-1)]
    if time_diffs:
        # Use the median time difference to determine bar width
        typical_diff = np.median(time_diffs)
        # Convert to days for matplotlib (which uses days as units)
        width = max(typical_diff / 86400 * 0.8, 0.00001)  # At least some minimal width
    else:
        width = 0.01  # Default if we can't determine
    
    # Create the bar plot
    bars = plt.bar(timestamps, values, width=width, alpha=0.7, 
             color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Mark gaps with red dots on the x-axis
    if gaps:
        gap_times = [gap[0] for gap in gaps]
        plt.scatter(gap_times, [0] * len(gap_times), color='red', s=50, 
                   zorder=5, label='Time Gap')
        
        # Add annotations for significant gaps
        for gap_time, gap_size in gaps:
            if gap_size > args.gap_threshold * 10:  # Only annotate very large gaps
                plt.annotate(
                    f"Gap: {format_time_diff(gap_size)}",
                    xy=(gap_time, 0),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
    
    # Format the plot
    plt.title(f"{param} over Time", fontsize=14, fontweight='bold')
    plt.ylabel(param, fontsize=12)
    plt.xlabel("Date & Time", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format the x-axis date labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(args.date_format))
    plt.gcf().autofmt_xdate(rotation=45)
    
    # Add y-axis locator to ensure readable ticks
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10, integer=False))
    
    # Add legend if we have gaps
    if gaps:
        plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi)
    plt.close()
    
    print(f"Saved plot for {param} to {output_path}")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input_dir}...")
    data, params = load_json_files(args.input_dir)
    
    # Check if we found any data
    if not data['timestamp']:
        print("No valid data found in the JSON files!")
        return
    
    print(f"Found data for {len(data['timestamp'])} timestamps")
    
    # Identify time gaps
    gaps = identify_gaps(data['timestamp'], args.gap_threshold)
    if gaps:
        print(f"Identified {len(gaps)} time gaps larger than {args.gap_threshold} seconds:")
        for gap_time, gap_size in gaps[:5]:  # Show first 5 gaps
            print(f"  Gap at {gap_time.strftime('%Y-%m-%d %H:%M:%S')}: {format_time_diff(gap_size)}")
        if len(gaps) > 5:
            print(f"  ... and {len(gaps) - 5} more gaps")
    else:
        print("No significant time gaps identified")
    
    # Plot QoE
    print("Generating plots...")
    plot_parameter(
        data, 
        'QoE', 
        os.path.join(args.output_dir, 'QoE_vs_time.png'),
        gaps,
        args
    )
    
    # Plot each network parameter
    for param in params:
        output_path = os.path.join(args.output_dir, f'{param}_vs_time.png')
        plot_parameter(data, param, output_path, gaps, args)
    
    print(f"All plots have been saved to {args.output_dir}")
    print("Parameters plotted: QoE, " + ", ".join(params))

if __name__ == "__main__":
    main()
