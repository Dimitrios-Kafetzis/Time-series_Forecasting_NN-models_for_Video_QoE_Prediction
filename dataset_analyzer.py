#!/usr/bin/env python3
# Network Metrics Analysis Script
# This script analyzes network performance metrics from JSON files and generates a report with visualizations

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze network metrics from JSON files')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the report and plots')
    parser.add_argument('--analyze_time_gaps', action='store_true', help='Enable detailed analysis of time gaps in the dataset')
    return parser.parse_args()

def load_json_files(directory):
    """Load all JSON files from the specified directory into a single DataFrame."""
    all_data = []
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {directory}")
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract the base metrics (QoE and timestamp)
            file_data = {
                'file': os.path.basename(file_path),
                'QoE': data.get('QoE'),
                'timestamp': data.get('timestamp')
            }
            
            # Process each timestamp entry (excluding the base metrics)
            for timestamp, metrics in data.items():
                if timestamp not in ['QoE', 'timestamp']:
                    timestamp_data = file_data.copy()
                    timestamp_data['measurement_timestamp'] = timestamp
                    timestamp_data.update(metrics)
                    all_data.append(timestamp_data)
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if not all_data:
        raise ValueError("No valid data could be extracted from the JSON files")
    
    df = pd.DataFrame(all_data)
    return df

def calculate_basic_statistics(df):
    """Calculate basic statistics for each metric."""
    # Identify numeric columns (metrics)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove timestamp columns from numeric analysis
    metrics = [col for col in numeric_cols if not ('timestamp' in col.lower())]
    
    stats_df = df[metrics].describe(percentiles=[.05, .25, .5, .75, .95]).T
    stats_df['skewness'] = df[metrics].skew()
    stats_df['kurtosis'] = df[metrics].kurt()
    
    return stats_df, metrics

def generate_distribution_plots(df, metrics, output_dir):
    """Generate distribution plots for each metric."""
    os.makedirs(os.path.join(output_dir, 'distributions'), exist_ok=True)
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # Create subplot with 1 row and 2 columns
        plt.subplot(1, 2, 1)
        sns.histplot(df[metric], kde=True)
        plt.title(f'Histogram of {metric}')
        plt.grid(True, alpha=0.3)
        
        # Add boxplot to show outliers and quartiles
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[metric])
        plt.title(f'Boxplot of {metric}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distributions', f'{metric}_distribution.png'))
        plt.close()

def generate_range_coverage_analysis(df, metrics, output_dir):
    """Analyze the coverage of different value ranges for each metric."""
    os.makedirs(os.path.join(output_dir, 'range_coverage'), exist_ok=True)
    
    range_coverage = {}
    
    for metric in metrics:
        # Determine appropriate number of bins based on data
        data = df[metric].dropna()
        if len(data.unique()) < 10:  # For metrics with few unique values
            bins = len(data.unique())
        else:
            bins = min(30, int(np.sqrt(len(data))))  # Rule of thumb for bin count
        
        # Calculate histogram
        hist, bin_edges = np.histogram(data, bins=bins)
        
        # Calculate bin coverage percentage
        total_samples = len(data)
        bin_coverage = (hist / total_samples) * 100
        
        # Save data for the report
        range_coverage[metric] = {
            'bin_edges': bin_edges,
            'bin_counts': hist,
            'bin_coverage': bin_coverage,
            'empty_bins': sum(hist == 0),
            'total_bins': len(hist),
            'coverage_percentage': (sum(hist > 0) / len(hist)) * 100
        }
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot histogram with bin coverage
        plt.bar(range(len(hist)), bin_coverage, width=0.8)
        plt.xlabel(f'{metric} Range Bins')
        plt.ylabel('Coverage Percentage (%)')
        plt.title(f'Range Coverage Analysis for {metric}')
        plt.grid(True, alpha=0.3)
        
        # Add bin edges as x-tick labels, but only show a subset for readability
        if len(bin_edges) > 10:
            # Show fewer tick labels
            tick_indices = np.linspace(0, len(hist)-1, 10, dtype=int)
            plt.xticks(tick_indices, [f'{bin_edges[i]:.2f}' for i in tick_indices])
        else:
            plt.xticks(range(len(hist)), [f'{edge:.2f}' for edge in bin_edges[:-1]])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'range_coverage', f'{metric}_range_coverage.png'))
        plt.close()
    
    return range_coverage

def generate_correlation_analysis(df, metrics, output_dir):
    """Analyze correlations between metrics."""
    # Create correlation matrix
    corr_matrix = df[metrics].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
    plt.title('Correlation Matrix of Network Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    # Focus on correlations with QoE if it exists
    if 'QoE' in metrics:
        plt.figure(figsize=(10, 6))
        qoe_corr = corr_matrix['QoE'].drop('QoE').sort_values(ascending=False)
        sns.barplot(x=qoe_corr.values, y=qoe_corr.index)
        plt.title('Correlation of Metrics with QoE')
        plt.xlabel('Correlation Coefficient')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'qoe_correlations.png'))
        plt.close()
    
    return corr_matrix

def generate_pairplot(df, metrics, output_dir):
    """Generate pairplot to visualize relationships between metrics."""
    # Select a subset of important metrics to avoid overwhelming visuals
    if 'QoE' in metrics:
        # Find the top correlating metrics with QoE
        corr_with_qoe = df[metrics].corr()['QoE'].abs().sort_values(ascending=False)
        key_metrics = ['QoE'] + list(corr_with_qoe.drop('QoE').head(3).index)
    else:
        # If QoE not in metrics, just take a few key ones
        key_metrics = metrics[:4]  # Take first 4 metrics
    
    # Create pairplot
    plt.figure(figsize=(12, 12))
    pair_plot = sns.pairplot(df[key_metrics], diag_kind='kde', height=2.5)
    pair_plot.fig.suptitle('Relationships Between Key Metrics', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_pairplot.png'))
    plt.close()

def generate_multivariate_distribution(df, output_dir):
    """Visualize multivariate distributions of key metrics."""
    # Select key metrics for analysis
    if 'throughput' in df.columns and 'jitter' in df.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='throughput', y='jitter', hue='QoE' if 'QoE' in df.columns else None, 
                         alpha=0.7, palette='viridis')
        plt.title('Distribution of Network Conditions (Throughput vs Jitter)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'throughput_jitter_distribution.png'))
        plt.close()
    
    # If we have packet loss data
    if 'packet_loss_rate' in df.columns and 'throughput' in df.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='throughput', y='packet_loss_rate', 
                         hue='QoE' if 'QoE' in df.columns else None,
                         alpha=0.7, palette='viridis')
        plt.title('Distribution of Network Conditions (Throughput vs Packet Loss)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'throughput_packetloss_distribution.png'))
        plt.close()

def check_for_gaps(range_coverage, output_dir):
    """Identify gaps in the data coverage."""
    gap_report = {}
    
    for metric, coverage in range_coverage.items():
        # Calculate gap statistics
        total_bins = coverage['total_bins']
        empty_bins = coverage['empty_bins']
        gap_percentage = (empty_bins / total_bins) * 100 if total_bins > 0 else 0
        
        # Identify specific gap regions
        bin_edges = coverage['bin_edges']
        bin_counts = coverage['bin_counts']
        gaps = []
        
        in_gap = False
        gap_start = None
        
        for i, count in enumerate(bin_counts):
            if count == 0 and not in_gap:
                in_gap = True
                gap_start = bin_edges[i]
            elif count > 0 and in_gap:
                in_gap = False
                gap_end = bin_edges[i]
                gaps.append((gap_start, gap_end))
        
        # If we ended in a gap, close it
        if in_gap:
            gaps.append((gap_start, bin_edges[-1]))
        
        gap_report[metric] = {
            'gap_percentage': gap_percentage,
            'specific_gaps': gaps
        }
    
    return gap_report

def generate_report(df, stats_df, corr_matrix, range_coverage, gap_report, output_dir):
    """Generate a comprehensive text report with analysis findings."""
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        # Header
        f.write("=================================================\n")
        f.write("    NETWORK METRICS DATASET ANALYSIS REPORT      \n")
        f.write("=================================================\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset Overview
        f.write("1. DATASET OVERVIEW\n")
        f.write("===================\n\n")
        f.write(f"Total number of data points: {len(df)}\n")
        f.write(f"Number of unique timestamps: {df['measurement_timestamp'].nunique()}\n")
        f.write(f"Time range: {df['measurement_timestamp'].min()} to {df['measurement_timestamp'].max()}\n\n")
        
        # Basic Statistics
        f.write("2. BASIC STATISTICS\n")
        f.write("===================\n\n")
        f.write(stats_df.to_string())
        f.write("\n\n")
        
        # Distribution Analysis
        f.write("3. DISTRIBUTION ANALYSIS\n")
        f.write("=======================\n\n")
        
        for metric in stats_df.index:
            f.write(f"3.{list(stats_df.index).index(metric) + 1}. {metric.upper()}\n")
            f.write("-" * (len(metric) + 6) + "\n\n")
            
            # Add distributional characteristics
            f.write(f"Mean: {stats_df.loc[metric, 'mean']:.4f}\n")
            f.write(f"Median: {stats_df.loc[metric, '50%']:.4f}\n")
            f.write(f"Standard Deviation: {stats_df.loc[metric, 'std']:.4f}\n")
            f.write(f"Skewness: {stats_df.loc[metric, 'skewness']:.4f} ")
            
            # Interpret skewness
            skew = stats_df.loc[metric, 'skewness']
            if abs(skew) < 0.5:
                f.write("(approximately symmetric)\n")
            elif skew < 0:
                f.write("(negatively skewed)\n")
            else:
                f.write("(positively skewed)\n")
            
            f.write(f"Kurtosis: {stats_df.loc[metric, 'kurtosis']:.4f} ")
            
            # Interpret kurtosis
            kurt = stats_df.loc[metric, 'kurtosis']
            if abs(kurt) < 0.5:
                f.write("(approximately mesokurtic - normal distribution)\n")
            elif kurt < 0:
                f.write("(platykurtic - flatter than normal distribution)\n")
            else:
                f.write("(leptokurtic - more peaked than normal distribution)\n")
            
            f.write(f"Range: {stats_df.loc[metric, 'min']:.4f} to {stats_df.loc[metric, 'max']:.4f}\n")
            f.write(f"IQR (Interquartile Range): {stats_df.loc[metric, '75%'] - stats_df.loc[metric, '25%']:.4f}\n\n")
            
            # Add normality test results
            _, p_value = stats.normaltest(df[metric].dropna())
            f.write(f"Normality Test p-value: {p_value:.6f}\n")
            if p_value < 0.05:
                f.write("The distribution is significantly different from a normal distribution.\n\n")
            else:
                f.write("The distribution is not significantly different from a normal distribution.\n\n")
            
            # Add range coverage information
            coverage = range_coverage[metric]
            f.write(f"Range Coverage: {coverage['coverage_percentage']:.2f}% of possible value ranges have data\n")
            f.write(f"Empty Ranges: {coverage['empty_bins']} out of {coverage['total_bins']} bins have no data\n\n")
            
            # Add gap information
            if gap_report[metric]['specific_gaps']:
                f.write("Identified gaps in data coverage:\n")
                for i, (start, end) in enumerate(gap_report[metric]['specific_gaps']):
                    f.write(f"  Gap {i+1}: {start:.4f} to {end:.4f}\n")
                f.write("\n")
            else:
                f.write("No significant gaps identified in the data coverage.\n\n")
        
        # Correlation Analysis
        f.write("4. CORRELATION ANALYSIS\n")
        f.write("======================\n\n")
        f.write("Correlation Matrix:\n")
        f.write(corr_matrix.to_string())
        f.write("\n\n")
        
        # If QoE is present, add QoE-specific analysis
        if 'QoE' in corr_matrix.columns:
            f.write("4.1. QOE CORRELATION ANALYSIS\n")
            f.write("---------------------------\n\n")
            
            qoe_corr = corr_matrix['QoE'].drop('QoE').sort_values(ascending=False)
            
            f.write("Correlations with QoE (from strongest positive to strongest negative):\n")
            for metric, corr in qoe_corr.items():
                f.write(f"{metric}: {corr:.4f} - ")
                
                if abs(corr) < 0.2:
                    f.write("Very weak correlation\n")
                elif abs(corr) < 0.4:
                    f.write("Weak correlation\n")
                elif abs(corr) < 0.6:
                    f.write("Moderate correlation\n")
                elif abs(corr) < 0.8:
                    f.write("Strong correlation\n")
                else:
                    f.write("Very strong correlation\n")
            
            f.write("\n")
        
        # Data Diversity Assessment
        f.write("5. DATA DIVERSITY ASSESSMENT\n")
        f.write("==========================\n\n")
        
        f.write("Overall Assessment:\n")
        
        # Calculate overall coverage
        total_bins = sum(cov['total_bins'] for cov in range_coverage.values())
        total_empty_bins = sum(cov['empty_bins'] for cov in range_coverage.values())
        overall_coverage = ((total_bins - total_empty_bins) / total_bins) * 100 if total_bins > 0 else 0
        
        f.write(f"Overall range coverage: {overall_coverage:.2f}%\n")
        
        if overall_coverage > 80:
            f.write("The dataset has excellent coverage across most metrics.\n")
        elif overall_coverage > 60:
            f.write("The dataset has good coverage but with some gaps.\n")
        else:
            f.write("The dataset has significant gaps in coverage that may impact model training.\n")
        
        f.write("\nKey metrics with poorest coverage:\n")
        coverage_by_metric = {m: (1 - c['empty_bins']/c['total_bins'])*100 for m, c in range_coverage.items()}
        worst_covered = sorted(coverage_by_metric.items(), key=lambda x: x[1])[:3]
        
        for metric, coverage in worst_covered:
            f.write(f"- {metric}: {coverage:.2f}% coverage\n")
        
        f.write("\n")
        
        # Conclusion and Recommendations
        f.write("6. CONCLUSION AND RECOMMENDATIONS\n")
        f.write("===============================\n\n")
        
        # Overall data quality assessment
        if overall_coverage > 70 and df['QoE'].nunique() > 10:
            f.write("The dataset appears to have good diversity and coverage for training QoE prediction models.\n")
        else:
            f.write("The dataset has some limitations that should be addressed before using it for training:\n")
            
            if overall_coverage <= 70:
                f.write("- Limited coverage across the full range of possible network conditions\n")
            
            if 'QoE' in df.columns and df['QoE'].nunique() <= 10:
                f.write("- Limited diversity in QoE values, which may limit the model's ability to predict across the full QoE range\n")
        
        f.write("\nSpecific recommendations for improving the dataset:\n")
        
        # Generate specific recommendations
        recommendations = []
        
        # Check for imbalanced QoE distribution
        if 'QoE' in df.columns:
            qoe_counts = df['QoE'].value_counts(normalize=True)
            most_common = qoe_counts.idxmax()
            if qoe_counts.iloc[0] > 0.5:  # If more than 50% of data has the same QoE value
                recommendations.append(
                    f"Collect more data for QoE values other than {most_common:.4f}, which currently dominates the dataset"
                )
        
        # Check for gaps in key metrics
        for metric, gap_info in gap_report.items():
            if gap_info['gap_percentage'] > 30:  # If more than 30% of the range has no data
                recommendations.append(
                    f"Collect more data for {metric} in the identified gap regions"
                )
        
        # Add recommendations to report
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
        else:
            f.write("- No specific recommendations needed - the dataset appears well-suited for model training.\n")
        
        f.write("\n\n")
        f.write("=================================================\n")
        f.write("                END OF REPORT                   \n")
        f.write("=================================================\n")
    
    return report_path

def analyze_time_gaps(df, output_dir):
    """
    Analyze gaps in the time series data to identify potential issues
    that might affect model training and inference.
    """
    os.makedirs(os.path.join(output_dir, 'time_gaps'), exist_ok=True)
    
    # Extract unique timestamps and sort them
    timestamps = sorted(df['timestamp'].unique())
    
    # Calculate time differences between consecutive timestamps (in seconds)
    time_diffs = []
    for i in range(1, len(timestamps)):
        # Convert string timestamps to integers if necessary
        current = int(str(timestamps[i]))
        previous = int(str(timestamps[i-1]))
        
        # Calculate time difference using datetime objects for more accuracy
        current_dt = datetime.strptime(str(current), '%Y%m%d%H%M%S')
        previous_dt = datetime.strptime(str(previous), '%Y%m%d%H%M%S')
        diff_seconds = (current_dt - previous_dt).total_seconds()
        
        time_diffs.append({
            'from': previous,
            'to': current,
            'gap_seconds': diff_seconds,
            'from_dt': previous_dt,
            'to_dt': current_dt
        })
    
    # Convert to DataFrame for easier analysis
    gaps_df = pd.DataFrame(time_diffs)
    
    # Calculate basic statistics about gaps
    mean_gap = gaps_df['gap_seconds'].mean()
    median_gap = gaps_df['gap_seconds'].median()
    std_gap = gaps_df['gap_seconds'].std()
    min_gap = gaps_df['gap_seconds'].min()
    max_gap = gaps_df['gap_seconds'].max()
    
    # Define what constitutes a "large gap" (e.g., 3x the median or >60 seconds)
    large_gap_threshold = max(3 * median_gap, 60)
    
    # Identify large gaps
    large_gaps = gaps_df[gaps_df['gap_seconds'] > large_gap_threshold].copy()
    large_gaps['gap_minutes'] = large_gaps['gap_seconds'] / 60
    large_gaps['gap_hours'] = large_gaps['gap_minutes'] / 60
    
    # Calculate percentage of gaps that are "large"
    large_gap_percentage = (len(large_gaps) / len(gaps_df)) * 100 if len(gaps_df) > 0 else 0
    
    # Visualize gap distribution
    plt.figure(figsize=(12, 6))
    plt.hist(gaps_df['gap_seconds'], bins=30, alpha=0.7)
    plt.axvline(median_gap, color='r', linestyle='--', label=f'Median Gap ({median_gap:.2f}s)')
    plt.axvline(large_gap_threshold, color='g', linestyle='--', label=f'Large Gap Threshold ({large_gap_threshold:.2f}s)')
    plt.xlabel('Gap Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Time Gaps Between Consecutive Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_gaps', 'gap_distribution.png'))
    plt.close()
    
    # Visualize gaps over time
    if len(gaps_df) > 1:
        plt.figure(figsize=(14, 6))
        plt.scatter(range(len(gaps_df)), gaps_df['gap_seconds'], alpha=0.7)
        plt.axhline(large_gap_threshold, color='r', linestyle='--', label=f'Large Gap Threshold ({large_gap_threshold:.2f}s)')
        plt.xlabel('Gap Index (Chronological Order)')
        plt.ylabel('Gap Duration (seconds)')
        plt.title('Time Gaps Over the Dataset Timeline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_gaps', 'gap_timeline.png'))
        plt.close()
    
    # Analyze impact on sequence generation using sequence_length=5
    sequence_length = 5
    problematic_sequences = []
    
    for i in range(len(timestamps) - sequence_length + 1):
        seq_timestamps = timestamps[i:i+sequence_length]
        
        # Check time differences within this potential sequence
        has_large_gap = False
        for j in range(1, len(seq_timestamps)):
            current_dt = datetime.strptime(str(seq_timestamps[j]), '%Y%m%d%H%M%S')
            previous_dt = datetime.strptime(str(seq_timestamps[j-1]), '%Y%m%d%H%M%S')
            diff_seconds = (current_dt - previous_dt).total_seconds()
            
            if diff_seconds > large_gap_threshold:
                has_large_gap = True
                break
        
        if has_large_gap:
            problematic_sequences.append({
                'start_idx': i,
                'end_idx': i + sequence_length - 1,
                'start_timestamp': seq_timestamps[0],
                'end_timestamp': seq_timestamps[-1]
            })
    
    # Calculate the percentage of potential sequences that are problematic
    total_possible_sequences = len(timestamps) - sequence_length + 1
    problematic_percentage = (len(problematic_sequences) / total_possible_sequences * 100) if total_possible_sequences > 0 else 0
    
    # Generate a time gap report
    gap_report = {
        'total_samples': len(timestamps),
        'total_gaps': len(gaps_df),
        'mean_gap_seconds': mean_gap,
        'median_gap_seconds': median_gap,
        'std_gap_seconds': std_gap,
        'min_gap_seconds': min_gap,
        'max_gap_seconds': max_gap,
        'large_gap_threshold_seconds': large_gap_threshold,
        'large_gaps_count': len(large_gaps),
        'large_gap_percentage': large_gap_percentage,
        'sequence_length_analyzed': sequence_length,
        'problematic_sequences_count': len(problematic_sequences),
        'problematic_sequences_percentage': problematic_percentage,
        'large_gaps_detail': large_gaps.to_dict('records') if len(large_gaps) > 0 else [],
        'problematic_sequences_detail': problematic_sequences if len(problematic_sequences) > 0 else []
    }
    
    # Save the large gaps to a CSV file
    if len(large_gaps) > 0:
        large_gaps_path = os.path.join(output_dir, 'time_gaps', 'large_gaps.csv')
        large_gaps[['from', 'to', 'gap_seconds', 'gap_minutes', 'gap_hours']].to_csv(large_gaps_path, index=False)
    
    return gap_report

def generate_time_gap_report(gap_report, output_dir):
    """Generate a report specifically about time gaps in the dataset."""
    report_path = os.path.join(output_dir, 'time_gap_report.txt')
    
    with open(report_path, 'w') as f:
        # Header
        f.write("=================================================\n")
        f.write("          TIME GAP ANALYSIS REPORT               \n")
        f.write("=================================================\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overview
        f.write("1. TIME GAP OVERVIEW\n")
        f.write("===================\n\n")
        f.write(f"Total samples in dataset: {gap_report['total_samples']}\n")
        f.write(f"Total gaps analyzed: {gap_report['total_gaps']}\n\n")
        
        f.write("Gap duration statistics:\n")
        f.write(f"  - Mean gap: {gap_report['mean_gap_seconds']:.2f} seconds\n")
        f.write(f"  - Median gap: {gap_report['median_gap_seconds']:.2f} seconds\n")
        f.write(f"  - Standard deviation: {gap_report['std_gap_seconds']:.2f} seconds\n")
        f.write(f"  - Minimum gap: {gap_report['min_gap_seconds']:.2f} seconds\n")
        f.write(f"  - Maximum gap: {gap_report['max_gap_seconds']:.2f} seconds\n\n")
        
        # Large Gap Analysis
        f.write("2. LARGE GAP ANALYSIS\n")
        f.write("=====================\n\n")
        f.write(f"Large gap threshold: {gap_report['large_gap_threshold_seconds']:.2f} seconds\n")
        f.write(f"Number of large gaps: {gap_report['large_gaps_count']} ({gap_report['large_gap_percentage']:.2f}% of all gaps)\n\n")
        
        if gap_report['large_gaps_count'] > 0:
            f.write("Top 10 largest gaps (in seconds):\n")
            sorted_gaps = sorted(gap_report['large_gaps_detail'], key=lambda x: x['gap_seconds'], reverse=True)
            for i, gap in enumerate(sorted_gaps[:10]):
                from_str = datetime.strftime(gap['from_dt'], '%Y-%m-%d %H:%M:%S')
                to_str = datetime.strftime(gap['to_dt'], '%Y-%m-%d %H:%M:%S')
                f.write(f"  {i+1}. {from_str} to {to_str}: {gap['gap_seconds']:.2f} seconds ({gap['gap_minutes']:.2f} minutes)\n")
            f.write("\n")
        else:
            f.write("No large gaps were identified in the dataset.\n\n")
        
        # Impact on Sequence Generation
        f.write("3. IMPACT ON SEQUENCE GENERATION\n")
        f.write("===============================\n\n")
        f.write(f"Sequence length analyzed: {gap_report['sequence_length_analyzed']}\n")
        f.write(f"Number of potential sequences: {gap_report['total_samples'] - gap_report['sequence_length_analyzed'] + 1}\n")
        f.write(f"Number of problematic sequences containing large gaps: {gap_report['problematic_sequences_count']}\n")
        f.write(f"Percentage of problematic sequences: {gap_report['problematic_sequences_percentage']:.2f}%\n\n")
        
        # Risk Assessment
        f.write("4. RISK ASSESSMENT\n")
        f.write("=================\n\n")
        
        # Determine overall risk level
        if gap_report['problematic_sequences_percentage'] < 5:
            risk_level = "LOW"
            risk_desc = "Time gaps are unlikely to significantly impact model training or inference."
        elif gap_report['problematic_sequences_percentage'] < 15:
            risk_level = "MODERATE"
            risk_desc = "Time gaps may have some impact on model training and inference quality."
        else:
            risk_level = "HIGH"
            risk_desc = "Time gaps are likely to significantly impact model training and inference quality."
        
        f.write(f"Overall Risk Level: {risk_level}\n")
        f.write(f"{risk_desc}\n\n")
        
        # Recommendations
        f.write("5. RECOMMENDATIONS\n")
        f.write("=================\n\n")
        
        if risk_level == "LOW":
            f.write("- No specific action needed for time gaps.\n")
        else:
            f.write("Recommended actions based on analysis:\n")
            f.write("- Filter out sequences that span large time gaps during training\n")
            f.write("- Consider adding a 'gap indicator' feature to the model to mark significant time breaks\n")
            
            if risk_level == "HIGH":
                f.write("- Consider collecting more data to fill large gaps in the timeline\n")
                f.write("- Segment the dataset into continuous chunks separated by large gaps\n")
                f.write("- Train separate models for different time periods if there are distribution shifts\n")
            
            f.write("\nSpecific data handling strategies:\n")
            f.write("1. Use the '--sequence_based' flag in prepare_validation_data.py to ensure validation sequences don't span large gaps\n")
            f.write("2. Modify model training code to skip sequences that span large time gaps\n")
            f.write("3. For inference, ensure the model is aware of potential time discontinuities\n")
        
        f.write("\n\n")
        f.write("=================================================\n")
        f.write("                END OF REPORT                   \n")
        f.write("=================================================\n")
    
    return report_path

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input_dir}...")
    df = load_json_files(args.input_dir)
    print(f"Loaded {len(df)} data points.")
    
    if args.analyze_time_gaps:
        print("Performing time gap analysis...")
        gap_report = analyze_time_gaps(df, args.output_dir)
        print("Generating time gap report...")
        gap_report_path = generate_time_gap_report(gap_report, args.output_dir)
        print(f"Time gap report saved to {gap_report_path}")
    
    print("Calculating basic statistics...")
    stats_df, metrics = calculate_basic_statistics(df)
    
    print("Generating distribution plots...")
    generate_distribution_plots(df, metrics, args.output_dir)
    
    print("Analyzing range coverage...")
    range_coverage = generate_range_coverage_analysis(df, metrics, args.output_dir)
    
    print("Calculating correlations...")
    corr_matrix = generate_correlation_analysis(df, metrics, args.output_dir)
    
    print("Generating pairplot...")
    generate_pairplot(df, metrics, args.output_dir)
    
    print("Analyzing multivariate distributions...")
    generate_multivariate_distribution(df, args.output_dir)
    
    print("Checking for gaps in data coverage...")
    gap_report = check_for_gaps(range_coverage, args.output_dir)
    
    print("Generating final report...")
    report_path = generate_report(df, stats_df, corr_matrix, range_coverage, gap_report, args.output_dir)
    
    print(f"Analysis complete! Report saved to {report_path}")
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()