#!/usr/bin/env python3
"""
Dataset Split Generator

This script splits a complete network dataset into training, validation, and testing sets
using consecutive days while ensuring QoE coverage across all sets.
"""

import os
import json
import shutil
import argparse
import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

class DatasetSplitter:
    """
    Splits a complete dataset into training, validation, and testing sets
    """
    
    def __init__(self, input_dir, output_base_dir):
        """
        Initialize the splitter
        
        Args:
            input_dir: Directory containing the complete dataset
            output_base_dir: Base directory where train/val/test directories will be created
        """
        self.input_dir = input_dir
        self.output_base_dir = output_base_dir
        self.train_dir = os.path.join(output_base_dir, 'train_set')
        self.val_dir = os.path.join(output_base_dir, 'validation_set')
        self.test_dir = os.path.join(output_base_dir, 'test_set')
        
        # Create output directories if they don't exist
        for directory in [self.output_base_dir, self.train_dir, self.val_dir, self.test_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        # File data storage
        self.files_by_day = defaultdict(list)
        self.day_stats = {}
        self.days_list = []
        
    def scan_dataset(self):
        """
        Scan the dataset to group files by day and analyze QoE distributions
        """
        print(f"Scanning dataset in {self.input_dir}...")
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(self.input_dir, "*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        # Process each file
        for file_path in tqdm(json_files):
            try:
                # Extract date from filename
                filename = os.path.basename(file_path)
                if len(filename) >= 8:  # Ensure there are enough characters
                    # Format is YYYYMMDDHHMMSS.json
                    date_str = filename[0:8]  # Extract YYYYMMDD
                    
                    # Extract QoE from file content
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        qoe = data.get('QoE', 0)
                    
                    # Store file info
                    file_info = {
                        'file_path': file_path,
                        'filename': filename,
                        'qoe': qoe
                    }
                    
                    # Add to files by day
                    self.files_by_day[date_str].append(file_info)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        # Calculate statistics for each day
        for day, files in self.files_by_day.items():
            qoe_values = [file_info['qoe'] for file_info in files]
            
            # Calculate statistics
            stats = {
                'day': day,
                'file_count': len(files),
                'qoe_min': min(qoe_values) if qoe_values else 0,
                'qoe_max': max(qoe_values) if qoe_values else 0,
                'qoe_mean': np.mean(qoe_values) if qoe_values else 0,
                'qoe_median': np.median(qoe_values) if qoe_values else 0,
                'qoe_std': np.std(qoe_values) if qoe_values else 0,
                'qoe_histogram': np.histogram(qoe_values, bins=[0, 20, 40, 60, 80, 90, 100])[0].tolist()
            }
            
            self.day_stats[day] = stats
            
        # Sort days chronologically
        self.days_list = sorted(self.day_stats.keys())
        
        print(f"Dataset spans {len(self.days_list)} days from {self.days_list[0]} to {self.days_list[-1]}")
        print(f"Total files: {sum(len(files) for files in self.files_by_day.values())}")
        
    def split_dataset(self, train_days, val_days, test_days):
        """
        Split the dataset using the specified number of days for each set
        
        Args:
            train_days: Number of days for training set
            val_days: Number of days for validation set
            test_days: Number of days for testing set
            
        Returns:
            dict: Statistics for each set
        """
        if len(self.days_list) == 0:
            print("No data found. Please run scan_dataset() first.")
            return None
            
        if train_days + val_days + test_days > len(self.days_list):
            print(f"Warning: Requested {train_days + val_days + test_days} days but only {len(self.days_list)} available.")
            return None
            
        # Assign days to each set
        train_days_list = self.days_list[:train_days]
        val_days_list = self.days_list[train_days:train_days + val_days]
        test_days_list = self.days_list[train_days + val_days:train_days + val_days + test_days]
        
        # Copy files to respective directories
        set_stats = {}
        
        print(f"\nCopying files to training set ({len(train_days_list)} days)...")
        train_stats = self._copy_files_for_days(train_days_list, self.train_dir, "train")
        set_stats['train'] = train_stats
        
        print(f"\nCopying files to validation set ({len(val_days_list)} days)...")
        val_stats = self._copy_files_for_days(val_days_list, self.val_dir, "validation")
        set_stats['validation'] = val_stats
        
        print(f"\nCopying files to testing set ({len(test_days_list)} days)...")
        test_stats = self._copy_files_for_days(test_days_list, self.test_dir, "test")
        set_stats['test'] = test_stats
        
        # Generate QoE distribution comparison
        self._generate_qoe_comparison(set_stats)
        
        return set_stats
        
    def _copy_files_for_days(self, days_list, output_dir, set_name):
        """
        Copy files for the specified days to the output directory
        
        Args:
            days_list: List of days to copy
            output_dir: Output directory
            set_name: Set name for printing
            
        Returns:
            dict: Statistics for the set
        """
        # Initialize statistics
        total_files = 0
        all_qoe_values = []
        
        # Copy files for each day
        for day in tqdm(days_list):
            files = self.files_by_day.get(day, [])
            total_files += len(files)
            
            for file_info in files:
                # Copy file
                source_path = file_info['file_path']
                dest_path = os.path.join(output_dir, file_info['filename'])
                shutil.copy2(source_path, dest_path)
                
                # Collect QoE for statistics
                all_qoe_values.append(file_info['qoe'])
                
        # Calculate statistics
        if all_qoe_values:
            qoe_histogram, qoe_bins = np.histogram(all_qoe_values, bins=[0, 20, 40, 60, 80, 90, 100])
            
            stats = {
                'set_name': set_name,
                'days': days_list,
                'file_count': total_files,
                'qoe_min': min(all_qoe_values),
                'qoe_max': max(all_qoe_values),
                'qoe_mean': np.mean(all_qoe_values),
                'qoe_median': np.median(all_qoe_values),
                'qoe_std': np.std(all_qoe_values),
                'qoe_histogram': qoe_histogram.tolist(),
                'qoe_bins': qoe_bins.tolist(),
                'qoe_values': all_qoe_values
            }
        else:
            stats = {
                'set_name': set_name,
                'days': days_list,
                'file_count': 0,
                'qoe_min': 0,
                'qoe_max': 0,
                'qoe_mean': 0,
                'qoe_median': 0,
                'qoe_std': 0,
                'qoe_histogram': [0, 0, 0, 0, 0, 0],
                'qoe_bins': [0, 20, 40, 60, 80, 90, 100],
                'qoe_values': []
            }
            
        print(f"{set_name.capitalize()} set: {total_files} files, QoE range: {stats['qoe_min']:.2f}-{stats['qoe_max']:.2f}, Mean: {stats['qoe_mean']:.2f}")
        
        return stats
    
    def _generate_qoe_comparison(self, set_stats):
        """
        Generate visualizations comparing QoE distributions
        
        Args:
            set_stats: Statistics for each set
        """
        # Create output directory for visualizations
        vis_dir = os.path.join(self.output_base_dir, 'visualizations')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
            
        # Generate QoE histogram comparison
        plt.figure(figsize=(12, 8))
        
        # Define QoE category labels
        qoe_categories = ['Very Low (0-20)', 'Low (20-40)', 'Medium-Low (40-60)', 
                          'Medium (60-80)', 'Medium-High (80-90)', 'High (90-100)']
        
        # Plot histograms as bar charts
        bar_width = 0.25
        index = np.arange(len(qoe_categories))
        
        for i, (set_name, stats) in enumerate([('train', set_stats['train']), 
                                              ('validation', set_stats['validation']), 
                                              ('test', set_stats['test'])]):
            # Calculate percentage distribution
            if stats['file_count'] > 0:
                hist_pct = [count / stats['file_count'] * 100 for count in stats['qoe_histogram']]
            else:
                hist_pct = [0] * len(qoe_categories)
                
            plt.bar(index + i * bar_width, hist_pct, bar_width, 
                   label=f"{set_name.capitalize()} Set ({stats['file_count']} files)")
            
        plt.xlabel('QoE Category')
        plt.ylabel('Percentage of Files')
        plt.title('QoE Distribution Comparison')
        plt.xticks(index + bar_width, qoe_categories, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'qoe_distribution_comparison.png'))
        plt.close()
        
        # Generate QoE violin plot comparison
        plt.figure(figsize=(10, 8))
        
        data = [
            set_stats['train']['qoe_values'],
            set_stats['validation']['qoe_values'],
            set_stats['test']['qoe_values']
        ]
        
        plt.violinplot(data, showmeans=True, showmedians=True)
        plt.xticks([1, 2, 3], ['Training Set', 'Validation Set', 'Test Set'])
        plt.ylabel('QoE Value')
        plt.title('QoE Distribution Across Sets')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add mean and median labels
        for i, stats in enumerate([set_stats['train'], set_stats['validation'], set_stats['test']]):
            plt.text(i + 1, stats['qoe_mean'] + 5, f"Mean: {stats['qoe_mean']:.2f}", 
                    ha='center', va='bottom')
            plt.text(i + 1, stats['qoe_median'] - 5, f"Median: {stats['qoe_median']:.2f}", 
                    ha='center', va='top')
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'qoe_violin_comparison.png'))
        plt.close()
        
        # Generate day-by-day QoE box plot
        plt.figure(figsize=(15, 8))
        
        # Group days by set
        train_days = set_stats['train']['days']
        val_days = set_stats['validation']['days']
        test_days = set_stats['test']['days']
        
        # Collect data for each day
        all_days = []
        day_data = []
        day_labels = []
        day_colors = []
        
        for day in self.days_list:
            if day in train_days:
                set_name = 'Training'
                color = 'blue'
            elif day in val_days:
                set_name = 'Validation'
                color = 'green'
            elif day in test_days:
                set_name = 'Testing'
                color = 'red'
            else:
                continue
                
            qoe_values = [file_info['qoe'] for file_info in self.files_by_day[day]]
            if qoe_values:
                all_days.append(day)
                day_data.append(qoe_values)
                # Format day label
                try:
                    day_date = datetime.strptime(day, '%Y%m%d')
                    day_label = day_date.strftime('%m/%d')
                except:
                    day_label = day
                day_labels.append(f"{day_label}\n({set_name})")
                day_colors.append(color)
                
        # Create box plot
        if day_data:
            plt.boxplot(day_data, labels=day_labels, patch_artist=True,
                       boxprops=dict(facecolor='white'))
            
            # Color the background based on the set
            for i, color in enumerate(day_colors):
                plt.axvspan(i + 0.5, i + 1.5, alpha=0.1, color=color)
                
            plt.ylabel('QoE Value')
            plt.title('QoE Distribution by Day and Set')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', alpha=0.1, label='Training'),
                Patch(facecolor='green', alpha=0.1, label='Validation'),
                Patch(facecolor='red', alpha=0.1, label='Testing')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'qoe_by_day.png'))
            plt.close()
        
        # Generate text report
        report_path = os.path.join(self.output_base_dir, 'split_report.md')
        with open(report_path, 'w') as f:
            f.write("# Dataset Split Report\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"- **Total dataset span**: {len(self.days_list)} days from {self.days_list[0]} to {self.days_list[-1]}\n")
            total_files = sum(stats['file_count'] for stats in set_stats.values())
            f.write(f"- **Total files**: {total_files}\n\n")
            
            f.write("## Sets Summary\n\n")
            f.write("| Set | Days | Files | QoE Min | QoE Max | QoE Mean | QoE Median |\n")
            f.write("|-----|------|-------|---------|---------|----------|------------|\n")
            
            for set_name, stats in set_stats.items():
                f.write(f"| {set_name.capitalize()} | {len(stats['days'])} | {stats['file_count']} | "
                       f"{stats['qoe_min']:.2f} | {stats['qoe_max']:.2f} | "
                       f"{stats['qoe_mean']:.2f} | {stats['qoe_median']:.2f} |\n")
                
            f.write("\n## QoE Distribution\n\n")
            f.write("| Set | Very Low<br>(0-20) | Low<br>(20-40) | Medium-Low<br>(40-60) | Medium<br>(60-80) | Medium-High<br>(80-90) | High<br>(90-100) |\n")
            f.write("|-----|-------------------|----------------|----------------------|------------------|------------------------|-----------------|\n")
            
            for set_name, stats in set_stats.items():
                hist = stats['qoe_histogram']
                if stats['file_count'] > 0:
                    pct = [count / stats['file_count'] * 100 for count in hist]
                    f.write(f"| {set_name.capitalize()} | {pct[0]:.1f}% | {pct[1]:.1f}% | {pct[2]:.1f}% | {pct[3]:.1f}% | {pct[4]:.1f}% | {pct[5]:.1f}% |\n")
                else:
                    f.write(f"| {set_name.capitalize()} | 0% | 0% | 0% | 0% | 0% | 0% |\n")
                    
            f.write("\n## Day Allocation\n\n")
            
            f.write("### Training Set\n\n")
            f.write("| Day | Files | QoE Min | QoE Max | QoE Mean |\n")
            f.write("|-----|-------|---------|---------|----------|\n")
            for day in train_days:
                stats = self.day_stats[day]
                f.write(f"| {day} | {stats['file_count']} | {stats['qoe_min']:.2f} | {stats['qoe_max']:.2f} | {stats['qoe_mean']:.2f} |\n")
                
            f.write("\n### Validation Set\n\n")
            f.write("| Day | Files | QoE Min | QoE Max | QoE Mean |\n")
            f.write("|-----|-------|---------|---------|----------|\n")
            for day in val_days:
                stats = self.day_stats[day]
                f.write(f"| {day} | {stats['file_count']} | {stats['qoe_min']:.2f} | {stats['qoe_max']:.2f} | {stats['qoe_mean']:.2f} |\n")
                
            f.write("\n### Testing Set\n\n")
            f.write("| Day | Files | QoE Min | QoE Max | QoE Mean |\n")
            f.write("|-----|-------|---------|---------|----------|\n")
            for day in test_days:
                stats = self.day_stats[day]
                f.write(f"| {day} | {stats['file_count']} | {stats['qoe_min']:.2f} | {stats['qoe_max']:.2f} | {stats['qoe_mean']:.2f} |\n")
                
            f.write("\n## Visualizations\n\n")
            f.write("Visualizations of the QoE distributions are available in the 'visualizations' directory:\n\n")
            f.write("- QoE distribution comparison: [qoe_distribution_comparison.png](visualizations/qoe_distribution_comparison.png)\n")
            f.write("- QoE violin plot comparison: [qoe_violin_comparison.png](visualizations/qoe_violin_comparison.png)\n")
            f.write("- QoE by day: [qoe_by_day.png](visualizations/qoe_by_day.png)\n")
            
        print(f"\nReport generated: {report_path}")
        print(f"Visualizations saved in: {vis_dir}")
        
    def suggest_optimized_split(self, train_pct=70, val_pct=15, test_pct=15):
        """
        Suggest an optimized split that tries to ensure good QoE coverage in all sets
        
        Args:
            train_pct: Percentage of days for training
            val_pct: Percentage of days for validation
            test_pct: Percentage of days for testing
            
        Returns:
            tuple: Suggested number of days for each set
        """
        if len(self.days_list) == 0:
            print("No data found. Please run scan_dataset() first.")
            return (0, 0, 0)
            
        # Calculate initial day counts
        total_days = len(self.days_list)
        train_days = int(total_days * train_pct / 100)
        val_days = int(total_days * val_pct / 100)
        test_days = total_days - train_days - val_days
        
        # Ensure at least 1 day for each set
        train_days = max(1, train_days)
        val_days = max(1, val_days)
        test_days = max(1, test_days)
        
        # Adjust if total exceeds available days
        if train_days + val_days + test_days > total_days:
            excess = (train_days + val_days + test_days) - total_days
            # Reduce proportionally
            train_excess = int(excess * train_pct / 100)
            val_excess = int(excess * val_pct / 100)
            test_excess = excess - train_excess - val_excess
            
            train_days -= min(train_days - 1, train_excess)
            val_days -= min(val_days - 1, val_excess)
            test_days -= min(test_days - 1, test_excess)
            
        # Check the QoE coverage of the suggested split
        train_days_list = self.days_list[:train_days]
        val_days_list = self.days_list[train_days:train_days + val_days]
        test_days_list = self.days_list[train_days + val_days:train_days + val_days + test_days]
        
        # Analyze QoE coverage
        for set_name, days_list in [("Training", train_days_list), 
                                  ("Validation", val_days_list), 
                                  ("Testing", test_days_list)]:
            # Collect QoE values for all files in the set
            qoe_values = []
            for day in days_list:
                for file_info in self.files_by_day.get(day, []):
                    qoe_values.append(file_info['qoe'])
                    
            if qoe_values:
                qoe_min = min(qoe_values)
                qoe_max = max(qoe_values)
                qoe_range = qoe_max - qoe_min
                qoe_histogram, _ = np.histogram(qoe_values, bins=[0, 20, 40, 60, 80, 90, 100])
                
                print(f"{set_name} set ({len(days_list)} days, {len(qoe_values)} files):")
                print(f"  QoE range: {qoe_min:.2f}-{qoe_max:.2f} (span: {qoe_range:.2f})")
                print(f"  QoE distribution: {qoe_histogram}")
                
                # Check if any category has zero files
                empty_categories = [i for i, count in enumerate(qoe_histogram) if count == 0]
                if empty_categories:
                    category_names = ['0-20', '20-40', '40-60', '60-80', '80-90', '90-100']
                    empty_names = [category_names[i] for i in empty_categories]
                    print(f"  Warning: Missing QoE categories: {', '.join(empty_names)}")
                    
            else:
                print(f"{set_name} set ({len(days_list)} days): No files")
                
        return (train_days, val_days, test_days)


def main():
    """
    Main function to run the dataset splitter
    """
    parser = argparse.ArgumentParser(description='Split a network dataset into train/validation/test sets')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing complete dataset')
    parser.add_argument('--output-dir', type=str, default='split_dataset', help='Base directory for output sets')
    parser.add_argument('--train-days', type=int, default=0, help='Number of days for training (0 for auto)')
    parser.add_argument('--val-days', type=int, default=0, help='Number of days for validation (0 for auto)')
    parser.add_argument('--test-days', type=int, default=0, help='Number of days for testing (0 for auto)')
    parser.add_argument('--train-pct', type=float, default=70, help='Percentage of days for training (if auto)')
    parser.add_argument('--val-pct', type=float, default=15, help='Percentage of days for validation (if auto)')
    parser.add_argument('--test-pct', type=float, default=15, help='Percentage of days for testing (if auto)')
    
    args = parser.parse_args()
    
    # Initialize splitter
    splitter = DatasetSplitter(args.input_dir, args.output_dir)
    
    # Scan dataset
    splitter.scan_dataset()
    
    # Determine day split
    if args.train_days > 0 and args.val_days > 0 and args.test_days > 0:
        # Use specified days
        train_days = args.train_days
        val_days = args.val_days
        test_days = args.test_days
        print(f"\nUsing specified split: {train_days} training days, {val_days} validation days, {test_days} testing days")
    else:
        # Use automatic split
        print(f"\nCalculating optimized split with percentages: {args.train_pct}% training, {args.val_pct}% validation, {args.test_pct}% testing")
        train_days, val_days, test_days = splitter.suggest_optimized_split(args.train_pct, args.val_pct, args.test_pct)
        print(f"\nSuggested split: {train_days} training days, {val_days} validation days, {test_days} testing days")
        
        # Ask for confirmation
        confirmation = input("\nProceed with this split? (y/n): ")
        if confirmation.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return
    
    # Split dataset
    splitter.split_dataset(train_days, val_days, test_days)
    
    print("\nDataset split complete!")
    print(f"Training set: {args.output_dir}/train_set")
    print(f"Validation set: {args.output_dir}/validation_set")
    print(f"Test set: {args.output_dir}/test_set")
    print(f"Report: {args.output_dir}/split_report.md")
    print(f"Visualizations: {args.output_dir}/visualizations")


if __name__ == "__main__":
    main()
