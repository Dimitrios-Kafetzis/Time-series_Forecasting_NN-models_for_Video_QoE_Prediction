import os
import json
import glob
import argparse
import shutil
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

class NetworkDatasetGenerator:
    """
    Generates a complete network dataset by reusing existing JSON files
    and organizing them into a predictable pattern for urban networks.
    """
    
    def __init__(self, input_dir, output_dir):
        """
        Initialize the generator
        
        Args:
            input_dir: Directory containing original JSON files
            output_dir: Directory where the new dataset will be saved
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.files_metadata = []
        self.category_pools = {
            'A': [],  # Excellent: QoE 90-100
            'B': [],  # Good: QoE 80-90
            'C': [],  # Average: QoE 70-80
            'D': [],  # Fair: QoE 60-70
            'E': []   # Poor: QoE < 60
        }
        self.category_indices = {
            'A': 0,  # Current index for category A
            'B': 0,  # Current index for category B
            'C': 0,  # Current index for category C
            'D': 0,  # Current index for category D
            'E': 0   # Current index for category E
        }
        self.weekly_pattern = self._create_weekly_pattern()
        
    def _create_weekly_pattern(self):
        """
        Create a weekly pattern mapping hours to categories
        
        Returns:
            dict: Mapping of day and hour to network quality category
        """
        # Initialize pattern dictionary
        pattern = {}
        
        # Days of week (0 = Monday, 6 = Sunday)
        days = range(7)
        
        # Weekday pattern (Monday-Friday)
        weekday_pattern = {
            0: 'A',  # 12AM-1AM
            1: 'A',  # 1AM-2AM
            2: 'A',  # 2AM-3AM
            3: 'A',  # 3AM-4AM
            4: 'A',  # 4AM-5AM
            5: 'A',  # 5AM-6AM
            6: 'B',  # 6AM-7AM
            7: 'B',  # 7AM-8AM
            8: 'B',  # 8AM-9AM
            9: 'A',  # 9AM-10AM
            10: 'A', # 10AM-11AM
            11: 'A', # 11AM-12PM
            12: 'B', # 12PM-1PM
            13: 'B', # 1PM-2PM
            14: 'A', # 2PM-3PM
            15: 'A', # 3PM-4PM
            16: 'A', # 4PM-5PM
            17: 'C', # 5PM-6PM
            18: 'C', # 6PM-7PM
            19: 'D', # 7PM-8PM
            20: 'D', # 8PM-9PM
            21: 'B', # 9PM-10PM
            22: 'B', # 10PM-11PM
            23: 'B'  # 11PM-12AM
        }
        
        # Saturday pattern
        saturday_pattern = {
            0: 'A',  # 12AM-1AM
            1: 'A',  # 1AM-2AM
            2: 'A',  # 2AM-3AM
            3: 'A',  # 3AM-4AM
            4: 'A',  # 4AM-5AM
            5: 'A',  # 5AM-6AM
            6: 'A',  # 6AM-7AM
            7: 'A',  # 7AM-8AM
            8: 'B',  # 8AM-9AM
            9: 'B',  # 9AM-10AM
            10: 'B', # 10AM-11AM
            11: 'B', # 11AM-12PM
            12: 'B', # 12PM-1PM
            13: 'B', # 1PM-2PM
            14: 'C', # 2PM-3PM
            15: 'C', # 3PM-4PM
            16: 'C', # 4PM-5PM
            17: 'C', # 5PM-6PM
            18: 'C', # 6PM-7PM
            19: 'D', # 7PM-8PM
            20: 'D', # 8PM-9PM
            21: 'D', # 9PM-10PM
            22: 'B', # 10PM-11PM
            23: 'B'  # 11PM-12AM
        }
        
        # Sunday pattern
        sunday_pattern = {
            0: 'A',  # 12AM-1AM
            1: 'A',  # 1AM-2AM
            2: 'A',  # 2AM-3AM
            3: 'A',  # 3AM-4AM
            4: 'A',  # 4AM-5AM
            5: 'A',  # 5AM-6AM
            6: 'A',  # 6AM-7AM
            7: 'A',  # 7AM-8AM
            8: 'B',  # 8AM-9AM
            9: 'B',  # 9AM-10AM
            10: 'B', # 10AM-11AM
            11: 'B', # 11AM-12PM
            12: 'B', # 12PM-1PM
            13: 'B', # 1PM-2PM
            14: 'B', # 2PM-3PM
            15: 'B', # 3PM-4PM
            16: 'B', # 4PM-5PM
            17: 'C', # 5PM-6PM
            18: 'C', # 6PM-7PM
            19: 'D', # 7PM-8PM
            20: 'D', # 8PM-9PM
            21: 'B', # 9PM-10PM
            22: 'B', # 10PM-11PM
            23: 'B'  # 11PM-12AM
        }
        
        # Assign patterns to days
        for day in days:
            if day < 5:  # Monday-Friday
                pattern[day] = weekday_pattern
            elif day == 5:  # Saturday
                pattern[day] = saturday_pattern
            else:  # Sunday
                pattern[day] = sunday_pattern
                
        return pattern
    
    def load_files(self):
        """
        Load all JSON files and categorize them
        """
        print(f"Loading files from {self.input_dir}...")
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(self.input_dir, "*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        # Process each file
        for file_path in tqdm(json_files):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract QoE
                qoe = data.get('QoE', 0)
                
                # Extract timestamps and parameters
                timestamps = []
                throughputs = []
                packet_loss_rates = []
                jitters = []
                
                for key, value in data.items():
                    if key not in ['QoE', 'timestamp'] and isinstance(value, dict):
                        throughputs.append(value.get('throughput', 0))
                        packet_loss_rates.append(value.get('packet_loss_rate', 0))
                        jitters.append(value.get('jitter', 0))
                        timestamps.append(key)
                
                # Calculate averages
                avg_throughput = np.mean(throughputs) if throughputs else 0
                avg_packet_loss = np.mean(packet_loss_rates) if packet_loss_rates else 0
                avg_jitter = np.mean(jitters) if jitters else 0
                
                # Create file metadata
                file_metadata = {
                    'file_path': file_path,
                    'qoe': qoe,
                    'avg_throughput': avg_throughput,
                    'avg_packet_loss': avg_packet_loss,
                    'avg_jitter': avg_jitter,
                    'timestamps': timestamps,
                    'first_timestamp': min(timestamps) if timestamps else '',
                    'last_timestamp': max(timestamps) if timestamps else '',
                    'filename': os.path.basename(file_path)
                }
                
                # Determine category
                category = self._categorize_file(qoe, avg_throughput, avg_packet_loss, avg_jitter)
                file_metadata['category'] = category
                
                # Add to appropriate category pool
                self.category_pools[category].append(file_metadata)
                
                # Add to overall metadata
                self.files_metadata.append(file_metadata)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        # Sort each category by QoE for better transitions
        for category in self.category_pools:
            self.category_pools[category].sort(key=lambda x: x['qoe'])
            print(f"Category {category}: {len(self.category_pools[category])} files")
        
        # Verify we have files in each category
        for category, files in self.category_pools.items():
            if not files:
                print(f"Warning: No files found for category {category}")
                # If a category is empty, try to fill it from adjacent categories
                self._fill_empty_category(category)
    
    def _categorize_file(self, qoe, throughput, packet_loss, jitter):
        """
        Categorize a file based on its network parameters
        
        Args:
            qoe: Quality of Experience value
            throughput: Average throughput
            packet_loss: Average packet loss rate
            jitter: Average jitter
            
        Returns:
            str: Category (A, B, C, D, or E)
        """
        # Primarily categorize by QoE
        if qoe >= 90:
            return 'A'  # Excellent
        elif qoe >= 80:
            return 'B'  # Good
        elif qoe >= 70:
            return 'C'  # Average
        elif qoe >= 60:
            return 'D'  # Fair
        else:
            return 'E'  # Poor
    
    def _fill_empty_category(self, empty_category):
        """
        Fill an empty category with files from adjacent categories
        
        Args:
            empty_category: The category that needs files
        """
        if empty_category == 'A' and self.category_pools['B']:
            # Take top QoE files from B
            print(f"Filling empty category A with top files from category B")
            files_to_move = sorted(self.category_pools['B'], key=lambda x: x['qoe'], reverse=True)[:5]
            for file in files_to_move:
                file['category'] = 'A'
                self.category_pools['A'].append(file)
                
        elif empty_category == 'E' and self.category_pools['D']:
            # Take lowest QoE files from D
            print(f"Filling empty category E with bottom files from category D")
            files_to_move = sorted(self.category_pools['D'], key=lambda x: x['qoe'])[:5]
            for file in files_to_move:
                file['category'] = 'E'
                self.category_pools['E'].append(file)
                
        elif empty_category in ['B', 'C', 'D']:
            # Try to fill from adjacent categories
            adjacent_categories = []
            if empty_category == 'B':
                adjacent_categories = ['A', 'C']
            elif empty_category == 'C':
                adjacent_categories = ['B', 'D']
            elif empty_category == 'D':
                adjacent_categories = ['C', 'E']
                
            for adj_cat in adjacent_categories:
                if self.category_pools[adj_cat]:
                    print(f"Filling empty category {empty_category} with files from category {adj_cat}")
                    files_to_move = self.category_pools[adj_cat][:5]  # Take 5 files
                    for file in files_to_move:
                        file['category'] = empty_category
                        self.category_pools[empty_category].append(file)
                    break
    
    def get_file_for_category(self, category):
        """
        Get the next file for a given category, cycling through available files
        
        Args:
            category: The quality category (A, B, C, D, or E)
            
        Returns:
            dict: File metadata
        """
        files = self.category_pools.get(category, [])
        
        if not files:
            # If category is empty, try to use an adjacent category
            if category == 'A':
                return self.get_file_for_category('B')
            elif category == 'E':
                return self.get_file_for_category('D')
            elif category == 'B':
                return self.get_file_for_category('C')
            elif category == 'D':
                return self.get_file_for_category('C')
            else:  # category == 'C'
                return self.get_file_for_category('B')
        
        # Get current index and increment it
        idx = self.category_indices[category]
        self.category_indices[category] = (idx + 1) % len(files)
        
        return files[idx]
    
    def generate_dataset(self, num_weeks, start_date=None):
        """
        Generate the complete dataset
        
        Args:
            num_weeks: Number of weeks to generate
            start_date: Optional start date (default is today)
            
        Returns:
            int: Number of files generated
        """
        if start_date is None:
            # Use current date and set time to midnight
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        print(f"Generating {num_weeks} weeks of data starting from {start_date}...")
        
        # Calculate total time span
        end_date = start_date + timedelta(weeks=num_weeks)
        
        # Create timestamp sequence (10-second intervals)
        current_time = start_date
        time_points = []
        
        while current_time < end_date:
            time_points.append(current_time)
            current_time += timedelta(seconds=10)
            
        print(f"Total time points: {len(time_points)}")
        
        # Generate files for each time point
        files_generated = 0
        
        for time_point in tqdm(time_points):
            # Get day of week (0 = Monday, 6 = Sunday)
            day_of_week = time_point.weekday()
            
            # Get hour of day
            hour = time_point.hour
            
            # Determine category based on day and hour
            category = self.weekly_pattern[day_of_week][hour]
            
            # Get file for this category
            file_metadata = self.get_file_for_category(category)
            
            # Generate new file with updated timestamps
            self._generate_file(file_metadata, time_point)
            
            files_generated += 1
            
        print(f"Generated {files_generated} files in {self.output_dir}")
        return files_generated
    
    def _generate_file(self, file_metadata, time_point):
        """
        Generate a new file by updating timestamps in an existing file
        
        Args:
            file_metadata: Metadata of the file to use as template
            time_point: End time for the new file
        """
        try:
            # Read original file
            with open(file_metadata['file_path'], 'r') as f:
                data = json.load(f)
                
            # Create a copy of the data
            new_data = data.copy()
            
            # Calculate new timestamps
            # The time_point represents the end time of the 10-second window
            end_time = time_point
            start_time = end_time - timedelta(seconds=8)  # 8 seconds earlier for 5 measurements at 2-second intervals
            
            # Update the main timestamp
            new_data['timestamp'] = int(end_time.strftime("%Y%m%d%H%M%S"))
            
            # Get the original timestamps and measurements
            orig_timestamps = []
            measurements = {}
            
            for key, value in data.items():
                if key not in ['QoE', 'timestamp'] and isinstance(value, dict):
                    orig_timestamps.append(key)
                    measurements[key] = value
                    
            # Sort timestamps
            orig_timestamps.sort()
            
            # Create new timestamps at 2-second intervals
            current_time = start_time
            new_timestamps = []
            
            for _ in range(5):  # 5 measurements at 2-second intervals
                new_timestamps.append(current_time.strftime("%Y%m%d%H%M%S"))
                current_time += timedelta(seconds=2)
                
            # Update measurements with new timestamps
            for i, orig_ts in enumerate(orig_timestamps):
                if i < len(new_timestamps):
                    new_ts = new_timestamps[i]
                    # Copy measurement to new timestamp and remove old one
                    new_data[new_ts] = measurements[orig_ts]
                    if orig_ts in new_data and orig_ts not in ['QoE', 'timestamp']:
                        del new_data[orig_ts]
                        
            # Create filename based on end timestamp
            filename = f"{end_time.strftime('%Y%m%d%H%M%S')}.json"
            output_path = os.path.join(self.output_dir, filename)
            
            # Write the new file
            with open(output_path, 'w') as f:
                json.dump(new_data, f, indent=4)
                
        except Exception as e:
            print(f"Error generating file for {time_point}: {e}")
    
    def summarize_dataset(self):
        """
        Summarize the generated dataset
        """
        if not os.path.exists(self.output_dir):
            print("Output directory does not exist. Generate dataset first.")
            return
            
        # Count files
        json_files = glob.glob(os.path.join(self.output_dir, "*.json"))
        file_count = len(json_files)
        
        # Sample a few files for verification
        sample_size = min(5, file_count)
        sample_files = np.random.choice(json_files, sample_size, replace=False)
        
        print(f"\nDataset Summary:")
        print(f"Total files: {file_count}")
        
        if sample_size > 0:
            print("\nSample files:")
            
            for file_path in sample_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    print(f"  - {os.path.basename(file_path)}: QoE = {data.get('QoE')}")
                    
                    # Print one timestamp
                    for key in data:
                        if key not in ['QoE', 'timestamp'] and isinstance(data[key], dict):
                            print(f"    Measurement at {key}: Throughput = {data[key].get('throughput')}, "
                                  f"Packet Loss = {data[key].get('packet_loss_rate')}%, "
                                  f"Jitter = {data[key].get('jitter')}")
                            break
                            
                except Exception as e:
                    print(f"  - Error reading {file_path}: {e}")
        
        # Print category distribution
        print("\nFiles per category in original dataset:")
        for category, files in self.category_pools.items():
            print(f"  - Category {category}: {len(files)} files")


def main():
    """
    Main function to run the dataset generator
    """
    parser = argparse.ArgumentParser(description='Generate a complete network dataset')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing original JSON files')
    parser.add_argument('--output-dir', type=str, default='generated_dataset', help='Directory to save generated files')
    parser.add_argument('--weeks', type=int, default=2, help='Number of weeks to generate')
    parser.add_argument('--start-date', type=str, default=None, 
                      help='Start date (YYYY-MM-DD). Defaults to today.')
    
    args = parser.parse_args()
    
    # Parse start date if provided
    start_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date format: {args.start_date}. Using today's date.")
    
    # Initialize generator
    generator = NetworkDatasetGenerator(args.input_dir, args.output_dir)
    
    # Load and categorize files
    generator.load_files()
    
    # Generate dataset
    generator.generate_dataset(args.weeks, start_date)
    
    # Summarize dataset
    generator.summarize_dataset()


if __name__ == "__main__":
    main()
