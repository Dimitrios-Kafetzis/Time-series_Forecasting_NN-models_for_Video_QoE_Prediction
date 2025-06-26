import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class NetworkDataAnalyzer:
    """
    Analyzes network data JSON files to extract patterns, distributions, and temporal effects
    """
    def __init__(self, data_directory):
        """
        Initialize the analyzer with the directory containing JSON files
        
        Args:
            data_directory: Path to directory containing JSON files
        """
        self.data_directory = data_directory
        self.all_measurements = pd.DataFrame()
        self.files_metadata = pd.DataFrame()
        self.time_gaps = pd.DataFrame()
        
    def load_data(self, limit=None, parallel=True):
        """
        Load all JSON files and extract measurements
        
        Args:
            limit: Optional limit on number of files to process for testing
            parallel: Use parallel processing for faster loading
        """
        print(f"Loading data from {self.data_directory}...")
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(self.data_directory, "*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        # Limit files if specified
        if limit:
            json_files = json_files[:limit]
            print(f"Limited to {len(json_files)} files for testing")
        
        # Process files
        if parallel and len(json_files) > 10:
            # Use parallel processing for efficiency with many files
            print("Using parallel processing to load files...")
            measurements, file_data = zip(*Parallel(n_jobs=-1)(
                delayed(self._process_file)(file_path) 
                for file_path in tqdm(json_files)
            ))
            
            # Flatten the list of measurements
            all_measurements = [item for sublist in measurements for item in sublist if item]
            all_file_data = [item for item in file_data if item]
            
        else:
            # Process files sequentially for smaller datasets
            all_measurements = []
            all_file_data = []
            
            for file_path in tqdm(json_files):
                measurements, file_data = self._process_file(file_path)
                all_measurements.extend(measurements)
                if file_data:
                    all_file_data.append(file_data)
        
        # Convert to DataFrame
        self.all_measurements = pd.DataFrame(all_measurements)
        self.files_metadata = pd.DataFrame(all_file_data)
        
        # Convert timestamp strings to datetime objects
        self.all_measurements['timestamp'] = pd.to_datetime(self.all_measurements['timestamp'])
        self.files_metadata['timestamp'] = pd.to_datetime(self.files_metadata['timestamp'])
        
        # Sort by timestamp
        self.all_measurements = self.all_measurements.sort_values('timestamp')
        self.files_metadata = self.files_metadata.sort_values('timestamp')
        
        print(f"Processed {len(self.all_measurements)} individual measurements from {len(self.files_metadata)} files")
        
        # Add temporal features
        self._add_temporal_features()
        
        # Detect gaps
        self._detect_time_gaps()
        
        return self.all_measurements
        
    def _process_file(self, file_path):
        """
        Process a single JSON file and extract measurements
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            tuple: (list of measurements, file metadata)
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            file_measurements = []
            qoe = data.get('QoE')
            main_timestamp = data.get('timestamp')
            
            # Convert main timestamp to datetime for file metadata
            if isinstance(main_timestamp, (int, str)):
                main_timestamp_str = str(main_timestamp)
                dt = self._parse_timestamp(main_timestamp_str)
            else:
                dt = None
            
            # File metadata
            file_data = {
                'filename': os.path.basename(file_path),
                'timestamp': dt,
                'qoe': qoe,
                'measurement_count': 0  # Will be updated
            }
            
            # Process each timestamp in the file (except QoE and main timestamp)
            for timestamp, measurement in data.items():
                if timestamp not in ['QoE', 'timestamp'] and isinstance(measurement, dict):
                    # Parse timestamp
                    dt = self._parse_timestamp(timestamp)
                    
                    # Extract measurements
                    throughput = measurement.get('throughput')
                    packets_lost = measurement.get('packets_lost')
                    packet_loss_rate = measurement.get('packet_loss_rate')
                    jitter = measurement.get('jitter')
                    speed = measurement.get('speed')
                    
                    # Add measurement
                    file_measurements.append({
                        'timestamp': dt,
                        'throughput': throughput,
                        'packets_lost': packets_lost,
                        'packet_loss_rate': packet_loss_rate,
                        'jitter': jitter,
                        'speed': speed,
                        'qoe': qoe,
                        'filename': os.path.basename(file_path)
                    })
                    
                    # Increment count
                    file_data['measurement_count'] += 1
            
            return file_measurements, file_data
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return [], None
            
    def _parse_timestamp(self, timestamp_str):
        """
        Parse timestamps in the format YYYYMMDDHHMMSS
        
        Args:
            timestamp_str: Timestamp string
            
        Returns:
            datetime object
        """
        timestamp_str = str(timestamp_str)
        if len(timestamp_str) >= 14:
            year = int(timestamp_str[0:4])
            month = int(timestamp_str[4:6])
            day = int(timestamp_str[6:8])
            hour = int(timestamp_str[8:10])
            minute = int(timestamp_str[10:12])
            second = int(timestamp_str[12:14])
            
            return datetime(year, month, day, hour, minute, second)
        else:
            print(f"Warning: Invalid timestamp format: {timestamp_str}")
            return None
            
    def _add_temporal_features(self):
        """
        Add temporal features to the measurements
        """
        if self.all_measurements.empty:
            return
            
        # Add time-related features
        self.all_measurements['hour'] = self.all_measurements['timestamp'].dt.hour
        self.all_measurements['day_of_week'] = self.all_measurements['timestamp'].dt.dayofweek
        self.all_measurements['is_weekend'] = self.all_measurements['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        self.all_measurements['day_of_month'] = self.all_measurements['timestamp'].dt.day
        self.all_measurements['month'] = self.all_measurements['timestamp'].dt.month
        
        # Add time of day category
        def time_of_day(hour):
            if 5 <= hour < 9:
                return 'early_morning'
            elif 9 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 14:
                return 'noon'
            elif 14 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 20:
                return 'evening'
            elif 20 <= hour < 23:
                return 'night'
            else:
                return 'late_night'
                
        self.all_measurements['time_of_day'] = self.all_measurements['hour'].apply(time_of_day)
            
    def _detect_time_gaps(self):
        """
        Detect significant gaps in the time series data
        """
        if self.files_metadata.empty:
            return
            
        # Sort by timestamp
        sorted_data = self.files_metadata.sort_values('timestamp')
        
        # Calculate time differences between consecutive files
        sorted_data['next_timestamp'] = sorted_data['timestamp'].shift(-1)
        sorted_data['time_diff'] = sorted_data['next_timestamp'] - sorted_data['timestamp']
        
        # Convert to seconds for easier threshold setting
        sorted_data['time_diff_seconds'] = sorted_data['time_diff'].dt.total_seconds()
        
        # Calculate normal interval (most common difference)
        normal_interval = sorted_data['time_diff_seconds'].value_counts().idxmax()
        print(f"Normal interval between files: {normal_interval} seconds")
        
        # Define significant gap threshold (e.g., 5x normal interval)
        gap_threshold = max(5 * normal_interval, 300)  # At least 5 minutes
        
        # Identify significant gaps
        gaps = sorted_data[sorted_data['time_diff_seconds'] > gap_threshold].copy()
        
        if not gaps.empty:
            # Calculate gap duration in different units
            gaps['gap_minutes'] = gaps['time_diff_seconds'] / 60
            gaps['gap_hours'] = gaps['gap_minutes'] / 60
            gaps['gap_days'] = gaps['gap_hours'] / 24
            
            # Store gaps for reporting
            self.time_gaps = gaps[['timestamp', 'next_timestamp', 'time_diff', 
                                   'gap_minutes', 'gap_hours', 'gap_days']]
            
            print(f"Detected {len(gaps)} significant time gaps")
        else:
            print("No significant time gaps detected")
            
    def analyze_statistics(self):
        """
        Calculate statistics for each network parameter
        
        Returns:
            DataFrame: Summary statistics
        """
        if self.all_measurements.empty:
            print("No data loaded. Please call load_data() first.")
            return None
            
        print("\nCalculating parameter statistics...")
        
        # Parameters to analyze
        parameters = ['throughput', 'packets_lost', 'packet_loss_rate', 'jitter', 'qoe']
        
        # Calculate statistics
        statistics = []
        
        for param in parameters:
            values = self.all_measurements[param].dropna()
            
            stats_data = {
                'parameter': param,
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'q1': np.percentile(values, 25),
                'q3': np.percentile(values, 75),
                'skew': stats.skew(values),
                'kurtosis': stats.kurtosis(values)
            }
            
            # Test for normality
            if len(values) > 8:  # Minimum sample size for normality test
                k2, p_value = stats.normaltest(values)
                stats_data['is_normal'] = p_value > 0.05
                stats_data['normality_p_value'] = p_value
            else:
                stats_data['is_normal'] = None
                stats_data['normality_p_value'] = None
                
            statistics.append(stats_data)
            
        # Convert to DataFrame
        stats_df = pd.DataFrame(statistics)
        
        return stats_df
        
    def analyze_correlations(self):
        """
        Analyze correlations between network parameters
        
        Returns:
            DataFrame: Correlation matrix
        """
        if self.all_measurements.empty:
            print("No data loaded. Please call load_data() first.")
            return None
            
        print("\nAnalyzing parameter correlations...")
        
        # Parameters to analyze
        parameters = ['throughput', 'packets_lost', 'packet_loss_rate', 'jitter', 'qoe']
        
        # Calculate correlation matrix
        correlation_matrix = self.all_measurements[parameters].corr()
        
        return correlation_matrix
        
    def analyze_temporal_patterns(self):
        """
        Analyze how network parameters vary by time of day and day of week
        
        Returns:
            dict: Temporal pattern analysis results
        """
        if self.all_measurements.empty:
            print("No data loaded. Please call load_data() first.")
            return None
            
        print("\nAnalyzing temporal patterns...")
        
        # Parameters to analyze
        parameters = ['throughput', 'packets_lost', 'packet_loss_rate', 'jitter', 'qoe']
        
        # Initialize results dictionary
        temporal_patterns = {
            'hourly': {},
            'daily': {},
            'time_of_day': {}
        }
        
        # Analyze hourly patterns
        for param in parameters:
            hourly_stats = self.all_measurements.groupby('hour')[param].agg(['mean', 'std', 'count'])
            temporal_patterns['hourly'][param] = hourly_stats
            
        # Analyze daily patterns
        for param in parameters:
            daily_stats = self.all_measurements.groupby('day_of_week')[param].agg(['mean', 'std', 'count'])
            temporal_patterns['daily'][param] = daily_stats
            
        # Analyze time of day patterns
        for param in parameters:
            tod_stats = self.all_measurements.groupby('time_of_day')[param].agg(['mean', 'std', 'count'])
            temporal_patterns['time_of_day'][param] = tod_stats
            
        return temporal_patterns
        
    def detect_anomalies(self, threshold=3.0):
        """
        Detect anomalies in the data using Z-scores
        
        Args:
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame: Anomalies in the data
        """
        if self.all_measurements.empty:
            print("No data loaded. Please call load_data() first.")
            return None
            
        print("\nDetecting anomalies...")
        
        # Parameters to check for anomalies
        parameters = ['throughput', 'packets_lost', 'packet_loss_rate', 'jitter', 'qoe']
        
        # Create a copy of the data
        data = self.all_measurements.copy()
        
        # Calculate Z-scores for each parameter
        for param in parameters:
            mean = data[param].mean()
            std = data[param].std()
            
            if std > 0:  # Avoid division by zero
                data[f'{param}_zscore'] = (data[param] - mean) / std
            else:
                data[f'{param}_zscore'] = 0
                
        # Identify anomalies
        anomalies = pd.DataFrame()
        
        for param in parameters:
            zscore_col = f'{param}_zscore'
            
            # Find values outside the threshold
            param_anomalies = data[abs(data[zscore_col]) > threshold].copy()
            
            if not param_anomalies.empty:
                param_anomalies['anomaly_parameter'] = param
                param_anomalies['anomaly_zscore'] = param_anomalies[zscore_col]
                
                # Add to overall anomalies
                if anomalies.empty:
                    anomalies = param_anomalies[['timestamp', 'anomaly_parameter', 
                                                param, 'anomaly_zscore']]
                else:
                    anomalies = pd.concat([
                        anomalies,
                        param_anomalies[['timestamp', 'anomaly_parameter', 
                                        param, 'anomaly_zscore']]
                    ])
                    
        # Sort by timestamp
        if not anomalies.empty:
            anomalies = anomalies.sort_values(['timestamp', 'anomaly_parameter'])
            
        print(f"Detected {len(anomalies)} anomalies with threshold {threshold}")
            
        return anomalies
    
    def analyze_qoe_relationship(self):
        """
        Analyze the relationship between QoE and network parameters
        
        Returns:
            dict: QoE analysis results
        """
        if self.all_measurements.empty:
            print("No data loaded. Please call load_data() first.")
            return None
            
        print("\nAnalyzing QoE relationships...")
        
        # Network parameters
        network_params = ['throughput', 'packets_lost', 'packet_loss_rate', 'jitter']
        
        # Initialize results
        qoe_analysis = {
            'correlations': {},
            'regression': {}
        }
        
        # Calculate correlations with QoE
        for param in network_params:
            correlation = self.all_measurements[[param, 'qoe']].corr().iloc[0, 1]
            qoe_analysis['correlations'][param] = correlation
            
        # Simple linear regression for each parameter
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        for param in network_params:
            X = self.all_measurements[param].values.reshape(-1, 1)
            y = self.all_measurements['qoe'].values
            
            # Filter out NaN values
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) > 0:
                model = LinearRegression()
                model.fit(X_clean, y_clean)
                
                y_pred = model.predict(X_clean)
                r2 = r2_score(y_clean, y_pred)
                
                qoe_analysis['regression'][param] = {
                    'coefficient': model.coef_[0],
                    'intercept': model.intercept_,
                    'r2': r2
                }
            else:
                qoe_analysis['regression'][param] = {
                    'coefficient': None,
                    'intercept': None,
                    'r2': None
                }
                
        # Multiple regression with all parameters
        from sklearn.impute import SimpleImputer
        
        X_multi = self.all_measurements[network_params]
        y_multi = self.all_measurements['qoe'].values
        
        # Handle missing values for the multiple regression
        imputer = SimpleImputer(strategy='mean')
        X_multi_imputed = imputer.fit_transform(X_multi)
        
        # Filter out rows with NaN in y
        mask = ~np.isnan(y_multi)
        X_multi_clean = X_multi_imputed[mask]
        y_multi_clean = y_multi[mask]
        
        if len(X_multi_clean) > 0:
            multi_model = LinearRegression()
            multi_model.fit(X_multi_clean, y_multi_clean)
            
            y_multi_pred = multi_model.predict(X_multi_clean)
            multi_r2 = r2_score(y_multi_clean, y_multi_pred)
            
            qoe_analysis['regression']['multiple'] = {
                'coefficients': dict(zip(network_params, multi_model.coef_)),
                'intercept': multi_model.intercept_,
                'r2': multi_r2
            }
        else:
            qoe_analysis['regression']['multiple'] = {
                'coefficients': None,
                'intercept': None,
                'r2': None
            }
            
        return qoe_analysis
        
    def create_report(self, output_dir='network_analysis_report'):
        """
        Generate a comprehensive analysis report
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            str: Path to the generated report
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"\nGenerating analysis report in {output_dir}...")
        
        # Run all analyses
        statistics = self.analyze_statistics()
        correlations = self.analyze_correlations()
        temporal_patterns = self.analyze_temporal_patterns()
        anomalies = self.detect_anomalies()
        qoe_analysis = self.analyze_qoe_relationship()
        
        # Create report
        report_file = os.path.join(output_dir, 'analysis_report.md')
        
        with open(report_file, 'w') as f:
            # Title
            f.write("# Network Data Analysis Report\n\n")
            
            # Dataset overview
            f.write("## Dataset Overview\n\n")
            f.write(f"- Total files: {len(self.files_metadata)}\n")
            f.write(f"- Total measurements: {len(self.all_measurements)}\n")
            
            if not self.all_measurements.empty:
                f.write(f"- Time range: {self.all_measurements['timestamp'].min()} to {self.all_measurements['timestamp'].max()}\n")
                f.write(f"- Time span: {(self.all_measurements['timestamp'].max() - self.all_measurements['timestamp'].min()).days} days\n")
                
            f.write("\n")
            
            # Time gaps
            f.write("## Significant Time Gaps\n\n")
            
            if not self.time_gaps.empty:
                f.write("| Start Time | End Time | Duration (days) | Duration (hours) |\n")
                f.write("|------------|----------|-----------------|------------------|\n")
                
                for _, gap in self.time_gaps.iterrows():
                    f.write(f"| {gap['timestamp']} | {gap['next_timestamp']} | {gap['gap_days']:.1f} | {gap['gap_hours']:.1f} |\n")
            else:
                f.write("No significant time gaps detected.\n")
                
            f.write("\n")
            
            # Parameter statistics
            f.write("## Parameter Statistics\n\n")
            
            if statistics is not None:
                # Format statistics table
                stats_table = statistics[['parameter', 'count', 'mean', 'median', 
                                         'std', 'min', 'max', 'q1', 'q3']]
                
                f.write("| Parameter | Count | Mean | Median | Std Dev | Min | Max | Q1 | Q3 |\n")
                f.write("|-----------|-------|------|--------|---------|-----|-----|----|----|")
                
                for _, row in stats_table.iterrows():
                    f.write(f"\n| {row['parameter']} | {row['count']:.0f} | {row['mean']:.2f} | {row['median']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['max']:.2f} | {row['q1']:.2f} | {row['q3']:.2f} |")
            else:
                f.write("No statistics available.\n")
                
            f.write("\n\n")
            
            # Distribution normality
            f.write("### Distribution Analysis\n\n")
            
            if statistics is not None:
                f.write("| Parameter | Skewness | Kurtosis | Normal Distribution? |\n")
                f.write("|-----------|----------|----------|----------------------|\n")
                
                for _, row in statistics.iterrows():
                    normal_status = "Yes" if row['is_normal'] else "No" if row['is_normal'] is not None else "Insufficient data"
                    f.write(f"| {row['parameter']} | {row['skew']:.2f} | {row['kurtosis']:.2f} | {normal_status} |\n")
            else:
                f.write("No distribution analysis available.\n")
                
            f.write("\n")
            
            # Parameter correlations
            f.write("## Parameter Correlations\n\n")
            
            if correlations is not None:
                f.write("| Parameter | Throughput | Packets Lost | Packet Loss Rate | Jitter | QoE |\n")
                f.write("|-----------|------------|--------------|------------------|--------|-----|\n")
                
                for param, row in correlations.iterrows():
                    f.write(f"| {param} |")
                    for _, val in row.items():
                        f.write(f" {val:.2f} |")
                    f.write("\n")
            else:
                f.write("No correlation analysis available.\n")
                
            f.write("\n")
            
            # QoE relationships
            f.write("## QoE Relationships\n\n")
            
            if qoe_analysis is not None:
                # Correlations with QoE
                f.write("### Correlations with QoE\n\n")
                f.write("| Parameter | Correlation with QoE |\n")
                f.write("|-----------|----------------------|\n")
                
                for param, corr in qoe_analysis['correlations'].items():
                    f.write(f"| {param} | {corr:.2f} |\n")
                    
                f.write("\n")
                
                # Individual regressions
                f.write("### Linear Regression Models\n\n")
                f.write("| Parameter | Coefficient | Intercept | R² |\n")
                f.write("|-----------|-------------|-----------|----|\n")
                
                for param, reg in qoe_analysis['regression'].items():
                    if param != 'multiple':
                        coef = reg['coefficient']
                        intercept = reg['intercept']
                        r2 = reg['r2']
                        
                        coef_str = f"{coef:.4f}" if coef is not None else "N/A"
                        intercept_str = f"{intercept:.4f}" if intercept is not None else "N/A"
                        r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
                        
                        f.write(f"| {param} | {coef_str} | {intercept_str} | {r2_str} |\n")
                        
                f.write("\n")
                
                # Multiple regression
                f.write("### Multiple Regression Model\n\n")
                
                multi_reg = qoe_analysis['regression'].get('multiple', {})
                multi_r2 = multi_reg.get('r2')
                
                if multi_r2 is not None:
                    f.write(f"**R²:** {multi_r2:.4f}\n\n")
                    f.write("**Coefficients:**\n\n")
                    
                    coefs = multi_reg.get('coefficients', {})
                    
                    if coefs:
                        f.write("| Parameter | Coefficient |\n")
                        f.write("|-----------|-------------|\n")
                        
                        for param, coef in coefs.items():
                            f.write(f"| {param} | {coef:.4f} |\n")
                            
                        f.write("\n")
                        
                    f.write(f"**Intercept:** {multi_reg.get('intercept', 'N/A'):.4f}\n\n")
                    
                    f.write("**QoE Prediction Formula:**\n\n")
                    
                    formula = "QoE = "
                    formula += " + ".join([f"{coef:.4f} × {param}" for param, coef in coefs.items()])
                    formula += f" + {multi_reg.get('intercept', 0):.4f}"
                    
                    f.write(f"`{formula}`\n\n")
                else:
                    f.write("Multiple regression model not available.\n\n")
            else:
                f.write("No QoE relationship analysis available.\n\n")
                
            # Temporal patterns
            f.write("## Temporal Patterns\n\n")
            
            if temporal_patterns is not None:
                # Hourly patterns
                f.write("### Hourly Patterns\n\n")
                f.write("Average values by hour of day:\n\n")
                
                for param in ['throughput', 'packet_loss_rate', 'jitter', 'qoe']:
                    if param in temporal_patterns['hourly']:
                        hourly_means = temporal_patterns['hourly'][param]['mean']
                        
                        f.write(f"**{param.capitalize()}:**\n\n")
                        f.write("| Hour | Mean Value | Sample Count |\n")
                        f.write("|------|------------|---------------|\n")
                        
                        for hour in range(24):
                            if hour in hourly_means.index:
                                mean_val = hourly_means[hour]
                                count = temporal_patterns['hourly'][param]['count'][hour]
                                f.write(f"| {hour:02d}:00 | {mean_val:.2f} | {count} |\n")
                                
                        f.write("\n")
                        
                # Daily patterns
                f.write("### Daily Patterns\n\n")
                f.write("Average values by day of week:\n\n")
                
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                for param in ['throughput', 'packet_loss_rate', 'jitter', 'qoe']:
                    if param in temporal_patterns['daily']:
                        daily_means = temporal_patterns['daily'][param]['mean']
                        
                        f.write(f"**{param.capitalize()}:**\n\n")
                        f.write("| Day | Mean Value | Sample Count |\n")
                        f.write("|-----|------------|---------------|\n")
                        
                        for day_idx in range(7):
                            if day_idx in daily_means.index:
                                mean_val = daily_means[day_idx]
                                count = temporal_patterns['daily'][param]['count'][day_idx]
                                f.write(f"| {day_names[day_idx]} | {mean_val:.2f} | {count} |\n")
                                
                        f.write("\n")
                        
                # Time of day patterns
                f.write("### Time of Day Patterns\n\n")
                f.write("Average values by time of day category:\n\n")
                
                for param in ['throughput', 'packet_loss_rate', 'jitter', 'qoe']:
                    if param in temporal_patterns['time_of_day']:
                        tod_means = temporal_patterns['time_of_day'][param]['mean']
                        
                        f.write(f"**{param.capitalize()}:**\n\n")
                        f.write("| Time of Day | Mean Value | Sample Count |\n")
                        f.write("|-------------|------------|---------------|\n")
                        
                        # Order time of day categories
                        tod_order = ['early_morning', 'morning', 'noon', 'afternoon', 
                                    'evening', 'night', 'late_night']
                        
                        for tod in tod_order:
                            if tod in tod_means.index:
                                mean_val = tod_means[tod]
                                count = temporal_patterns['time_of_day'][param]['count'][tod]
                                
                                # Format time of day for display
                                tod_display = ' '.join(w.capitalize() for w in tod.split('_'))
                                f.write(f"| {tod_display} | {mean_val:.2f} | {count} |\n")
                                
                        f.write("\n")
            else:
                f.write("No temporal pattern analysis available.\n\n")
                
            # Anomalies
            f.write("## Anomalies\n\n")
            
            if anomalies is not None and not anomalies.empty:
                f.write(f"Detected {len(anomalies)} anomalies using Z-score threshold of 3.0\n\n")
                f.write("| Timestamp | Parameter | Value | Z-Score |\n")
                f.write("|-----------|-----------|-------|---------|")
                
                # Limit to top 50 anomalies if there are many
                display_anomalies = anomalies.head(50) if len(anomalies) > 50 else anomalies
                
                for _, row in display_anomalies.iterrows():
                    param = row['anomaly_parameter']
                    value = row[param]
                    f.write(f"\n| {row['timestamp']} | {param} | {value:.2f} | {row['anomaly_zscore']:.2f} |")
                    
                if len(anomalies) > 50:
                    f.write(f"\n\n*Showing only the first 50 of {len(anomalies)} anomalies.*")
            else:
                f.write("No anomalies detected or anomaly detection not available.\n")
                
            f.write("\n\n")
            
            # Recommendations
            f.write("## Recommendations for Synthetic Data Generation\n\n")
            
            f.write("Based on the analysis, consider the following when generating synthetic data:\n\n")
            
            # Parameter distributions
            f.write("### 1. Parameter Distributions\n\n")
            
            if statistics is not None:
                for _, row in statistics.iterrows():
                    param = row['parameter']
                    if row['is_normal']:
                        f.write(f"- {param.capitalize()}: Use a normal distribution with mean={row['mean']:.2f} and std={row['std']:.2f}\n")
                    else:
                        f.write(f"- {param.capitalize()}: Not normally distributed. Consider a custom distribution or transformation.\n")
                        
                        # Additional guidance based on skewness
                        if row['skew'] > 1:
                            f.write(f"  - Positively skewed (right-tailed). Consider log-normal or gamma distribution.\n")
                        elif row['skew'] < -1:
                            f.write(f"  - Negatively skewed (left-tailed). Consider beta or reflected distributions.\n")
            else:
                f.write("- No distribution information available.\n")
                
            f.write("\n")
            
            # Parameter correlations
            f.write("### 2. Parameter Correlations\n\n")
            
            if correlations is not None:
                # Find strong correlations
                strong_correlations = []
                
                for i, param1 in enumerate(correlations.index):
                    for j, param2 in enumerate(correlations.columns):
                        if i < j:  # Only consider upper triangle to avoid duplicates
                            corr = correlations.iloc[i, j]
                            if abs(corr) > 0.3:  # Arbitrary threshold for "strong" correlation
                                strong_correlations.append((param1, param2, corr))
                                
                if strong_correlations:
                    f.write("Maintain these significant correlations:\n\n")
                    
                    for param1, param2, corr in strong_correlations:
                        direction = "positive" if corr > 0 else "negative"
                        f.write(f"- {param1.capitalize()} and {param2.capitalize()}: {direction} correlation ({corr:.2f})\n")
                else:
                    f.write("No strong correlations detected that need to be maintained.\n")
            else:
                f.write("- No correlation information available.\n")
                
            f.write("\n")
            
            # QoE relationships
            f.write("### 3. QoE Calculation\n\n")
            
            if qoe_analysis is not None and 'multiple' in qoe_analysis['regression']:
                multi_reg = qoe_analysis['regression']['multiple']
                
                if multi_reg['r2'] is not None and multi_reg['r2'] > 0.6:
                    # If multiple regression model is good
                    f.write("Use the multiple regression model to calculate QoE values:\n\n")
                    
                    formula = "QoE = "
                    formula += " + ".join([f"{coef:.4f} × {param}" for param, coef in multi_reg['coefficients'].items()])
                    formula += f" + {multi_reg['intercept']:.4f}"
                    
                    f.write(f"```\n{formula}\n```\n\n")
                    f.write(f"(This model explains {multi_reg['r2']*100:.1f}% of QoE variance)\n")
                else:
                    # If multiple regression model is not good enough
                    f.write("Consider a more complex model for QoE calculation as the linear regression model is insufficient.\n")
                    f.write("Possible approaches:\n\n")
                    f.write("1. Use a non-linear model (polynomial regression, random forest, etc.)\n")
                    f.write("2. Include interaction terms between parameters\n")
                    f.write("3. Transform the parameters (log, square root, etc.) before regression\n")
            else:
                f.write("- No QoE relationship information available.\n")
                
            f.write("\n")
            
            # Temporal patterns
            f.write("### 4. Temporal Patterns\n\n")
            
            if temporal_patterns is not None:
                f.write("Incorporate these temporal patterns:\n\n")
                
                # Check if there are significant hourly patterns
                significant_hourly = False
                
                for param in ['throughput', 'packet_loss_rate', 'jitter', 'qoe']:
                    if param in temporal_patterns['hourly']:
                        hourly_means = temporal_patterns['hourly'][param]['mean']
                        
                        # Calculate coefficient of variation
                        if len(hourly_means) > 0:
                            cv = hourly_means.std() / hourly_means.mean() if hourly_means.mean() > 0 else 0
                            
                            if cv > 0.1:  # Arbitrary threshold for significance
                                significant_hourly = True
                                break
                                
                if significant_hourly:
                    f.write("- **Hour of Day**: Apply hourly pattern modifiers\n")
                else:
                    f.write("- **Hour of Day**: No significant hourly patterns detected\n")
                    
                # Check if there are significant daily patterns
                significant_daily = False
                
                for param in ['throughput', 'packet_loss_rate', 'jitter', 'qoe']:
                    if param in temporal_patterns['daily']:
                        daily_means = temporal_patterns['daily'][param]['mean']
                        
                        # Calculate coefficient of variation
                        if len(daily_means) > 0:
                            cv = daily_means.std() / daily_means.mean() if daily_means.mean() > 0 else 0
                            
                            if cv > 0.1:  # Arbitrary threshold for significance
                                significant_daily = True
                                break
                                
                if significant_daily:
                    f.write("- **Day of Week**: Apply day-of-week pattern modifiers\n")
                else:
                    f.write("- **Day of Week**: No significant day-of-week patterns detected\n")
                    
                # Suggest network condition profiles
                f.write("\nRecommended network condition profiles for different times:\n\n")
                f.write("1. **Workday Morning (7-9 AM)**: Higher network usage, potentially congested\n")
                f.write("2. **Workday Business Hours (9 AM-5 PM)**: Stable network conditions\n")
                f.write("3. **Workday Evening Peak (5-8 PM)**: Highest network usage, potentially congested\n")
                f.write("4. **Night (10 PM-6 AM)**: Low network usage, better conditions\n")
                f.write("5. **Weekend**: Different pattern from weekdays\n")
            else:
                f.write("- No temporal pattern information available.\n")
                
            f.write("\n")
            
            # Anomalies
            f.write("### 5. Anomalies\n\n")
            
            if anomalies is not None and not anomalies.empty:
                anomaly_count = len(anomalies)
                total_count = len(self.all_measurements)
                anomaly_rate = anomaly_count / total_count
                
                f.write(f"The original data contains {anomaly_count} anomalies out of {total_count} measurements ({anomaly_rate:.2%}).\n\n")
                f.write("For realistic synthetic data, occasionally introduce anomalies at a similar rate.\n")
            else:
                f.write("No anomalies detected or anomaly detection not available.\n")
                
            f.write("\n")
            
            # Gaps recommendations
            f.write("### 6. Filling Time Gaps\n\n")
            
            if not self.time_gaps.empty:
                f.write("When filling the identified time gaps:\n\n")
                f.write("1. **Maintain proper timestamp sequencing** with correct 2-second intervals within files\n")
                f.write("2. **Group measurements into 10-second windows** matching the original file structure\n")
                f.write("3. **Preserve parameter correlations** as identified above\n")
                f.write("4. **Apply appropriate temporal patterns** based on time of day and day of week\n")
                f.write("5. **Calculate QoE** using the recommended model or formula\n")
                f.write("6. **Ensure smooth transitions** at the boundaries between original and synthetic data\n")
            else:
                f.write("No significant time gaps detected.\n")
                
        # Generate visualizations
        vis_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
            
        self._generate_visualizations(vis_dir)
        
        print(f"Analysis report generated: {report_file}")
        return report_file
        
    def _generate_visualizations(self, vis_dir):
        """
        Generate visualizations for the report
        
        Args:
            vis_dir: Directory to save visualizations
        """
        if self.all_measurements.empty:
            return
            
        print("\nGenerating visualizations...")
        
        # Set style
        plt.style.use('ggplot')
        sns.set_palette('colorblind')
        
        # Parameter distributions
        plt.figure(figsize=(20, 15))
        
        for i, param in enumerate(['throughput', 'packets_lost', 'packet_loss_rate', 'jitter', 'qoe']):
            plt.subplot(3, 2, i+1)
            sns.histplot(self.all_measurements[param], kde=True)
            plt.title(f'{param.capitalize()} Distribution')
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'parameter_distributions.png'))
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_params = ['throughput', 'packets_lost', 'packet_loss_rate', 'jitter', 'qoe']
        corr_matrix = self.all_measurements[correlation_params].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Parameter Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'correlation_heatmap.png'))
        plt.close()
        
        # Hourly patterns
        plt.figure(figsize=(20, 15))
        
        for i, param in enumerate(['throughput', 'packet_loss_rate', 'jitter', 'qoe']):
            plt.subplot(2, 2, i+1)
            
            hourly_data = self.all_measurements.groupby('hour')[param].mean()
            hourly_std = self.all_measurements.groupby('hour')[param].std()
            
            plt.errorbar(hourly_data.index, hourly_data.values, yerr=hourly_std.values, 
                        fmt='o-', ecolor='gray', alpha=0.7)
            
            plt.title(f'Hourly {param.capitalize()} Pattern')
            plt.xlabel('Hour of Day')
            plt.ylabel(param.capitalize())
            plt.xticks(range(0, 24, 2))
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'hourly_patterns.png'))
        plt.close()
        
        # Daily patterns
        plt.figure(figsize=(20, 15))
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for i, param in enumerate(['throughput', 'packet_loss_rate', 'jitter', 'qoe']):
            plt.subplot(2, 2, i+1)
            
            daily_data = self.all_measurements.groupby('day_of_week')[param].mean()
            daily_std = self.all_measurements.groupby('day_of_week')[param].std()
            
            plt.errorbar(daily_data.index, daily_data.values, yerr=daily_std.values, 
                        fmt='o-', ecolor='gray', alpha=0.7)
            
            plt.title(f'Daily {param.capitalize()} Pattern')
            plt.xlabel('Day of Week')
            plt.ylabel(param.capitalize())
            plt.xticks(range(7), day_names, rotation=45)
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'daily_patterns.png'))
        plt.close()
        
        # QoE relationships
        plt.figure(figsize=(20, 15))
        
        for i, param in enumerate(['throughput', 'packet_loss_rate', 'jitter']):
            plt.subplot(2, 2, i+1)
            
            sns.scatterplot(x=param, y='qoe', data=self.all_measurements, alpha=0.5)
            
            # Add regression line
            try:
                sns.regplot(x=param, y='qoe', data=self.all_measurements, 
                           scatter=False, ci=None, line_kws={'color': 'red'})
            except:
                pass  # Skip regression line if there's an error
                
            plt.title(f'QoE vs {param.capitalize()}')
            plt.xlabel(param.capitalize())
            plt.ylabel('QoE')
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'qoe_relationships.png'))
        plt.close()
        
        # Time series plots
        plt.figure(figsize=(20, 15))
        
        # Get a subset of data for clarity if there are too many points
        if len(self.all_measurements) > 5000:
            plot_data = self.all_measurements.sort_values('timestamp').iloc[::20].copy()  # Every 20th point
        else:
            plot_data = self.all_measurements.sort_values('timestamp').copy()
            
        for i, param in enumerate(['throughput', 'packet_loss_rate', 'jitter', 'qoe']):
            plt.subplot(4, 1, i+1)
            
            plt.plot(plot_data['timestamp'], plot_data[param], 'b-', alpha=0.7)
            
            plt.title(f'{param.capitalize()} Time Series')
            plt.ylabel(param.capitalize())
            
            if i == 3:  # Only show x-label for the bottom plot
                plt.xlabel('Timestamp')
                
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'time_series.png'))
        plt.close()
        
        print(f"Visualizations saved to {vis_dir}")


def main():
    """
    Main function to run the data analysis
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze network data JSON files')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing JSON files')
    parser.add_argument('--output-dir', type=str, default='network_analysis_report', help='Output directory for report')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to process (for testing)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = NetworkDataAnalyzer(args.data_dir)
        
        # Load and analyze data
        analyzer.load_data(limit=args.limit)
        
        # Generate report
        report_file = analyzer.create_report(output_dir=args.output_dir)
        
        print(f"\nAnalysis complete! Report generated at: {report_file}")
        print(f"Visualizations saved in: {os.path.join(args.output_dir, 'visualizations')}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        
if __name__ == "__main__":
    main()
