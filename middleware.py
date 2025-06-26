from predictor import Predictor
import random
import time
from datetime import datetime

trace_data = [
    {
        "timestamp": 20250510125044,
        "20250510124636": {
            "throughput": 994.2,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 21.442,
            "speed": "0"
        },
        "20250510124638": {
            "throughput": 1150.4,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.4,
            "jitter": 21.398,
            "speed": "0"
        },
        "20250510124640": {
            "throughput": 1351.5,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 16.687,
            "speed": "0"
        },
        "20250510124642": {
            "throughput": 1384.5,
            "packets_lost": 1.0,
            "packet_loss_rate": 0.3,
            "jitter": 20.939,
            "speed": "0"
        },
        "20250510124644": {
            "throughput": 1169.0,
            "packets_lost": 0.0,
            "packet_loss_rate": 0.0,
            "jitter": 20.542,
            "speed": "0"
        }
    }
]

# Initialize the QoE predictor
qoe_predictor = Predictor(
    model_path="gru_basic.h5",  # Path to the model
    scaler_path="scaler.save",  # Path to the scaler
    seq_length=5,               # Sequence length (should match training)
    use_stats=True              # Set this based on how you trained the model
)


def generate_random_subtrace(base_timestamp=None):
    """
    Generate a random network subtrace with realistic values.
    
    Args:
        base_timestamp: Optional timestamp to use. If None, generates based on current time.
        
    Returns:
        tuple: (timestamp_key, subtrace_dict) where timestamp_key is the string timestamp
               and subtrace_dict contains the network metrics
    """
    if base_timestamp is None:
        # Generate timestamp based on current time
        now = datetime.now()
        base_timestamp = int(now.strftime("%Y%m%d%H%M%S"))
    
    # Generate realistic random values for network metrics
    subtrace = {
        "throughput": round(random.uniform(800.0, 1500.0), 1),  # Mbps
        "packets_lost": round(random.uniform(0.0, 3.0), 1),     # Count
        "packet_loss_rate": round(random.uniform(0.0, 1.0), 1), # Percentage
        "jitter": round(random.uniform(10.0, 30.0), 3),         # ms
        "speed": str(random.choice([0, 0, 0, 10, 20, 30]))      # km/h
    }
    
    # Ensure packet_loss_rate is consistent with packets_lost
    if subtrace["packets_lost"] == 0.0:
        subtrace["packet_loss_rate"] = 0.0
    
    timestamp_key = str(base_timestamp)
    return timestamp_key, subtrace


def update_trace_data_rolling_window(trace_data, new_timestamp=None):
    """
    Update trace_data by removing the oldest subtrace and adding a new one.
    This implements a rolling window mechanism.
    
    Args:
        trace_data: List containing trace dictionaries
        new_timestamp: Optional timestamp for the new subtrace. If None, generates automatically.
        
    Returns:
        None (modifies trace_data in place)
    """
    if not trace_data:
        print("Error: trace_data is empty")
        return
    
    # For each trace in the trace_data list
    for trace in trace_data:
        # Get all timestamp keys (excluding the main "timestamp" key)
        subtrace_keys = [k for k in trace.keys() if k != "timestamp"]
        subtrace_keys.sort()  # Sort to ensure we remove the oldest
        
        if len(subtrace_keys) > 0:
            # Remove the oldest subtrace
            oldest_key = subtrace_keys[0]
            del trace[oldest_key]
            print(f"Removed oldest subtrace: {oldest_key}")
            
            # Generate new timestamp based on the newest existing timestamp
            if subtrace_keys:
                newest_timestamp = int(subtrace_keys[-1])
                # Add 2 seconds to the newest timestamp
                new_subtrace_timestamp = newest_timestamp + 2
            else:
                # If no subtraces left, use provided timestamp or current time
                if new_timestamp:
                    new_subtrace_timestamp = new_timestamp
                else:
                    now = datetime.now()
                    new_subtrace_timestamp = int(now.strftime("%Y%m%d%H%M%S"))
            
            # Generate and add new subtrace
            timestamp_key, new_subtrace = generate_random_subtrace(new_subtrace_timestamp)
            trace[timestamp_key] = new_subtrace
            print(f"Added new subtrace: {timestamp_key} -> {new_subtrace}")
            
            # Update the main timestamp to be 2 seconds after the last subtrace
            trace["timestamp"] = new_subtrace_timestamp + 2


def run_continuous_prediction(interval_seconds=2, max_iterations=None):
    """
    Run continuous predictions with rolling window updates.
    
    Args:
        interval_seconds: Time interval between predictions (default: 2 seconds)
        max_iterations: Maximum number of iterations to run. If None, runs indefinitely.
    """
    iteration = 0
    
    print("Starting continuous prediction with rolling window...")
    print(f"Prediction interval: {interval_seconds} seconds")
    print(f"Max iterations: {'Unlimited' if max_iterations is None else max_iterations}")
    print("-" * 50)
    
    try:
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}:")
            
            # Perform prediction
            try:
                result = qoe_predictor.infer(trace_data)
                print(f"QoE Prediction: {result}")
                                
            except Exception as e:
                print(f"Error during prediction: {e}")
            
            # Update the rolling window
            update_trace_data_rolling_window(trace_data)
            
            # Wait for the specified interval
            if max_iterations is None or iteration < max_iterations:
                time.sleep(interval_seconds)
    
    except KeyboardInterrupt:
        print("\n\nStopping continuous prediction...")
        print(f"Completed {iteration} iterations")


# Example usage
if __name__ == "__main__":
    # Test the functions
    print("Initial trace_data:")
    for key in sorted([k for k in trace_data[0].keys() if k != "timestamp"]):
        print(f"  {key}: {trace_data[0][key]}")
    
    # Test single prediction
    print("\nPerforming initial prediction:")
    result = qoe_predictor.infer(trace_data)
    print(f"Initial QoE: {result}")
    
    # Test generating a random subtrace
    print("\nGenerating random subtrace:")
    timestamp, subtrace = generate_random_subtrace()
    print(f"  Timestamp: {timestamp}")
    print(f"  Subtrace: {subtrace}")
    
    # Test updating the rolling window once
    print("\nUpdating rolling window once:")
    update_trace_data_rolling_window(trace_data)
    
    # Uncomment the following line to run continuous predictions
    # run_continuous_prediction(interval_seconds=2, max_iterations=10)