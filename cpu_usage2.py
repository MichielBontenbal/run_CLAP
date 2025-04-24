import psutil
import time
import matplotlib.pyplot as plt
import sys
from collections import deque
import os # Import os module for path joining

# --- Configuration ---
INTERVAL = 0.2  # seconds between measurements
DURATION = 30   # total seconds to monitor
OUTPUT_FILENAME = "cpu_usage_plot.png" # Name for the saved plot image
# ---------------------

# Check if matplotlib is installed
try:
    import matplotlib
    # Use a non-interactive backend suitable for saving files without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("-----------------------------------------------------")
    print("ERROR: Matplotlib not found.")
    print("Please install it to generate the graph:")
    print("  pip install matplotlib")
    print("Or for system Python 3 on Debian/Raspberry Pi OS:")
    print("  sudo apt update && sudo apt install python3-matplotlib")
    print("-----------------------------------------------------")
    sys.exit(1) # Exit if plotting library is missing

# Use deque for efficient appending
cpu_usage_data = deque()
timestamps = deque()

# --- Monitoring Loop ---
start_time = time.monotonic() # Use monotonic clock for measuring duration
current_time = start_time

# Initialize psutil.cpu_percent() - first call returns 0.0 or None immediately
# and sets up the baseline for comparison on subsequent calls (interval=None).
psutil.cpu_percent(interval=None)
time.sleep(INTERVAL / 2) # Small initial pause helps first reading be more representative

print(f"Monitoring CPU usage for {DURATION} seconds (interval: {INTERVAL}s)... Press Ctrl+C to stop early.")
print("-" * 40)

try:
    while current_time - start_time < DURATION:
        loop_start_time = time.monotonic()

        # Get CPU usage since the last call (non-blocking)
        cpu_percent = psutil.cpu_percent(interval=None)
        elapsed_time = loop_start_time - start_time # Timestamp relative to start

        # Store data
        cpu_usage_data.append(cpu_percent)
        timestamps.append(elapsed_time)

        # Print current reading (using \r to overwrite the line)
        print(f"Time: {elapsed_time:5.1f}s / {DURATION}s | CPU: {cpu_percent:5.1f}%  ", end='\r')

        # Calculate time spent and sleep for the remaining interval duration
        loop_end_time = time.monotonic()
        time_spent_in_loop = loop_end_time - loop_start_time
        sleep_duration = INTERVAL - time_spent_in_loop

        if sleep_duration > 0:
             time.sleep(sleep_duration)

        current_time = time.monotonic() # Update current time for the next loop check

except KeyboardInterrupt:
    print("\nMonitoring stopped by user.")
finally:
    # Ensure the last print line is cleared or finalized
    print("\n" + "-" * 40)
    print("Monitoring finished.")

# Convert deques to lists for plotting
timestamps_list = list(timestamps)
cpu_usage_data_list = list(cpu_usage_data)

# --- Plotting ---
if not timestamps_list: # Check if any data was collected
    print("No data collected for plotting.")
else:
    print(f"Generating plot and saving to {OUTPUT_FILENAME}...")
    try:
        plt.figure(figsize=(12, 6)) # Create a figure for the plot
        plt.plot(timestamps_list, cpu_usage_data_list, marker='.', linestyle='-', markersize=4, label='CPU Usage')

        plt.xlabel("Time (seconds)")
        plt.ylabel("CPU Usage (%)")
        plt.title(f"Raspberry Pi CPU Usage Over ~{int(round(DURATION))} Seconds") # Use integer duration in title
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 105) # Set Y-axis limit slightly above 100% for visibility
        plt.xlim(0, max(timestamps_list) if timestamps_list else DURATION) # Adjust x-limit to actual data

        # Calculate and display average CPU usage
        if cpu_usage_data_list:
             avg_cpu = sum(cpu_usage_data_list) / len(cpu_usage_data_list)
             # Position text in the upper left corner
             plt.text(0.02, 0.95, f'Average CPU: {avg_cpu:.2f}%',
                      transform=plt.gca().transAxes, # Use axes coordinates
                      fontsize=10,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5)) # Add a text box

        plt.legend()
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        # --- Save the plot to a file ---
        plt.savefig(OUTPUT_FILENAME)
        # -------------------------------

        plt.close() # Close the plot figure to free up memory

        # Get absolute path for user clarity
        abs_path = os.path.abspath(OUTPUT_FILENAME)
        print(f"Plot successfully saved to: {abs_path}")

    except Exception as e:
        print(f"\nError generating or saving plot: {e}")
