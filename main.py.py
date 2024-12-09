import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import zscore
import logging
from statsmodels.tsa.seasonal import STL
from river.drift import ADWIN
import time

# -------------------- SETTING UP LOGGING -------------------- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Configure logging to display timestamps, log level, and messages in the console

# -------------------- ALGORITHM SELECTION -------------------- #
class IncrementalIsolationForest:
    """
    IncrementalIsolationForest class that fits the model incrementally on data chunks.
    It uses the Isolation Forest algorithm to detect anomalies in data streams.
    """
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)  # Initialize Isolation Forest with contamination level
        self.chunk_data = []  # Store data chunks for incremental training

    def partial_fit(self, X):
        """
        Incrementally fit the Isolation Forest model on incoming data.
        - X: Input data chunk.
        """
        try:
            self.chunk_data.extend(X)  # Add new data to chunk data
            if len(self.chunk_data) > 200:
                self.chunk_data = self.chunk_data[-200:]  # Limit stored data to the most recent 200 points
            self.model.fit(np.array(self.chunk_data).reshape(-1, 1))  # Fit model to the current chunk
        except Exception as e:
            logging.error(f"Error during model fitting: {e}")  # Log any fitting errors

    def predict(self, X):
        """
        Predict anomalies in the input data.
        - X: Input data chunk.
        Returns: Anomaly labels (-1 for anomaly, 1 for normal).
        """
        try:
            return self.model.predict(X)
        except Exception as e:
            logging.error(f"Error during anomaly prediction: {e}")  # Log any prediction errors
            return np.ones(len(X))  # Return a default prediction if error occurs

# -------------------- DATA STREAM SIMULATION FOR ENERGY MARKET -------------------- #
def generate_energy_market_data(chunk_size=100):
    """
    Simulates real-time electricity price and energy demand data.
    Generates random data with occasional anomalies to mimic real-world scenarios.
    - Prices range from $30 to $300 per MWh with occasional spikes.
    - Demand ranges from 4,000 MW to 9,000 MW with smooth fluctuations.
    - chunk_size: Number of data points generated per time step.
    """
    t = 0  # Initialize time variable
    while True:
        try:
            time_step = np.arange(t, t + chunk_size)  # Create time steps for the chunk
            t += chunk_size  # Update the time index

            # Generate electricity price data
            base_price = 100 + 50 * np.sin(time_step * 0.05)  # Simulate a periodic pattern in prices
            price_noise = np.random.normal(0, 10, chunk_size)  # Add noise to prices
            price_anomalies = np.random.choice([0, 200], size=chunk_size, p=[0.98, 0.02])  # Introduce occasional price spikes

            electricity_price = base_price + price_noise + price_anomalies  # Combine base price, noise, and anomalies

            # Generate energy demand data
            base_demand = 6500 + 1500 * np.sin(time_step * 0.03)  # Simulate a periodic pattern in demand
            demand_noise = np.random.normal(0, 100, chunk_size)  # Add noise to demand
            demand_anomalies = np.random.choice([0, -500], size=chunk_size, p=[0.98, 0.02])  # Introduce occasional demand drops

            energy_demand = base_demand + demand_noise + demand_anomalies  # Combine base demand, noise, and anomalies

            yield electricity_price, energy_demand  # Yield simulated data chunks
        except Exception as e:
            logging.error(f"Error during data stream generation: {e}")  # Log errors in data generation
            break  # Break if error occurs

# -------------------- REAL-TIME ANOMALY DETECTION WITH DYNAMIC AXIS UPDATE -------------------- #
def detect_anomalies_in_energy_market(window_size=400, chunk_size=100, max_duration=30):
    """
    Performs real-time anomaly detection on energy market data streams.
    - window_size: Number of data points to consider for each plot window.
    - chunk_size: Number of data points processed at each iteration.
    - max_duration: Maximum duration to run the detection in seconds.
    """
    rolling_price_data = []  # Store recent electricity price data for real-time analysis
    rolling_demand_data = []  # Store recent energy demand data
    anomaly_price_indices = []  # Indices where price anomalies are detected
    anomaly_demand_indices = []  # Indices where demand anomalies are detected
    all_data = []  # Track all data along with anomaly labels

    # Initialize two incremental Isolation Forest models, one for price and one for demand
    isolation_forest_price = IncrementalIsolationForest(contamination=0.05)
    isolation_forest_demand = IncrementalIsolationForest(contamination=0.05)

    # Initialize drift detectors for detecting concept drift in price and demand streams
    drift_detector_price = ADWIN()
    drift_detector_demand = ADWIN()

    # Set up Matplotlib figure and subplots for real-time plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Create two vertically stacked subplots
    plt.subplots_adjust(hspace=0.4)  # Add space between the subplots for clarity

    # Electricity price plot setup
    line_price, = ax1.plot([], [], lw=2, label="Electricity Price ($/MWh)")
    scatter_price = ax1.scatter([], [], color='red', label="Anomalies", zorder=2)
    ax1.set_title("Real-Time Electricity Price with Anomalies")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Price ($/MWh)")
    ax1.legend()

    # Energy demand plot setup
    line_demand, = ax2.plot([], [], lw=2, label="Energy Demand (MW)")
    scatter_demand = ax2.scatter([], [], color='red', label="Anomalies", zorder=2)
    ax2.set_title("Real-Time Energy Demand with Anomalies")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Demand (MW)")
    ax2.legend()

    # Initialize the plot lines and scatter points
    def init():
        line_price.set_data([], [])  # Clear the price line data
        scatter_price.set_offsets(np.empty((0, 2)))  # Clear the price anomaly scatter points
        line_demand.set_data([], [])  # Clear the demand line data
        scatter_demand.set_offsets(np.empty((0, 2)))  # Clear the demand anomaly scatter points
        return line_price, scatter_price, line_demand, scatter_demand  # Return initialized elements

    start_time = time.time()  # Record the start time for the duration limit

    def update(frame):
        """
        Update function to refresh the data stream and anomaly plots in real-time.
        """
        nonlocal rolling_price_data, rolling_demand_data, anomaly_price_indices, anomaly_demand_indices, all_data
        nonlocal isolation_forest_price, isolation_forest_demand

        elapsed_time = time.time() - start_time  # Calculate elapsed time
        if elapsed_time > max_duration:
            return line_price, scatter_price, line_demand, scatter_demand  # Stop updating after reaching max duration

        # Get the next chunk of data (electricity price and demand) from the stream generator
        price_data, demand_data = frame
        rolling_price_data.extend(price_data)  # Append new price data to the rolling window
        rolling_demand_data.extend(demand_data)  # Append new demand data to the rolling window

        # Limit data to the last 'window_size' points for display
        if len(rolling_price_data) > window_size:
            rolling_price_data = rolling_price_data[-window_size:]
        if len(rolling_demand_data) > window_size:
            rolling_demand_data = rolling_demand_data[-window_size:]

        # Perform seasonal decomposition using STL (Season-Trend Decomposition)
        try:
            if len(rolling_price_data) >= window_size:
                stl_price = STL(rolling_price_data, period=50)
                res_price = stl_price.fit()
                residual_price = res_price.resid  # Get the residual (detrended) data for price
            else:
                residual_price = rolling_price_data
        except Exception as e:
            logging.error(f"Error during price seasonal decomposition: {e}")
            residual_price = rolling_price_data

        try:
            if len(rolling_demand_data) >= window_size:
                stl_demand = STL(rolling_demand_data, period=50)
                res_demand = stl_demand.fit()
                residual_demand = res_demand.resid  # Get the residual (detrended) data for demand
            else:
                residual_demand = rolling_demand_data
        except Exception as e:
            logging.error(f"Error during demand seasonal decomposition: {e}")
            residual_demand = rolling_demand_data

        # Z-Score anomaly detection: find outliers based on standard deviation
        z_scores_price = zscore(residual_price)
        z_anomalies_price = [i for i, z in enumerate(z_scores_price) if abs(z) > 2.5]  # Mark anomalies in price

        z_scores_demand = zscore(residual_demand)
        z_anomalies_demand = [i for i, z in enumerate(z_scores_demand) if abs(z) > 2.5]  # Mark anomalies in demand

        # Incrementally fit Isolation Forest on price and demand, then predict anomalies
        isolation_forest_price.partial_fit(rolling_price_data)
        preds_price = isolation_forest_price.predict(np.array(rolling_price_data).reshape(-1, 1))
        if_anomalies_price = [i for i, pred in enumerate(preds_price[-chunk_size:]) if pred == -1]  # Isolation Forest anomalies in price

        isolation_forest_demand.partial_fit(rolling_demand_data)
        preds_demand = isolation_forest_demand.predict(np.array(rolling_demand_data).reshape(-1, 1))
        if_anomalies_demand = [i for i, pred in enumerate(preds_demand[-chunk_size:]) if pred == -1]  # Isolation Forest anomalies in demand

        # Combine anomalies from Z-score and Isolation Forest
        combined_anomalies_price = set(z_anomalies_price + if_anomalies_price)
        combined_anomalies_demand = set(z_anomalies_demand + if_anomalies_demand)

        anomaly_price_indices = list(combined_anomalies_price)  # Update anomaly indices for price
        anomaly_demand_indices = list(combined_anomalies_demand)  # Update anomaly indices for demand

        # Track data along with anomaly labels for saving later
        for i, price_value in enumerate(rolling_price_data[-chunk_size:]):
            price_anomaly_label = 1 if i in combined_anomalies_price else 0
            all_data.append([price_value, price_anomaly_label])

        for i, demand_value in enumerate(rolling_demand_data[-chunk_size:]):
            demand_anomaly_label = 1 if i in combined_anomalies_demand else 0
            all_data.append([demand_value, demand_anomaly_label])

        # Update the line plot for electricity price
        line_price.set_data(range(len(rolling_price_data)), rolling_price_data)

        # Update the line plot for energy demand
        line_demand.set_data(range(len(rolling_demand_data)), rolling_demand_data)

        # Update the scatter plot for price anomalies
        scatter_price.set_offsets(np.c_[anomaly_price_indices, [rolling_price_data[i] for i in anomaly_price_indices]])

        # Update the scatter plot for demand anomalies
        scatter_demand.set_offsets(np.c_[anomaly_demand_indices, [rolling_demand_data[i] for i in anomaly_demand_indices]])

        # Dynamically adjust the x-axis based on the current number of data points
        current_time_index = len(rolling_price_data)
        ax1.set_xlim(current_time_index - window_size, current_time_index)  # Update x-axis for price
        ax2.set_xlim(current_time_index - window_size, current_time_index)  # Update x-axis for demand

        # Adjust the y-axis to fit the latest price and demand values
        ax1.set_ylim(min(rolling_price_data) - 50, max(rolling_price_data) + 50)
        ax2.set_ylim(min(rolling_demand_data) - 500, max(rolling_demand_data) + 500)

        # Check for concept drift (sudden changes in data patterns) in price and demand
        drift_detector_price.update(np.mean(rolling_price_data[-chunk_size:]))  # Check for drift in price
        if drift_detector_price.drift_detected:
            logging.info("Concept drift detected in price, retraining the model...")
            isolation_forest_price = IncrementalIsolationForest(contamination=0.05)  # Reset Isolation Forest if drift detected

        drift_detector_demand.update(np.mean(rolling_demand_data[-chunk_size:]))  # Check for drift in demand
        if drift_detector_demand.drift_detected:
            logging.info("Concept drift detected in demand, retraining the model...")
            isolation_forest_demand = IncrementalIsolationForest(contamination=0.05)  # Reset Isolation Forest if drift detected

        return line_price, scatter_price, line_demand, scatter_demand  # Return updated plot elements

    # Create a data generator that produces chunks of energy market data
    data_generator = generate_energy_market_data(chunk_size=chunk_size)

    # Setup the animation function to update the plot with new data frames
    ani = FuncAnimation(fig, update, frames=data_generator, init_func=init, blit=True, interval=100, cache_frame_data=False)

    # Display the dynamic plot in real-time
    plt.show(block=True)

    # Save the anomaly detection data to a CSV file after the plot closes
    df = pd.DataFrame(all_data, columns=["Value", "Anomaly"])  # Convert data to a DataFrame
    df.to_csv("energy_market_anomaly_data.csv", index=False)  # Save to CSV
    logging.info("Data saved to energy_market_anomaly_data.csv")  # Log completion
    return df  # Return the saved DataFrame

# -------------------- RUN ANOMALY DETECTION -------------------- #
if __name__ == "__main__":
    logging.info("Starting real-time anomaly detection in energy market...")  # Log start of detection process
    detect_anomalies_in_energy_market(window_size=400, chunk_size=100, max_duration=30)  # Start anomaly detection with the specified parameters
