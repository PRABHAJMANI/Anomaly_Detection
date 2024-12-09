
# Real-Time Anomaly Detection in Energy Market Data Streams

## Project Overview
This project is designed to detect anomalies in real-time energy market data streams, focusing on electricity prices and energy demand. By simulating real-world scenarios, the system identifies unusual patterns using advanced machine learning techniques such as Isolation Forest and ADWIN.

## Features
- **Data Simulation**: Generates realistic electricity price and energy demand data with occasional anomalies.
- **Real-Time Processing**: Detects anomalies dynamically in rolling windows of data.
- **Incremental Learning**: Utilizes an Incremental Isolation Forest for adaptive anomaly detection.
- **Visualization**: Plots real-time electricity prices and energy demand with anomalies highlighted.
- **Drift Detection**: Implements ADWIN to monitor concept drift in the data streams.
- **CSV Export**: Saves anomaly detection results in `energy_market_anomaly_data.csv`.

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- Python 3.7 or higher
- Necessary Python libraries listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/PRABHAJMANI/Anomaly_Detection.git
   cd Anomaly_Detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
Start the anomaly detection system by executing:
```bash
python main.py
```

## Key Functions

### Data Simulation
The `generate_energy_market_data()` function simulates energy market data, including periodic trends with noise and occasional anomalies.

### Incremental Learning
The `IncrementalIsolationForest` class supports incremental fitting of the Isolation Forest model for incoming data chunks.

### Real-Time Anomaly Detection
The `detect_anomalies_in_energy_market()` function performs real-time anomaly detection and visualizes the results dynamically.

## Output
- Real-time plots for electricity prices and energy demand with anomalies highlighted.
- A CSV file (`energy_market_anomaly_data.csv`) containing anomaly detection results.

## Customization
- Adjust the maximum runtime by modifying the `max_duration` parameter in the `detect_anomalies_in_energy_market()` function.
- Change the data chunk size by setting the `chunk_size` parameter in `generate_energy_market_data()`.

## Contribution
This project was created to demonstrate real-time anomaly detection in energy markets using machine learning.

## Author
**Prabhpreet Singh Ajmani**  
**Institution**: BITS Pilani

---