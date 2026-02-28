"""Data loading utilities for the LUMO and QUGS datasets."""

import os

import numpy as np
import pandas as pd


def load_dataset(data_dir, history=200, num_sensors=18, max_rows=500000):
    """Load and segment vibration sensor data from CSV files.

    Each CSV file in ``data_dir`` contains raw accelerometer readings.
    The first ``num_sensors`` columns are sensor channels and the last
    column is the class label.  Data is segmented into non-overlapping
    windows of length ``history``.

    Args:
        data_dir: Path to directory containing CSV files.
        history: Window length (number of time steps per sample).
        num_sensors: Number of sensor columns to read.
        max_rows: Maximum rows to read per CSV file.

    Returns:
        signals: np.ndarray of shape (N, history, num_sensors).
        labels: np.ndarray of shape (N,).
    """
    signals = []
    labels = []

    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv'))
    print(f"Found {len(csv_files)} CSV files in {data_dir}")

    for fname in csv_files:
        fpath = os.path.join(data_dir, fname)
        df = pd.read_csv(fpath, nrows=max_rows, skiprows=1)
        print(f"  {fname}: {len(df)} rows")

        label = int(df.iloc[0, -1])
        sensor_data = df.iloc[:, :num_sensors].values

        for i in range(0, len(sensor_data) - history + 1, history):
            signals.append(sensor_data[i:i + history])
            labels.append(label)

    return np.array(signals), np.array(labels)
