"""
Data loading utilities for ECAC project.
File used for full exercise 1 and 2.

This module contains functions for:
- loading sensor data from CSV files for a given part,
- loading and caching the full dataset from all parts,
- handling missing files gracefully.

Column conventions expected in loaded arrays:
- Columns 0-12: sensor and metadata values
- Last column: part number (added during part loading)
"""

import numpy as np
import csv

def load_part_data(part_number):
    """
    Load sensor data for a given part from CSV files for all devices.

    Parameters
    ----------
    part_number : int
        The part index (0-13) to load data for.

    Returns
    -------
    data : ndarray, shape (n_samples, n_features+1)
        Array of loaded data with part number appended as last column."""
    
    data = []

    for device in range(1, 6):
        filename = f"data/part{part_number}/part{part_number}dev{device}.csv"
        try:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    data.append([float(x) for x in row] + [part_number])
        except FileNotFoundError:
            print(f"Arquivo não encontrado.")
            continue
    return np.array(data)

def load_data():
    """
    Load the full dataset from all parts, using cached .npy file if available.

    Returns
    -------
    data : ndarray, shape (n_samples, n_features+1)
        Array of loaded data from all parts and devices."""
    
    try:
        data = np.load("data/data.npy", allow_pickle=True)
        return data
    except FileNotFoundError:
        full_data = [load_part_data(i) for i in range(14)]
        data = np.concatenate(full_data, axis=0)
        np.save("data/data.npy", data)
        return data