"""
Dataset loading.
File used for exercise 1 and 2.

This module contains functions for:
-- loading sensor dataset from CSV files for a given part,
-- loading and caching the full dataset from all parts,
- handling missing files gracefully.

Column conventions expected in loaded arrays:
- Columns 0-12: sensor and metadata values
- Last column 13: participant number (added during part loading)
"""

from log import print_and_log
from features import extract_features
from outliers import compute_modules, remove_outliers
import numpy as np
import csv

def load_part_data(part_number):
    """Load sensor dataset for a given part from CSV files for all devices.

    Parameters
    ----------
    part_number : int
        The part index (0-13) to load dataset for.

    Returns
    -------
    dataset : matrix, shape (n_samples, n_features+1)
        Matrix of loaded dataset with part number appended as last column."""
    
    dataset = []

    for device in range(1, 6):
        filename = f"dataset/part{part_number}/part{part_number}dev{device}.csv"
        try:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    dataset.append([float(x) for x in row] + [part_number])
        except FileNotFoundError:
            print_and_log(f"File not found: {filename}")
            continue
    return np.array(dataset)

def load_data():
    """Load the full dataset from all parts, using cached .npy file if available.

    Returns
    -------
    dataset : matrix, shape (n_samples, n_features+1)
        Matrix of loaded dataset from all parts and devices."""
    
    try:
        dataset = np.load("npy/dataset.npy", allow_pickle=True)
        return dataset
    except FileNotFoundError:
        full_dataset = [load_part_data(i) for i in range(14)]
        dataset = np.concatenate(full_dataset, axis=0)
        np.save("npy/dataset.npy", dataset)
        return dataset
    
def save_data(dataset, features, labels):
    """Save dataset and variable modules to .npy files.

    Parameters
    ----------
    dataset : matrix, sha
        Raw dataset matrix.
    features : np.ndarray, shape (n_windows, n_features)
        Feature matrix.
    labels : np.ndarray, shape (n_windows, 2)
        Integer array: column 0 is activity, column 1 is participant ID."""
    
    np.save("npy/dataset.npy", dataset)
    np.save("npy/features.npy", features)
    np.save("npy/feature_labels.npy", labels)
    
def reload_data():
	"""
	Load or recompute cached feature artifacts from the `data/` folder.
	Loads precomputed numpy arrays if available; otherwise, recomputes features from raw data and saves them.

	Returns
	-------
	data : np.ndarray
		Raw dataset matrix.
	features : np.ndarray, shape (n_windows, n_features)
		Feature matrix.
	labels : np.ndarray, shape (n_windows, 2)
		Integer array: column 0 is activity, column 1 is participant ID.
	"""
	
	try:
		dataset = np.load("npy/dataset.npy", allow_pickle=True)
		features = np.load("npy/features.npy", allow_pickle=True)
		labels = np.load("npy/feature_labels.npy", allow_pickle=True)

	except FileNotFoundError:
		dataset = load_data()
		variables_modules = compute_modules(dataset)
		dataset, variables_modules = remove_outliers(dataset, variables_modules)

		features, labels = extract_features(dataset, variables_modules, window_duration=5.0, overlap_ratio=0.5)

		np.save("npy/dataset.npy", dataset)
		np.save("npy/features.npy", features)
		np.save("npy/feature_labels.npy", labels)

	return dataset, features, labels