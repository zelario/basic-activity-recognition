
"""
Data augmentation and feature artifact utilities for ECAC project.
File used for pre-game and feature engineering steps.

This module contains functions for:
- loading or recomputing cached feature artifacts,
- discarding samples by activity label,
- performing SMOTE oversampling,
- visualizing augmented data and PCA results.

Artifacts expected in `data/` folder:
- features.npy, pca.npy, labels.npy
"""
from load import *
from outliers import *
from features import *

import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# --- Pre Game ---

def reload_data():
	"""Load or recompute cached feature artifacts.

	Attempts to load precomputed numpy artifacts from the `data/` folder:
	`features.npy`, `pca.npy`, `names.npy`, and `labels.npy`. If all files
	are present they are loaded and returned. If a FileNotFoundError occurs
	(the cache is missing or incomplete) the function recomputes features by
	loading the raw dataset, computing module signals, extracting
	windowed features, running PCA and computing feature rankings. The
	computed matrixes are saved to disk for future runs.

	Returns
	-------
	features : matrix, shape (n_windows, n_features)
		Feature matrix.
	pca : matrix, shape (n_windows, n_components)
		PCA-transformed data.
	labels : matrix, shape (n_windows, 2)
		Integer array where column 0 is activity and column 1 is participant id."""
	
	try:
		data = np.load("data/data.npy", allow_pickle=True)
		features = np.load("data/features.npy", allow_pickle=True)
		labels = np.load("data/labels.npy", allow_pickle=True)

	except FileNotFoundError:
		data = load_data()
		variables_modules = compute_modules(data)

		features, labels = extract_features(data, variables_modules, window_duration=5.0, overlap_ratio=0.5)

		np.save("data/data.npy", data)
		np.save("data/features.npy", features)
		np.save("data/labels.npy", labels)

	return data, features, labels

def discard_activities(features=None, pca=None, labels=None, embeddings=None):
	"""Discard samples whose activity label is greater than 7.

	Parameters
	----------
	features : matrix, shape (n_samples, n_features)
		Feature matrix where rows correspond to samples.
	pca : matrix, shape (n_samples, n_components)
		PCA-transformed representation aligned with `features` (same row order).
	labels : matrix, shape (n_samples, 2)
		Integer array where column 0 is activity id and column 1 is participant id.

	Returns
	-------
	features, pca, labels : tuple of matrixes
		Filtered arrays containing only the rows for which activity <= 7."""
	
	if features is not None and labels is not None:
		mask = labels[:, 0] <= 7
		return features[mask], labels[mask]
	if embeddings is not None and labels is not None:
		mask = labels[:, 0] <= 7
		return embeddings[mask]
	if pca is not None and labels is not None:
		mask = labels[:, 0] <= 7
		return pca[mask], labels[mask]

# --- Exercise 1.1: Data Augmentation with SMOTE ---

def analyze_activity_balance(labels):
	"""Analyze the balance of activity samples in the dataset.

	Parameters
	----------
	labels : matrix, shape (n_samples, 2)
		Array of shape (n_samples, 2), where each row contains [activity, participant].

	Returns
	-------
	label_count: dictionary
		Dictionary mapping activity label to number of samples."""

	# Count samples per activity
	activities = labels[:, 0]
	unique, counts = np.unique(activities, return_counts=True)
	print("\n--- Activity sample count ---\n")

	for activity, count in zip(unique, counts):
		print(f"Activity {activity}: {count} samples")

def augment_activity_data(features, labels, activity=4, participant=3, n_samples=3, ):
    """Generate synthetic samples for a specific activity using SMOTE and project them into PCA space.

    Parameters
    ----------
    features : np.ndarray
        Original feature matrix (n_samples, n_features)
    labels : np.ndarray
        Label matrix where column 0: activity, column 1: participant
    activity : int
        Activity class to augment
    participant : int
        Participant ID for synthetic samples
    n_samples : int
        Number of synthetic samples to generate
    k_neighbors : int
        Number of neighbors for interpolation

    Returns
    -------
    features_augmented : np.ndarray
        Augmented feature matrix
    pca_augmented : np.ndarray
        Augmented PCA matrix
    labels_augmented : np.ndarray
        Augmented labels
    synthetic_indices : np.ndarray
        Indices of synthetic samples in the augmented matrices"""
	
    # Filter samples for the target activity
    mask = labels[:, 0] == activity
    activity_features = features[mask]

    # Adjust k_neighbors if too few samples
    k_neighbors = min(5, activity_features.shape[0] - 1)

    # Fit nearest neighbors model
    nearest_neighbors = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(activity_features).kneighbors(return_distance=False)

    # Generate synthetic samples
    synthetic_features = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(activity_features))
        base = activity_features[idx]
        nn_idx = np.random.choice(nearest_neighbors[idx][1:])
        neighbor = activity_features[nn_idx]

        delta = np.random.rand()
        synthetic_sample = base + delta * (neighbor - base)
        synthetic_features.append(synthetic_sample)

    synthetic_features = np.array(synthetic_features)

    return synthetic_features

def plot_synthetic_vs_real(features, labels, synthetic_features, activity=4, participant=3):
	"""Visualize real and synthetic samples using a 2D scatter plot of the first two features.

	Parameters
	----------
	features : matrix, shape (n_samples, n_features)
		Feature matrix of shape (n_samples, n_features) for all real samples.
	labels : matrix, shape (n_samples, 2)
		Array of shape (n_samples, 2) for all real samples.
	synthetic_features : matrix, shape (n_synthetic, n_features)
		Feature matrix for the synthetic samples.
	activity : int
		The activity ID to plot.
	participant : int
		The participant ID to plot.
	"""
	
	plt.figure(figsize=(8, 6))

	# Create a mask for the specific real samples to plot
	real_samples_mask = (labels[:, 0] == activity) & (labels[:, 1] == participant)
	
	# Plot the real samples for the selected activity and participant
	plt.scatter(features[real_samples_mask, 0], features[real_samples_mask, 1], label="Real", alpha=0.6)

	# Plot synthetic samples directly from the provided array
	if synthetic_features.any():
		plt.scatter(synthetic_features[:, 0], synthetic_features[:, 1], c='red', marker='o', label='Synthetic')
	
	plt.xlabel(FEATURES_NAMES[0])
	plt.ylabel(FEATURES_NAMES[1])
	plt.title(f'Activity {activity}, Participant {participant}: Synthetic vs Real Samples')
	plt.legend()
	plt.tight_layout()
	plt.show()