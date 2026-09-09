
"""
Data augmentation and feature artifact utilities for ECAC project.
File used for pre-game and feature engineering steps.

This module contains functions for:
- loading or recomputing cached feature artifacts,
- discarding samples by activity label,
- performing SMOTE oversampling,
- visualizing augmented data and PCA results.
"""

from load import *
from outliers import *
from features import *
from log import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE

# --- Pre Game ---

def discard_activities(features=None, pca=None, labels=None, embeddings=None):
	"""Discard samples whose activity label is greater than 7.

	Parameters
	----------
	features : np.ndarray, optional
		Feature matrix (n_samples, n_features).
	pca : np.ndarray, optional
		PCA-transformed matrix (n_samples, n_components).
	labels : np.ndarray, optional
		Integer array (n_samples, 2): column 0 is activity, column 1 is participant ID.
	embeddings : np.ndarray, optional
		Embeddings matrix (n_samples, n_embedding_features).

	Returns
	-------
	tuple
		Filtered arrays containing only rows for which activity <= 7."""
	
	valid_activities = [1, 2, 3, 4, 5, 6, 7]
	
	if features is not None and labels is not None:
		mask = np.isin(labels[:, 0], valid_activities)
		return features[mask], labels[mask]
	if embeddings is not None and labels is not None:
		mask = np.isin(labels[:, 0], valid_activities)
		return embeddings[mask], labels[mask]
	if pca is not None and labels is not None:
		mask = np.isin(labels[:, 0], valid_activities)
		return pca[mask], labels[mask]

# ---  1.1: Data Augmentation with SMOTE ---

def analyze_activity_balance(labels):
	"""Analyze the balance of activity samples in the dataset and print sample counts per activity.

	Parameters
	----------
	labels : np.ndarray, shape (n_samples, 2)
		Array where each row contains [activity, participant].

	Returns
	-------
	None
		Prints activity sample counts to stdout."""

	# Count samples per activity
	activities = labels[:, 0]
	unique, counts = np.unique(activities, return_counts=True)
	print_and_log("\n--- Activity sample count ---\n")

	for activity, count in zip(unique, counts):
		print_and_log(f"Activity {activity}: {count} samples")

# ---  1.2: Data Augmentation with SMOTE ---

def augment_activity_data(features, labels, activity=4, participant=3, n_samples=3):
	"""Generate synthetic samples for a specific activity using SMOTE-like interpolation.

	Parameters
	----------
	features : np.ndarray
		Original feature matrix (n_samples, n_features).
	labels : np.ndarray
		Label matrix (n_samples, 2): column 0 is activity, column 1 is participant.
	activity : int, optional
		Activity class to augment (default=4).
	participant : int, optional
		Participant ID for synthetic samples (default=3).
	n_samples : int, optional
		Number of synthetic samples to generate (default=3).

	Returns
	-------
	synthetic_features : np.ndarray
		Generated synthetic feature matrix (n_samples, n_features)."""
	
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

def augment_dataset(dataset, labels):
	"""Augment features using SMOTE to balance activity classes.

	Parameters
	----------
	features : np.ndarray
		Feature matrix (n_samples, n_features).
	labels : np.ndarray
		Label matrix (n_samples, 2): column 0 is activity, column 1 is participant.

	Returns
	-------
	augmented_features : np.ndarray
		Augmented feature matrix after SMOTE.
	augmented_labels : np.ndarray
		Corresponding labels for augmented samples."""
	
	# Ensure arrays have the same number of samples
	n_samples = min(dataset.shape[0], labels.shape[0])
	dataset = dataset[:n_samples]
	labels = labels[:n_samples]

	smote = SMOTE()
	augmented_dataset, augmented_activity_labels = smote.fit_resample(dataset, labels[:, 0])

	# Reconstruct labels with random participant and device ids for synthetic samples
	synthetic_count = augmented_activity_labels.shape[0] - labels.shape[0]
	original_participant_ids = labels[:, 1].reshape(-1, 1)
	rng = np.random.default_rng()
	synthetic_participant_ids = rng.integers(1, 15, size=(synthetic_count, 1))
	augmented_participant_ids = np.vstack((original_participant_ids, synthetic_participant_ids))
	augmented_labels = np.hstack((augmented_activity_labels.reshape(-1, 1), augmented_participant_ids))

	return augmented_dataset, augmented_labels

def augment_train_dataset(train_dataset):
	"""Augment each features and embeddings array in a train pipeline dict using SMOTE.

	Parameters
	----------
	train : dict
		Dictionary with keys 'features', 'embeddings', and 'labels'.

	Returns
	-------
	dict
		Updated train dict with augmented arrays."""

	augmented_train_dataset = {"features": {}, "embeddings": {}, "labels": None}
	labels = train_dataset["labels"]

	# Augment features
	for key, matrix in train_dataset["features"].items():
		augmented_matrix, augmented_labels = augment_dataset(matrix, labels)
		augmented_train_dataset["features"][key] = augmented_matrix
		augmented_train_dataset["labels"] = augmented_labels  

	# Augment embeddings
	for key, matrix in train_dataset["embeddings"].items():
		augmented_matrix, _ = augment_dataset(matrix, labels)
		augmented_train_dataset["embeddings"][key] = augmented_matrix

	return augmented_train_dataset

# ---  1.3: Visualize Synthetic vs Real Samples ---

def plot_synthetic_vs_real(features, labels, synthetic_features, activity=4, participant=3):
	"""Visualize real and synthetic samples using a 2D scatter plot of the first two features.

	Parameters
	----------
	features : np.ndarray
		Feature matrix for all real samples (n_samples, n_features).
	labels : np.ndarray
		Array for all real samples (n_samples, 2).
	synthetic_features : np.ndarray
		Feature matrix for synthetic samples (n_synthetic, n_features).
	activity : int, optional
		Activity ID to plot (default=4).
	participant : int, optional
		Participant ID to plot (default=3)."""
	
	plt.figure(figsize=(8, 6))

	# Create a mask for the specific real samples to plot
	real_samples_mask = (labels[:, 0] == activity) & (labels[:, 1] == participant)
	
	# Plot the real samples for the selected activity and participant
	plt.scatter(features[real_samples_mask, 0], features[real_samples_mask, 1], label="Real", alpha=0.6)

	# Plot synthetic samples directly from the provided array
	if synthetic_features.any():
		plt.scatter(synthetic_features[:, 0], synthetic_features[:, 1], c='red', label='Synthetic', alpha=0.6)
	
	plt.xlabel(FEATURES_NAMES[0])
	plt.ylabel(FEATURES_NAMES[1])
	plt.title(f'Activity {activity}, Participant {participant}: Synthetic vs Real Samples')
	plt.legend()
	plt.tight_layout()
	plt.show()