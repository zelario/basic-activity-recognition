
"""
Data augmentation and feature artifact utilities for ECAC project.
File used for pre-game and feature engineering steps.

This module contains functions for:
- loading or recomputing cached feature artifacts,
- discarding samples by activity label,
- performing SMOTE oversampling,
- visualizing augmented data and PCA results.

Artifacts expected in `data/` folder:
- features.npy, pca.npy, scores.npy, labels.npy
"""
from load import *
from outliers import *
from features import *

import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# --- Pre Game ---

def reload_data():
	"""Load or recompute cached feature artifacts.

	Attempts to load precomputed numpy artifacts from the `data/` folder:
	`features.npy`, `pca.npy`, `scores.npy`, and `labels.npy`. If all files
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
	scores : matrix, shape (3, n_features)
		Object array containing `[feature_names, fisher_features, relief_features]`.
	labels : matrix, shape (n_windows, 2)
		Integer array where column 0 is activity and column 1 is participant id. """
	
	try:
		features = np.load("data/features.npy", allow_pickle=True)
		pca = np.load("data/pca.npy", allow_pickle=True)
		scores = np.load("data/scores.npy", allow_pickle=True)
		labels = np.load("data/labels.npy", allow_pickle=True)

	except FileNotFoundError:
		data = load_data()
		variables_modules = compute_modules(data)

		features, labels, feature_names = extract_features(data, variables_modules, fs=51.5, window_duration=5.0, overlap_ratio=0.5)

		n_components = 36
		pca, explained_variance_ratio = compute_pca(features, n_components)

		analyse_pca(explained_variance_ratio)

		fisher_features = fisher(features, labels, feature_names)
		relief_features = relief(features, labels, feature_names=feature_names, n_neighbors=100)

		scores = np.array([
			feature_names,
			fisher_features,
			relief_features
		], dtype=object)

		np.save("data/features.npy", features)
		np.save("data/pca.npy", pca)
		np.save("data/labels.npy", labels)
		np.save("data/scores.npy", scores)

	return features, pca, scores, labels

def discard_activities(features, pca, labels):
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
	
	mask = labels[:, 0] <= 7
	return features[mask], pca[mask], labels[mask]

# --- Exercise 1.1: Data Augmentation with SMOTE ---

def analyze_activity_balance(labels):
	"""
	Analyze the balance of activity samples in the dataset.

	Parameters
	----------
	labels : matrix, shape (n_samples, 2)
		Array of shape (n_samples, 2), where each row contains [activity, participant].

	Returns
	-------
	label_count: dictionary
		Dictionary mapping activity label to number of samples."""

	activities = labels[:, 0]
	unique, counts = np.unique(activities, return_counts=True)
	print("\n--- Activity sample count ---\n")
	for activity, count in zip(unique, counts):
		print(f"Activity {activity}: {count} samples")
	label_count = dict(zip(unique, counts))
	return label_count


def augment_activity_data(features, pca, labels, n_samples=3, activity=4, participant=3):
	"""Augment the dataset by generating synthetic samples for a specific activity and participant using SMOTE.

	Parameters
	----------
	features : matrix, shape (n_samples, n_features)
		Feature matrix of shape (n_samples, n_features) for all samples.
	pca : matrix, shape (n_samples, n_components)
		PCA-transformed matrix of shape (n_samples, n_components) for all samples.
	labels : matrix, shape (n_samples, 2)
		Array of shape (n_samples, 2), where each row contains [activity, participant] for each sample.
	n_samples : int, optional
		Number of synthetic samples to generate (default is 3).
	activity : int, optional
		Activity label to augment (default is 4).
	participant : int, optional
		Participant ID to augment (default is 3).

	Returns
	-------
	features_augmented : matrix, shape (n_samples + n_synthetic, n_features)
		Feature matrix with synthetic samples appended.
	pca_augmented : matrix, shape (n_samples + n_synthetic, n_components)
		PCA matrix with synthetic samples appended.
	labels_augmented : matrix, shape (n_samples + n_synthetic, 2)
		Label array with synthetic labels appended.
	new_rows : np.ndarray
		Array of row indices (in the augmented arrays) corresponding to the new synthetic samples."""

	print("\n--- Augmenting dataset with SMOTE ---\n")

	test=False
	if test:
		n_samples= int(input("Number of samples: "))
		activity= int(input("Activity to augment: "))
		participant= int(input("Participant to augment: "))

	# Filter for the chosen activity
	mask = (labels[:, 0] == activity)
	activity_features = features[mask]
	
	# Create a dummy target for SMOTE (must have at least 2 samples)
	dummy = np.array([0] * (activity_features.shape[0] - 1) + [1])
	smote = SMOTE(sampling_strategy={0: activity_features.shape[0], 1: activity_features.shape[0] + n_samples}, k_neighbors=min(5, activity_features.shape[0]-1))
	new_activity_features, new_dummy = smote.fit_resample(activity_features, dummy)

	# Only keep the synthetic samples (those with class 1 and index >= activity_features.shape[0])
	synthetic_mask = (new_dummy == 1)[activity_features.shape[0]:]
	synthetic_features = new_activity_features[activity_features.shape[0]:][synthetic_mask]
	
	# Create synthetic labels
	synthetic_labels = np.tile([activity, participant], (synthetic_features.shape[0], 1))

	# Project synthetic features into PCA space
	pca_model = PCA(n_components=pca.shape[1])
	pca_model.fit(features)
	synthetic_pca = pca_model.transform(synthetic_features)

	# Concatenate synthetic samples to original data
	features_augmented = np.vstack([features, synthetic_features])
	pca_augmented = np.vstack([pca, synthetic_pca])
	labels_augmented = np.vstack([labels, synthetic_labels])

	# Indices of new synthetic samples in the augmented arrays
	synthetic_rows = np.arange(features.shape[0], features_augmented.shape[0])

	return features_augmented, pca_augmented, labels_augmented, synthetic_rows

def plot_synthetic_vs_real(features, labels, scores, synthetic_rows):
	"""Visualize real and synthetic samples using a 2D scatter plot of the first two features.

	Parameters
	----------
	features : matrix, shape (n_samples, n_features)
		Feature matrix of shape (n_samples, n_features) for all samples.
	labels : matrix, shape (n_samples, 2)
		Array of shape (n_samples, 2), where each row contains [activity, participant] for each sample.
	synthetic_rows : array
		Array of row indices corresponding to synthetic samples in the features/labels arrays."""
	
	plt.figure(figsize=(8, 6))

	# Plot real samples by activity (excluding synthetic)
	real_mask = np.ones(features.shape[0], dtype=bool)
	real_mask[synthetic_rows] = False
	for act in np.unique(labels[real_mask, 0]):
		mask = (labels[:, 0] == act) & real_mask
		plt.scatter(features[mask, 0], features[mask, 1], label=f"Activity {act}", alpha=0.6)

	# Plot synthetic samples on top, with black edge
	plt.scatter(features[synthetic_rows, 0], features[synthetic_rows, 1], c='red', marker='*', s=200, edgecolor='black', linewidths=1.5, label='Synthetic', zorder=10)
	plt.xlabel(scores[0][0])
	plt.ylabel('Feature 2')
	plt.title('Synthetic vs Real Samples (First 2 Features)')
	plt.legend()
	plt.tight_layout()
	plt.show()
