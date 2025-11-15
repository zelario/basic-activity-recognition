import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from load import *
from outliers import *
from features import *

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
		print("--- Loaded data from existing .npy files ---")

	except FileNotFoundError:
		data = load_data()
		acc_modules = variable_module(data, "Acceleration")
		mag_modules = variable_module(data, "Magnetic Field")
		gyro_modules = variable_module(data, "Angular Velocity")

		features, labels, feature_names = extract_features(
			data, acc_modules, mag_modules, gyro_modules,
			fs=51.5, window_duration=5.0, overlap_ratio=0.5)

		n_components = 36
		pca, explained_variance_ratio = pca(features, n_components)

		pca_analysis(explained_variance_ratio)

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

# --- Game On ---

def analyze_activity_balance(labels):
	"""
	Analyze the balance of activity samples in the dataset.

	Parameters
	----------
	labels : np.ndarray
		Array of shape (n_samples, 2), where each row contains [activity, participant].

	Returns
	-------
	dict
		Dictionary mapping activity label to number of samples.

	Prints
	------
	Number of samples for each activity in the dataset.
	Useful for checking if the dataset is balanced across activities.
	"""
	activities = labels[:, 0]
	unique, counts = np.unique(activities, return_counts=True)
	print("Activity balance:")
	for u, c in zip(unique, counts):
		print(f"Activity {u}: {c} samples")
	return dict(zip(unique, counts))

def smote_generate_samples(features, labels, activity, participant, k=1, random_state=42):
	"""
	Generate synthetic samples for a specific activity and participant using SMOTE.

	Parameters
	----------
	features : np.ndarray
		Feature matrix of shape (n_samples, n_features).
	labels : np.ndarray
		Array of shape (n_samples, 2), where each row contains [activity, participant].
	activity : int
		Activity label for which to generate synthetic samples.
	participant : int
		Participant ID for which to generate synthetic samples.
	k : int, optional
		Number of synthetic samples to generate (default is 1).
	random_state : int, optional
		Random seed for reproducibility (default is 42).

	Returns
	-------
	X_synth : np.ndarray
		Synthetic feature matrix of shape (k, n_features).
	labels_synth : np.ndarray
		Synthetic labels array of shape (k, 2), each row is [activity, participant].

	Raises
	------
	ValueError
		If there are not enough samples for SMOTE to operate (minimum 2 required).

	Notes
	-----
	Only samples from the specified activity and participant are used for generating synthetic data.
	SMOTE creates new samples by interpolating between existing samples.
	"""
	# Filter for the given activity and participant
	mask = (labels[:, 0] == activity) & (labels[:, 1] == participant)
	X = features[mask]
	y = labels[mask][:, 0]  # SMOTE expects 1D labels
	if len(X) < 2:
		raise ValueError("Not enough samples for SMOTE.")
	smote = SMOTE(sampling_strategy={activity: len(X) + k}, k_neighbors=min(5, len(X)-1), random_state=random_state)
	X_aug, y_aug = smote.fit_resample(X, y)
	# Get only the synthetic samples
	n_orig = len(X)
	X_synth = X_aug[n_orig:]
	y_synth = y_aug[n_orig:]
	# Build synthetic labels (activity, participant)
	labels_synth = np.column_stack([y_synth, np.full_like(y_synth, participant)])
	return X_synth, labels_synth

def visualize_synthetic_samples(features, labels, X_synth, labels_synth):
	"""
	Visualize real and synthetic samples using a 2D scatter plot of the first two features.

	Parameters
	----------
	features : np.ndarray
		Feature matrix of shape (n_samples, n_features) for real samples.
	labels : np.ndarray
		Array of shape (n_samples, 2), where each row contains [activity, participant] for real samples.
	X_synth : np.ndarray
		Feature matrix of shape (k, n_features) for synthetic samples.
	labels_synth : np.ndarray
		Array of shape (k, 2), where each row contains [activity, participant] for synthetic samples.

	Returns
	-------
	None

	Displays
	--------
	A matplotlib scatter plot:
		- Real samples are colored by activity.
		- Synthetic samples are highlighted in red with a star marker.
		- Only the first two features are plotted for visualization clarity.
	"""
	plt.figure(figsize=(8, 6))
	# Plot real samples
	for act in np.unique(labels[:, 0]):
		mask = labels[:, 0] == act
		plt.scatter(features[mask, 0], features[mask, 1], label=f"Activity {act}", alpha=0.6)
	# Plot synthetic samples
	plt.scatter(X_synth[:, 0], X_synth[:, 1], c='red', marker='*', s=150, label='Synthetic')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Synthetic vs Real Samples (First 2 Features)')
	plt.legend()
	plt.tight_layout()
	plt.show()
