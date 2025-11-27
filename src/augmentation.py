
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
	
	if features is not None and labels is not None:
		mask = labels[:, 0] <= 7
		return features[mask], labels[mask]
	if embeddings is not None and labels is not None:
		mask = labels[:, 0] <= 7
		return embeddings[mask], labels[mask]
	if pca is not None and labels is not None:
		mask = labels[:, 0] <= 7
		return pca[mask], labels[mask]

# --- Exercise 1.1: Data Augmentation with SMOTE ---

def analyze_activity_balance(labels):
	"""
	Analyze the balance of activity samples in the dataset and print sample counts per activity.

	Parameters
	----------
	labels : np.ndarray, shape (n_samples, 2)
		Array where each row contains [activity, participant].

	Returns
	-------
	None
		Prints activity sample counts to stdout.
	"""

	# Count samples per activity
	activities = labels[:, 0]
	unique, counts = np.unique(activities, return_counts=True)
	print_and_log("\n--- Activity sample count ---\n")

	for activity, count in zip(unique, counts):
		print_and_log(f"Activity {activity}: {count} samples")

# --- Exercise 1.2: Data Augmentation with SMOTE ---

def augment_activity_data(features, labels, activity=4, participant=3, n_samples=3):
	"""
	Generate synthetic samples for a specific activity using SMOTE-like interpolation.

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
		Generated synthetic feature matrix (n_samples, n_features).
	"""
	
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

def augment_dataset(train, random_state=None, k_neighbors=5, sampling_strategy='auto'):
    """Augment training dataset using SMOTE applied to combined (features+embeddings) per scenario.
    Returns new train dict with balanced features/embeddings per scenario and aligned full labels.

    Parameters
    ----------
    train : dict
        train dict with keys "features" and "embeddings" (each a dict with scenarios 'a','b','c')
        and "labels" (array, shape (n_samples, n_label_cols)). Uses first column as class label.
    random_state : int or None
        RNG seed for SMOTE and label sampling.
    k_neighbors : int
        SMOTE k_neighbors parameter.
    sampling_strategy : str or dict
        Sampling strategy for SMOTE.
    """
    rng = np.random.default_rng(random_state)

    labels = np.array(train["labels"])
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    # Use first column as class label
    y = labels[:, 0]

    new_train = {"features": {}, "embeddings": {}, "labels": None}

    for scenario in ["a", "b", "c"]:
        X_feat = np.array(train["features"][scenario])
        X_emb = np.array(train["embeddings"][scenario])

        # Ensure 2D
        if X_feat.ndim == 1:
            X_feat = X_feat.reshape(-1, 1)
        if X_emb.ndim == 1:
            X_emb = X_emb.reshape(-1, 1)

        # Combine feature spaces so SMOTE preserves relation between features and embeddings
        X_comb = np.hstack([X_feat.astype(float), X_emb.astype(float)])

        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors)
        X_res_comb, y_res = sm.fit_resample(X_comb, y)

        # Split back into features and embeddings
        n_feat_cols = X_feat.shape[1]
        feat_res = X_res_comb[:, :n_feat_cols]
        emb_res = X_res_comb[:, n_feat_cols:]

        new_train["features"][scenario] = feat_res
        new_train["embeddings"][scenario] = emb_res

        # Build aligned full labels for this resampled set by sampling original full-label rows of same class
        # For reproducibility, use rng.choice
        unique_classes = np.unique(y_res)
        resampled_full_labels = []
        # Precompute indices per class
        class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
        for cls in y_res:
            idx_candidates = class_indices.get(cls)
            # Should always exist, but guard
            if idx_candidates is None or len(idx_candidates) == 0:
                # fallback: pick a random row from labels
                chosen_idx = rng.integers(0, labels.shape[0])
            else:
                chosen_idx = rng.choice(idx_candidates)
            resampled_full_labels.append(labels[chosen_idx])

        resampled_full_labels = np.vstack(resampled_full_labels)

        # If first scenario, set new_train["labels"], else ensure labels shape matches and keep same labels
        if new_train["labels"] is None:
            new_train["labels"] = resampled_full_labels
        else:
            # Ensure that all scenarios produce the same number of samples; SMOTE should give same counts per scenario
            if new_train["labels"].shape[0] != resampled_full_labels.shape[0]:
                raise ValueError("SMOTE produced inconsistent sample counts across scenarios.")
            # otherwise nothing to do (labels already set)

    return new_train

# --- Exercise 1.3: Visualize Synthetic vs Real Samples ---

def plot_synthetic_vs_real(features, labels, synthetic_features, activity=4, participant=3):
	"""
	Visualize real and synthetic samples using a 2D scatter plot of the first two features.

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
		Participant ID to plot (default=3).

	Returns
	-------
	None
		Displays a scatter plot of real and synthetic samples.
	"""
	
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