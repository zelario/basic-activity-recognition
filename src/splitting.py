"""
Dataset splitting utilities.
File used in Exercise 3.

This module provides functions for:
- Splitting features, embeddings, and labels into train/validate/test sets
- Supporting mixed and participant-based splits
- Ensuring consistent splits across all representations
"""

from features import *

from sklearn.model_selection import train_test_split
import numpy as np

# --- Exercise 3.1: Mixed participant splitting ---

def mixed_splitting(features, embeddings, labels, seed=67, increment=0):
    """Split features, embeddings, and labels into training, validate, and test sets using mixed participant splitting.
    Ensures consistent splits across all representations and stratifies by the first label column.

    Parameters
    ----------
    features : np.ndarray, shape (n_windows, n_features)
        Feature matrix for all windows.
    embeddings : np.ndarray, shape (n_windows, n_embedding_features)
        Embeddings matrix for all windows.
    labels : np.ndarray, shape (n_windows, n_label_features)
        Labels array for all windows.

    Returns
    -------
    train_dataset : tuple
        (features, embeddings, labels) for training set.
    validate_dataset : tuple or None
        (features, embeddings, labels) for validate set, or None if validate=False.
    test_dataset : tuple
        (features, embeddings, labels) for test set."""

    # First split: Train+Val and Test
    train_val_features, test_features, train_val_embeddings, test_embeddings, train_val_labels, test_labels = train_test_split(
        features, embeddings, labels, test_size=0.2, random_state=seed+increment, stratify=labels[:, 1])

    # Second split: Train and Val
    train_features, val_features, train_embeddings, validate_embeddings, train_labels, validate_labels = train_test_split(
        train_val_features, train_val_embeddings, train_val_labels, test_size=0.2, random_state=seed+increment, stratify=train_val_labels[:, 1])
    
    train_dataset = np.array(train_features), np.array(train_embeddings), np.array(train_labels)
    validate_dataset = np.array(val_features), np.array(validate_embeddings), np.array(validate_labels)
    test_dataset = np.array(test_features), np.array(test_embeddings), np.array(test_labels)

    split = {"train": train_dataset, "validate": validate_dataset, "test": test_dataset}

    return split

# --- Exercise 3.2: Participant-based splitting ---

def participant_splitting(features, embeddings, labels, train_n=9, validate_n=3, test_n=3, seed=67, increment=0):
    """Split features, embeddings, and labels into training, validate, and test sets by participant groups.
    Ensures no data leakage between sets by assigning unique participants to each split.

    Parameters
    ----------
    features : np.ndarray, shape (n_windows, n_features)
        Feature matrix for all windows.
    embeddings : np.ndarray, shape (n_windows, n_embedding_features)
        Embeddings matrix for all windows.
    labels : np.ndarray, shape (n_windows, n_label_features)
        Labels array for all windows. Assumes participant ID is in column 1.
    train_n : int, optional
        Number of participants in training set (default=9).
    validate_n : int, optional
        Number of participants in validate set (default=3).
    test_n : int, optional
        Number of participants in test set (default=3).
    validate : bool, optional
        Whether to create a validate split (default=True).

    Returns
    -------
    train_dataset : tuple
        (features, embeddings, labels) for training set.
    validate_dataset : tuple or None
        (features, embeddings, labels) for validate set, or None if validate=False.
    test_dataset : tuple
        (features, embeddings, labels) for test set."""

    random = np.random.default_rng(seed+increment)
    participants = np.unique(labels[:, 1])
    participants_shuffled = random.permutation(participants)

    # Split participants into Train, validate, and Test
    train_participants = participants_shuffled[:train_n]
    validate_participants = participants_shuffled[train_n:train_n+validate_n]
    test_participants = participants_shuffled[train_n+validate_n:train_n+validate_n+test_n]

    # Create masks for each split
    train_mask = np.isin(labels[:, 1], train_participants)
    validate_mask = np.isin(labels[:, 1], validate_participants)
    test_mask = np.isin(labels[:, 1], test_participants)

    # Create datasets
    train_dataset = np.array(features[train_mask]), np.array(embeddings[train_mask]), np.array(labels[train_mask])
    validate_dataset = np.array(features[validate_mask]), np.array(embeddings[validate_mask]), np.array(labels[validate_mask])
    test_dataset = np.array(features[test_mask]), np.array(embeddings[test_mask]), np.array(labels[test_mask])

    split = {"train": train_dataset, "validate": validate_dataset, "test": test_dataset}

    return split

# --- Exercise 3.4: Pipeline preparation ---

def prepare_pipeline(split):
    """Prepare three feature transformation scenarios for train, validate, and test sets:
      a) All features/embeddings (no transformation)
      b) PCA-reduced features (retain 90% variance, fit on train only)
      c) ReliefF-selected top 15 features (fit on train only)

    All transformations (PCA, ReliefF) are fit ONLY on the training set and applied to validate and test sets.

    Parameters
    ----------
    train_dataset : tuple
        (features, embeddings, labels) for training set.
    validate_dataset : tuple
        (features, embeddings, labels) for validate set.
    test_dataset : tuple
        (features, embeddings, labels) for test set.

    Returns
    -------
    pipeline : np.ndarray, shape (3, 5)
        Array containing for each split (train, val, test):
        [all_features, pca_features, relief_features, embeddings, labels]"""

    # === Scenario a: All features/embeddings ===

    # Unpack datasets
    train_features, train_embeddings, train_labels = split["train"]
    validate_features, validate_embeddings, validate_labels = split["validate"]

    combined_features = np.concatenate([train_features, validate_features], axis=0)
    combined_embeddings = np.concatenate([train_embeddings, validate_embeddings], axis=0)
    combined_labels = np.concatenate([train_labels, validate_labels], axis=0)
    
    test_features, test_embeddings, test_labels = split["test"]

    # === Scenario b: PCA-reduced features and embeddings (90% variance) ===

    # Normalize training and combined features and embeddings
    normalized_train_features, train_features_means, train_features_stds = zscore_normalization(train_features, return_parameters=True)
    normalized_train_embeddings, train_embeddings_means, train_embeddings_stds = zscore_normalization(train_embeddings, return_parameters=True)

    normalized_combined_features, combined_features_means, combined_features_stds = zscore_normalization(combined_features, return_parameters=True)
    normalized_combined_embeddings, combined_embeddings_means, combined_embeddings_stds = zscore_normalization(combined_embeddings, return_parameters=True)

    # Compute PCA on normalized training and combined features and embeddings
    train_features_pca, explained_variance_train_features, pca_object_train_features = compute_pca(normalized_train_features)
    train_embeddings_pca, explained_variance_train_embeddings, pca_object_train_embeddings = compute_pca(normalized_train_embeddings)

    combined_features_pca, explained_variance_combined_features, pca_object_combined_features = compute_pca(normalized_combined_features)
    combined_embeddings_pca, explained_variance_combined_embeddings, pca_object_combined_embeddings = compute_pca(normalized_combined_embeddings)

    # Determine number of components to retain 90% variance
    cumulative_train_features = np.cumsum(explained_variance_train_features)
    n_components_90_train_features = np.argmax(cumulative_train_features >= 0.9) + 1

    cumulative_train_embeddings = np.cumsum(explained_variance_train_embeddings)
    n_components_90_train_embeddings = np.argmax(cumulative_train_embeddings >= 0.9) + 1

    cumulative_combined_features = np.cumsum(explained_variance_combined_features)
    n_components_90_features_combined = np.argmax(cumulative_combined_features >= 0.9) + 1

    cumulative_combined_embeddings = np.cumsum(explained_variance_combined_embeddings)
    n_components_90_embeddings_combined = np.argmax(cumulative_combined_embeddings >= 0.9) + 1

    # Reduce to selected number of components
    train_features_pca = train_features_pca[:, :n_components_90_train_features]
    train_embeddings_pca = train_embeddings_pca[:, :n_components_90_train_embeddings]

    combined_features_pca = combined_features_pca[:, :n_components_90_features_combined]
    combined_embeddings_pca = combined_embeddings_pca[:, :n_components_90_embeddings_combined]

    # Apply normalization to validate set
    normalized_validate_features = zscore_normalization(validate_features, mean_values=train_features_means, std_values=train_features_stds)
    normalized_validate_embeddings = zscore_normalization(validate_embeddings, mean_values=train_embeddings_means, std_values=train_embeddings_stds)

    # Project validate set using PCA fitted on training set
    validate_features_pca = compute_pca(normalized_validate_features, pca_object=pca_object_train_features)
    validate_embeddings_pca = compute_pca(normalized_validate_embeddings, pca_object=pca_object_train_embeddings)

    validate_features_pca = validate_features_pca[:, :n_components_90_train_features]
    validate_embeddings_pca = validate_embeddings_pca[:, :n_components_90_train_embeddings]

    # Apply normalization to test set
    normalized_test_features = zscore_normalization(test_features, mean_values=combined_features_means, std_values=combined_features_stds)
    normalized_test_embeddings = zscore_normalization(test_embeddings, mean_values=combined_embeddings_means, std_values=combined_embeddings_stds)

    # Project test set using PCA fitted on training set
    test_features_pca = compute_pca(normalized_test_features, pca_object=pca_object_combined_features)
    test_embeddings_pca = compute_pca(normalized_test_embeddings, pca_object=pca_object_combined_embeddings)

    test_features_pca = test_features_pca[:, :n_components_90_features_combined]
    test_embeddings_pca = test_embeddings_pca[:, :n_components_90_embeddings_combined]
    
    # === Scenario c: ReliefF-selected top 15 features ===

    # Select top 15 features using ReliefF on normalized training features and embeddings
    top_15_train_features_indices = relief(normalized_train_features, train_labels, top_n=15, print_output=False)
    top_15_train_embeddings_indices = relief(normalized_train_embeddings, train_labels, top_n=15, print_output=False)

    top_15_combined_features_indices = relief(normalized_combined_features, combined_labels, top_n=15, print_output=False)
    top_15_combined_embeddings_indices = relief(normalized_combined_embeddings, combined_labels, top_n=15, print_output=False)

    # Apply feature selection to all splits
    train_features_relief = normalized_train_features[:, top_15_train_features_indices]
    validate_features_relief = normalized_validate_features[:, top_15_train_features_indices]
    combined_features_relief = normalized_combined_features[:, top_15_combined_features_indices]
    test_features_relief = normalized_test_features[:, top_15_combined_features_indices]

    train_embeddings_relief = normalized_train_embeddings[:, top_15_train_embeddings_indices]
    validate_embeddings_relief = normalized_validate_embeddings[:, top_15_train_embeddings_indices]
    combined_embeddings_relief = normalized_combined_embeddings[:, top_15_combined_embeddings_indices]
    test_embeddings_relief = normalized_test_embeddings[:, top_15_combined_embeddings_indices]

    # Prepare pipeline dictionary
    pipeline = {
        "train": {
            "features": {
                "a": np.array(normalized_train_features),
                "b": np.array(train_features_pca),
                "c": np.array(train_features_relief)
            },
            "embeddings": {
                "a": np.array(normalized_train_embeddings),
                "b": np.array(train_embeddings_pca),
                "c": np.array(train_embeddings_relief)
            },
            "labels": np.array(train_labels)
        },
        "validate": {
            "features": {
                "a": np.array(normalized_validate_features),
                "b": np.array(validate_features_pca),
                "c": np.array(validate_features_relief)
            },
            "embeddings": {
                "a": np.array(normalized_validate_embeddings),
                "b": np.array(validate_embeddings_pca),
                "c": np.array(validate_embeddings_relief)
            },
            "labels": np.array(validate_labels)
        },
        "combined": {
            "features": {
                "a": np.array(normalized_combined_features),
                "b": np.array(combined_features_pca),
                "c": np.array(combined_features_relief)
            },
            "embeddings": {
                "a": np.array(normalized_combined_embeddings),
                "b": np.array(combined_embeddings_pca),
                "c": np.array(combined_embeddings_relief)
            },
            "labels": np.array(combined_labels)
        },
        "test": {
            "features": {
                "a": np.array(normalized_test_features),
                "b": np.array(test_features_pca),
                "c": np.array(test_features_relief)
            },
            "embeddings": {
                "a": np.array(normalized_test_embeddings),
                "b": np.array(test_embeddings_pca),
                "c": np.array(test_embeddings_relief)
            },
            "labels": np.array(test_labels)
        }
    }

    return pipeline