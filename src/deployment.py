"""
Deployment and sample generation utilities.
File used for exercise 6.

This module contains functions for:
- Generating random samples from the dataset for evaluation.
- Formatting sensor data for model input.
- Deploying the trained model to classify new sensor samples.
- Applying feature extraction, normalization, feature selection, and classification steps.
"""

from augmentation import discard_activities
from features import *
from outliers import *
from embeddings import *
from models import *
import numpy as np
from augmentation import augment_dataset

def get_random_sample(dataset):
    """Select a random sample of 256 rows from a chosen activity, participant, and device.

    Parameters
    ----------
    dataset : matrix
        Raw dataset matrix (n_samples, 13 columns).

    Returns
    -------
    sample : matrix, shape (256, 9)
        Array of shape (256, 9) with sensor data (acc x y z, gyr x y z, mag x y z)."""

    # Only consider activities 1 to 7
    activities_mask = (dataset[:, 11] >= 1) & (dataset[:, 11] <= 7)
    dataset = dataset[activities_mask]

    # Randomly choose activity, participant, device
    activities = np.unique(dataset[:, 11])
    participants = np.unique(dataset[:, 12])
    devices = np.unique(dataset[:, 0])

    activity = np.random.choice(activities)
    participant = np.random.choice(participants)
    device = np.random.choice(devices)

    print_and_log("\n--- Random Sample Generation ---")
    print_and_log("\nChoosing random sample from activity:", int(activity))

    mask = (
        (dataset[:, 12] == participant) &
        (dataset[:, 0] == device) &
        (dataset[:, 11] == activity)
    )

    group = dataset[mask]
    if group.shape[0] >= 256:
        shuffled_group = group.copy()
        np.random.shuffle(shuffled_group)
        sample = shuffled_group[:256, 1:10]
        return sample
    else:
        return get_random_sample(dataset)

def _format_sample(sample_dataset):
    """Format a sample to match the expected input for feature extraction and classification.

    Parameters
    ----------
    sample_dataset : matrix, shape (256, 9)
        Array with sensor data.

    Returns
    -------
    formated_sample : matrix, shape (256, 13)
        Array with all required columns for downstream processing."""
    
    formated_sample = np.zeros((256, 13))
    formated_sample[:, 0] = 0  # device id
    formated_sample[:, 1:4] = sample_dataset[:, 0:3]   # acc x, y, z
    formated_sample[:, 4:7] = sample_dataset[:, 3:6]   # gyr x, y, z
    formated_sample[:, 7:10] = sample_dataset[:, 6:9]  # mag x, y, z
    formated_sample[:, 10] = 0  # timestamp
    formated_sample[:, 11] = 1  # activity label
    formated_sample[:, 12] = 1  # participant id
    return formated_sample

def deployment_model(dataset, labels, k=19):
    """Train and return a knn model for deployment using scikit-learn's KNeighborsClassifier.

    Parameters
    ----------
    train_dataset : matrix"""

    dataset, means, stds = zscore_normalization(dataset, return_parameters=True)

    # Compute PCA on the dataset
    pca, explained_variances, pca_object = compute_pca(dataset)

    # Determine number of components to retain 90% variance
    cumulative = np.cumsum(explained_variances)
    n_components = np.argmax(cumulative >= 0.9) + 1

    pca = pca[:, :n_components]

    knn_model = sklearn_knn_classifier(pca, labels, k=k)

    model = (knn_model, means, stds, pca_object, n_components)

    return model

def classify_sample(model, sample):
    """Deploy the trained model to make predictions on a sample dataset.

    This function loads the trained model features and labels, applies activity filtering and augmentation,
    formats the input sample, extracts features, applies normalization and feature selection, and predicts
    the activity label using k-NN.

    Parameters
    ----------
    sample_dataset : matrix, shape (256, 9)
        Array of shape (256, 9) with sensor data to classify.
    """

    knn_model, means, stds, pca_object, n_components = model

    print_and_log("\n--- Making predictions on sample dataset ---\n")
    
    # Format sample dataset to match every other function

    sample = _format_sample(sample)

    # Extract features from sample 
    sample_features, _ = compute_features(sample, window_duration=5.0, overlap=0.0, reaload=False)

    # Normalize sample
    sample_features = zscore_normalization(sample_features, mean_values=means, std_values=stds)

    # Apply PCA to sample
    sample_pca = compute_pca(sample_features, pca_object=pca_object)
    sample_pca = sample_pca[:, :n_components]

    # Predict label using k-NN
    predicted_label = knn_model.predict(sample_pca)

    print_and_log(f"Predicted label for the sample dataset: {predicted_label}")
