"""
Deployment and sample generation utilities.
File used for exercise 6.

This module contains functions for:
- Generating random samples from the dataset for evaluation.
- Formatting sensor data for model input.
- Deploying the trained model to classify new sensor samples.
- Applying feature extraction, normalization, feature selection, and classification steps.
"""

from features import *
from outliers import *
from embeddings import *
from models import *
import numpy as np
from augmentation import *

def get_synthetic_sample(dataset, noise_std=0.03, activity=None, participant=None, device=None):
    """Generate a synthetic sample by selecting 256 consecutive rows from real data and adding noise.

    Parameters
    ----------
    dataset : np.ndarray
        Raw dataset (n_rows, 13 columns)
    activity : int or None
    participant : int or None
    device : int or None
    noise_std : float
        Gaussian noise standard deviation

    Returns
    -------
    sample : np.ndarray
        Synthetic sequence (256, 9)"""
    
    # Filter activities
    dataset = dataset[(dataset[:, 11] >= 1) & (dataset[:, 11] <= 7)]

    # Pick random if not specified
    rng = np.random.default_rng()
    if activity is None:
        activity = int(rng.choice(np.unique(dataset[:, 11].astype(int))))
    if participant is None:
        participant = int(rng.choice(np.unique(dataset[:, 12].astype(int))))
    if device is None:
        device = int(rng.choice(np.unique(dataset[:, 0].astype(int))))

    # Select subset
    mask = (dataset[:, 12] == participant) & \
           (dataset[:, 0] == device) & \
           (dataset[:, 11] == activity)
    masked_dataset = dataset[mask][:, 1:10]

    if masked_dataset.shape[0] < 256:
        get_synthetic_sample(dataset)

    try:
        start_index = rng.integers(0, masked_dataset.shape[0] - 256 + 1)
        sample = masked_dataset[start_index:start_index + 256]
        sample = sample + np.random.normal(0, noise_std, sample.shape)
    except ValueError:
        return get_synthetic_sample(dataset)

    label = np.array((activity, participant, device))

    return sample, label

def deployment_model(dataset, labels, k=19):
    """Train and return a knn model for deployment using scikit-learn's KNeighborsClassifier.

    Parameters
    ----------
    train_dataset : matrix"""

    # Apply activity augmentation
    dataset, labels = augment_dataset(dataset, labels)

    # Normalize the dataset and get mean/std for deployment
    dataset, means, stds = zscore_normalization(dataset, return_parameters=True)

    # Train k-NN model
    knn_model = sklearn_knn_classifier(dataset, labels, k=k)

    model = (knn_model, means, stds)

    return model

def classify_sample(model, sample, label):
    """Deploy the trained model to make predictions on a sample dataset.

    This function loads the trained model features and labels, applies activity filtering and augmentation,
    formats the input sample, extracts features, applies normalization and feature selection, and predicts
    the activity label using k-NN.

    Parameters
    ----------
    sample_dataset : matrix, shape (256, 9)
        Array of shape (256, 9) with sensor data to classify.
    """

    knn_model, means, stds = model

    print_and_log("\n--- Making predictions on sample dataset ---\n")
    
    # Format sample dataset to match every other function
    formated_sample = np.zeros((256, 13))
    formated_sample[:, 1:4] = sample[:, 0:3]   # acc x, y, z
    formated_sample[:, 4:7] = sample[:, 3:6]   # gyr x, y, z
    formated_sample[:, 7:10] = sample[:, 6:9]  # mag x, y, z

    # Extract features from sample 
    sample_features, _ = compute_features(formated_sample, window_duration=5.0, overlap=0.0, reload=False)

    # Normalize sample
    sample_features = zscore_normalization(sample_features, mean_values=means, std_values=stds)

    # Predict label using k-NN
    predicted_label = knn_model.predict(sample_features)[0]

    print_and_log(f"Generating synthetic sample for activity {label[0]}")
    print_and_log(f"Predicted label for the sample dataset: {predicted_label}")

    return predicted_label

def test_deployment_model(dataset, model, n=100):
    """Test the deployment model with a synthetic sample."""

    real_labels = []
    predicted_labels = []

    for i in range(n):
        sample, real_label = get_synthetic_sample(dataset)
        predicted_label = classify_sample(model, sample, real_label)
        real_labels.append(real_label[0])
        predicted_labels.append(predicted_label)
        print_and_log(f"Sample {i+1}/{n} - Real: {real_label[0]}, Predicted: {predicted_label}\n")

    real_labels = np.array(real_labels)
    predicted_labels = np.array(predicted_labels)

    # Compute metrics
    accuracy = accuracy_score(real_labels, predicted_labels)
    precision = precision_score(real_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(real_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(real_labels, predicted_labels, average='weighted', zero_division=0)
    confusion = confusion_matrix(real_labels, predicted_labels, labels=list(range(1, 8)))

    print_and_log(f"\n--- Deployment Model Test Metrics ---\n")
    print_and_log(f"\nAccuracy:  {accuracy:.4f}")
    print_and_log(f"Precision: {precision:.4f}")
    print_and_log(f"Recall:    {recall:.4f}")
    print_and_log(f"F1 Score:  {f1:.4f}")

    # Print confusion matrix with labels on axes
    print_and_log("Confusion Matrix:\n", path="log/validation_metrics.log")
    print_and_log("         PREDICTED", path="log/validation_metrics.log")
    print_and_log("      ", end="", path="log/validation_metrics.log")
    for label in range(1, 8):
        print_and_log(f"{label:>5}", end="", path="log/validation_metrics.log")
    print_and_log(path="log/validation_metrics.log")
    real_label = "REAL"
    for row_idx, row in enumerate(confusion):
        letter = real_label[row_idx] if row_idx < len(real_label) else " "
        print_and_log(f"  {letter}  {row_idx+1:>2} ", end="", path="log/validation_metrics.log")
        for val in row:
            print_and_log(f"{val:>5}", end="", path="log/validation_metrics.log")
        print_and_log(path="log/validation_metrics.log")

# --- EXTRA TESTS ---

def test_device_specific_model(dataset, features, labels):
    """Test the deployment model with a synthetic sample."""

    real_labels = []
    predicted_labels = []

    device1_features = features[labels[:, 2] == 1]
    device1_labels = labels[labels[:, 2] == 1]

    device2_features = features[labels[:, 2] == 2]
    device2_labels = labels[labels[:, 2] == 2]

    device3_features = features[labels[:, 2] == 3]
    device3_labels = labels[labels[:, 2] == 3]

    device4_features = features[labels[:, 2] == 4]
    device4_labels = labels[labels[:, 2] == 4]

    device5_features = features[labels[:, 2] == 5]
    device5_labels = labels[labels[:, 2] == 5]

    model1 = deployment_model(device1_features, device1_labels, k=19)
    model2 = deployment_model(device2_features, device2_labels, k=19)
    model3 = deployment_model(device3_features, device3_labels, k=19)
    model4 = deployment_model(device4_features, device4_labels, k=19)
    model5 = deployment_model(device5_features, device5_labels, k=19)

    for i in range(100):
        sample, real_label = get_synthetic_sample(dataset)

        if real_label[2] == 1:
            model = model1
        elif real_label[2] == 2:
            model = model2
        elif real_label[2] == 3:
            model = model3
        elif real_label[2] == 4:
            model = model4
        else:
            model = model5

        predicted_label = classify_sample(model, sample, real_label)
        real_labels.append(real_label[0])
        predicted_labels.append(predicted_label)
        print_and_log(f"Sample {i+1}/100 - Real: {real_label[0]}, Predicted: {predicted_label}\n")

    real_labels = np.array(real_labels)
    predicted_labels = np.array(predicted_labels)

    # Compute metrics
    accuracy = accuracy_score(real_labels, predicted_labels)
    precision = precision_score(real_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(real_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(real_labels, predicted_labels, average='weighted', zero_division=0)
    confusion = confusion_matrix(real_labels, predicted_labels, labels=list(range(1, 8)))

    print_and_log(f"\n--- Deployment Model Test Metrics ---\n")
    print_and_log(f"\nAccuracy:  {accuracy:.4f}")
    print_and_log(f"Precision: {precision:.4f}")
    print_and_log(f"Recall:    {recall:.4f}")
    print_and_log(f"F1 Score:  {f1:.4f}")

    # Print confusion matrix with labels on axes
    print_and_log("Confusion Matrix:\n", path="log/validation_metrics.log")
    print_and_log("         PREDICTED", path="log/validation_metrics.log")
    print_and_log("      ", end="", path="log/validation_metrics.log")
    for label in range(1, 8):
        print_and_log(f"{label:>5}", end="", path="log/validation_metrics.log")
    print_and_log(path="log/validation_metrics.log")
    real_label = "REAL"
    for row_idx, row in enumerate(confusion):
        letter = real_label[row_idx] if row_idx < len(real_label) else " "
        print_and_log(f"  {letter}  {row_idx+1:>2} ", end="", path="log/validation_metrics.log")
        for val in row:
            print_and_log(f"{val:>5}", end="", path="log/validation_metrics.log")
        print_and_log(path="log/validation_metrics.log")