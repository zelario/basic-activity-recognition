from augmentation import discard_activities
from features import *
from outliers import *
from embeddings import *
from models import *
import numpy as np

import numpy as np

def get_random_sample(dataset):

    # Only consider activities 1 to 7
    activities_mask = (dataset[:, 11] >= 1) & (dataset[:, 11] <= 7)
    dataset = dataset[activities_mask]

    # Randomly choose activity, participant, device
    activities = np.unique(dataset[:, 11])
    participants = np.unique(dataset[:, 12])
    devices = np.unique(dataset[:, 0])

    for _ in range(100):  # Try up to 100 random combinations
        activity_label = np.random.choice(activities)
        participant_id = np.random.choice(participants)
        device_id = np.random.choice(devices)

        mask = (
            (dataset[:, 12] == participant_id) &
            (dataset[:, 0] == device_id) &
            (dataset[:, 11] == activity_label)
        )

        group = dataset[mask]
        if group.shape[0] >= 256:
            shuffled_group = group.copy()
            np.random.shuffle(shuffled_group)
            sample = shuffled_group[:256, 1:10]
            return sample
        
    return None

def _format_sample(sample_dataset):
    expanded = np.zeros((256, 13))
    expanded[:, 0] = 0  # device id
    expanded[:, 1:4] = sample_dataset[:, 0:3]   # acc x, y, z
    expanded[:, 4:7] = sample_dataset[:, 3:6]   # gyr x, y, z
    expanded[:, 7:10] = sample_dataset[:, 6:9]  # mag x, y, z
    expanded[:, 10] = np.linspace(0, 4999, 256) # timestamps for 5s window
    expanded[:, 11] = 1  # activity label
    expanded[:, 12] = 1  # participant id
    return expanded 

def my_model(sample_dataset):
    """Deploy my model to make predictions on a sample dataset."""

    print_and_log("\n--- Deployment: Making predictions on sample dataset ---\n")

    try:
        model_features = np.load("npy/features.npy", allow_pickle=True)
        model_labels = np.load("npy/feature_labels.npy", allow_pickle=True)
        model_features, model_labels = discard_activities(features=model_features, labels=model_labels)
    except FileNotFoundError:
        return
    
    print_and_log("Sample dataset shape:", sample_dataset.shape)
    sample_dataset = _format_sample(sample_dataset)
    print_and_log("\nFormatted sample dataset shape:", sample_dataset.shape)

    # Extract features from sample 
    sample_variables_modules = compute_modules(sample_dataset)
    sample_features, sample_labels = extract_features(sample_dataset, sample_variables_modules)

    print_and_log("\nModel features shape:", model_features.shape)
    print_and_log("\nSample features shape:", sample_features.shape)
    
    # Select top 15 features using Relief for both model and sample features
    normalized_model_features, model_features_mean, model_features_std = zscore_normalization(model_features, return_params=True)
    top_15_model_features_indices = relief(normalized_model_features, model_labels, top_n=15, print_output=False)
    model_features_relief = normalized_model_features[:, top_15_model_features_indices]

    normalized_sample_features = zscore_normalization(sample_features, mean=model_features_mean, std=model_features_std)
    sample_features_relief = normalized_sample_features[:, top_15_model_features_indices]
    
    # Make prediction using knn with k=19
    knn_model = sklearn_knn_classifier(model_features_relief, model_labels, k=19)
    predicted_labels = knn_model.predict(sample_features_relief)

    print(predicted_labels)

    pass