from augmentation import discard_activities
from features import *
from outliers import *
from embeddings import *
from models import *
import numpy as np

import numpy as np

def _get_random_sample(dataset):

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
        return _get_random_sample(dataset)

def _format_sample(sample_dataset):
    formated_sample = np.zeros((256, 13))
    formated_sample[:, 0] = 0  # device id
    formated_sample[:, 1:4] = sample_dataset[:, 0:3]   # acc x, y, z
    formated_sample[:, 4:7] = sample_dataset[:, 3:6]   # gyr x, y, z
    formated_sample[:, 7:10] = sample_dataset[:, 6:9]  # mag x, y, z
    formated_sample[:, 10] = 0  # timestamp
    formated_sample[:, 11] = 1  # activity label
    formated_sample[:, 12] = 1  # participant id
    return formated_sample

def my_model(sample_dataset):
    """Deploy my model to make predictions on a sample dataset."""

    print_and_log("\n--- Deployment: Making predictions on sample dataset ---\n")

    try:
        model_features = np.load("npy/features.npy", allow_pickle=True)
        model_labels = np.load("npy/feature_labels.npy", allow_pickle=True)
        model_features, model_labels = discard_activities(features=model_features, labels=model_labels)
    except FileNotFoundError:
        return
    
    # Format sample dataset to match every other function
    sample_dataset = _format_sample(sample_dataset)

    # Extract features from sample 
    sample_variables_modules = compute_modules(sample_dataset)
    sample_features, sample_labels = extract_features(sample_dataset, sample_variables_modules, overlap=0.0)
    
    # Select top 15 features using Relief for both model and sample features
    normalized_model_features, model_features_mean, model_features_std = zscore_normalization(model_features, return_parameters=True)
    top_15_model_features_indices = relief(normalized_model_features, model_labels, top_n=15, print_output=False)
    model_features_relief = normalized_model_features[:, top_15_model_features_indices]

    normalized_sample_features = zscore_normalization(sample_features, mean_values=model_features_mean, std_values=model_features_std)
    sample_features_relief = normalized_sample_features[:, top_15_model_features_indices]
    
    # Make prediction using knn with k=19
    knn_model = sklearn_knn_classifier(model_features_relief, model_labels, k=19)
    predicted_labels = knn_model.predict(sample_features_relief)

    print_and_log(f"Predicted label for the sample dataset: {predicted_labels}\n")

    pass