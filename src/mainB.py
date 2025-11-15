from load import *
from outliers import *
from features import *
from augmentation import *

import numpy as np

if __name__ == "__main__":

    # Load files from first part

    try:
        data = np.load("data/labels.npy")
        features = np.load("data/features.npy")
        pca = np.load("data/pca.npy")
        scores = np.load("data/scores.npy")
        labels = np.load("data/labels.npy")
        print("--- Loaded data from existing .npy files ---")

    except FileNotFoundError:
        data = load_data_csv()
        acc_modules = variable_module(data, "Acceleration")
        mag_modules = variable_module(data, "Magnetic Field")
        gyro_modules = variable_module(data, "Angular Velocity")

        features_matrix, labels_matrix, feature_names = extract_features(
            data, acc_modules, mag_modules, gyro_modules,
            fs=51.5, window_duration=5.0, overlap_ratio=0.5)

        n_components = 36
        pca_matrix, explained_variance_ratio = pca(features_matrix, n_components)

        pca_analysis(explained_variance_ratio)

        fisher_features = fisher(features_matrix, labels_matrix, feature_names)
        relief_features = relief(features_matrix, labels_matrix, feature_names=feature_names, n_neighbors=100)

        scores_matrix = np.array([
            feature_names,
            fisher_features,
            relief_features
        ], dtype=object)

        np.save("data/data.npy", data)
        np.save("data/features.npy", features)
        np.save("data/pca.npy", pca)
        np.save("data/labels.npy", labels)
        np.save("data/scores.npy", scores)

    features, pca, labels = discard_activities(features, pca, labels)