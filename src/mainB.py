from load import *
from outliers import *
from features import *

import numpy as np

if __name__ == "__main__":

    try:
        features_matrix = np.load("data/features_matrix.npy")
        pca_matrix = np.load("data/pca_matrix.npy")
        labels_matrix = np.load("data/labels.npy")
        feature_names = np.load("data/feature_names.npy", allow_pickle=True)
        fisher_features = np.load("data/fisher_features.npy", allow_pickle=True)
        relief_features = np.load("data/relief_features.npy", allow_pickle=True)
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

        np.save("data/features_matrix.npy", features_matrix)
        np.save("data/pca_matrix.npy", pca_matrix)
        np.save("data/labels_matrix.npy", labels_matrix)
        np.save("data/feature_names.npy", feature_names)
        np.save("data/fisher_features.npy", fisher_features)
        np.save("data/relief_features.npy", relief_features)