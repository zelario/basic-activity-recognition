from load import *
from outliers import *
from features import *

import numpy as np

if __name__ == "__main__":

    try:
        data = np.load("data/labels.npy")
        features_matrix = np.load("data/features_matrix.npy")
        pca_matrix = np.load("data/pca_matrix.npy")
        scores_matrix = np.load("data/scores_matrix.npy")
        labels_matrix = np.load("data/labels_matrix.npy")
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
        np.save("data/features_matrix.npy", features_matrix)
        np.save("data/pca_matrix.npy", pca_matrix)
        np.save("data/labels_matrix.npy", labels_matrix)
        np.save("data/scores_matrix.npy", scores_matrix)