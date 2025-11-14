from load import *
from outliers import *
from features import *

if __name__ == "__main__":

    data = load_data_csv()
    acc_modules = variable_module(data, "Acceleration")
    mag_modules = variable_module(data, "Magnetic Field")
    gyro_modules = variable_module(data, "Angular Velocity")

    features_matrix, labels, feature_names = extract_features(
        data, acc_modules, mag_modules, gyro_modules,
        fs=51.5, window_duration=5.0, overlap_ratio=0.5)

    n_components = 36
    pca_matrix, explained_variance_ratio = pca(features_matrix, n_components)

    pca_analysis(explained_variance_ratio)

    fisher(features_matrix, labels, feature_names)
    relief(features_matrix, labels, feature_names=feature_names, n_neighbors=100)