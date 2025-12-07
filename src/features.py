"""
Feature extraction, statistical tests and dimensionality reduction utilities.
File used for modulo B exercise 4.

This file contains functions used in the project for:
- running normality and group significance tests across activities,
- creating windows from raw sensor data,
- extracting features from windows,
- normalizing matrices (z-score),
- performing PCA and visualizing explained variance,
- ranking features using Fisher score and a ReliefF.

Column conventions expected in `dataset` arrays used by windowing functions:
- Column 0: device id 
- Column 10: timestamp
- Column 11: activity id 
- Column 12: participant id 
"""

from log import *
from outliers import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, f_oneway, kruskal
from sklearn.decomposition import PCA

AXES = ["X", "Y", "Z"]
SENSORS = ["Acceleration", "Gyroscope", "Magnetometer"]
FEATURES = [
    "Mean", "Std", "Median", "Variance", "RMS", "Average Deviation", "Skewness",
    "Kurtosis", "IQR", "Zero Crossing Rate", "Mean Crossing Rate", "Spectral Entropy"
]
FEATURES_NAMES = [
    f"{sensor} {axis} {feature}"
    for sensor in SENSORS
    for axis in AXES
    for feature in FEATURES
]

# --- Exercise 4.1: Statistical Tests ---

def normality_and_significance(dataset, variables_modules, alpha=0.05):
    """Run normality checks per activity and perform a significance test.

    For each variable, splits by activity and performs Kolmogorov-Smirnov normality tests.
    If all groups pass normality, uses one-way ANOVA, else uses Kruskal-Wallis test.

    Parameters
    ----------
    dataset : matrix, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : matrix, shape (n_variables, n_samples)
        Matrix of variables, each row aligned with dataset rows."""

    print_and_log("--- Normality and Significance Tests ---")

    # For each variable
    for name, modules in zip(VARIABLE_NAMES, variables_modules):
        print_and_log(f"\n--- Variable: {name} ---\n")

        normality_results = {}
        activity_groups = []

        # For each activity
        for activity in range(1, 17):

            # Filter data for current activity
            activity_modules = modules[dataset[:, 11] == activity]

            # Perform normality test
            z_values = (activity_modules - np.mean(activity_modules)) / np.std(activity_modules)
            stat, p_value = kstest(z_values, 'norm')
            normality_results[activity] = p_value
            activity_groups.append(activity_modules)

            mean_val = np.mean(activity_modules)
            normal_str = "Normal" if p_value > alpha else "Not normal"
            print_and_log(f"Activity {activity}: mean = {mean_val:.4f}, p = {p_value:.4f} -> {normal_str}")

        # If all normal, use ANOVA, else Kruskal-Wallis
        if all(p > alpha for p in normality_results.values()):
            test_stat, test_p = f_oneway(*activity_groups)
            test_name = "ANOVA"
        else:
            test_stat, test_p = kruskal(*activity_groups)
            test_name = "Kruskal-Wallis"

        print_and_log(f"\nTest used: {test_name}")
        print_and_log(f"Statistic = {test_stat:.4f} | p-value = {test_p:.4f}")
        if test_p < alpha:
            print_and_log("Result: Significant differences between activities")
        else:
            print_and_log("Result: No significant differences between activities")

def sliding_windows(dataset, window_duration, overlap):
    """Create windows, enforcing label/device/participant continuity.

    Parameters
    ----------
    dataset : matrix, shape (n_samples, 13)
        Raw data matrix with timestamp and label columns.
    window_duration : float, optional
        Duration of each window in seconds (default=5.0).
    overlap : float, optional
        Overlap ratio between consecutive windows (default=0.5).

    Returns
    -------
    windows : list of tuples
        Each tuple is (start_idx, end_idx, activity_id, participant_id, device_id)."""

    # Extract id and timestamp columns
    activity_ids = np.asarray(dataset[:, 11]).astype(int)
    device_ids = np.asarray(dataset[:, 0]).astype(int)
    participant_ids = np.asarray(dataset[:, 12]).astype(int)
    timestamps = np.asarray(dataset[:, 10]).astype(float)

    window_duration_ms = int(window_duration * 1000)
    dataset_length = len(dataset)
    windows = []
    start = 0

    minimum_window_size = 20

    while start < dataset_length:

        activity = activity_ids[start]
        device = device_ids[start]
        participant = participant_ids[start]

        # Find window end
        end = start
        while (end < dataset_length and
               activity_ids[end] == activity and
               device_ids[end] == device and
               participant_ids[end] == participant and
               timestamps[end] - timestamps[start] < window_duration_ms):
            end += 1

        if end - start >= minimum_window_size:
            windows.append((start, end, activity, participant, device))

        # Overlap
        if overlap > 0:
            step = int((end - start) * (1.0 - overlap))
            start += max(1, step)
        else:
            start = end

    return windows

def extract_features(signal):
    """Extract features for a window.

    Features (in order): mean, std, median, variance, rms, average deviation, skewness, 
    kurtosis, IQR, zero crossing rate, mean crossing rate, spectral entropy.

    Parameters
    ----------
    signal : array, shape (n_samples)
        Vector of signal samples for a single window.

    Returns
    -------
    feature_values : list of float
        List of 12 features extracted from signal."""

    mean_value = np.mean(signal)

    std_value = np.std(signal, ddof=1) if signal.size > 1 else 0.0

    median_value = np.median(signal)

    variance_value = np.var(signal, ddof=1) if signal.size > 1 else 0.0

    rms_value = np.sqrt(np.mean(signal ** 2))

    average_deviation_value = np.mean(np.abs(signal - mean_value))

    skewness_value = (np.mean((signal - mean_value) ** 3) / (std_value ** 3)) if std_value > 0 else 0.0

    kurtosis_value = (np.mean((signal - mean_value) ** 4) / (std_value ** 4)) - 3 if std_value > 0 else 0.0

    iqr_value = np.percentile(signal, 75) - np.percentile(signal, 25)

    zero_crossings = np.sum(np.sign(signal[:-1]) * np.sign(signal[1:]) < 0) if signal.size > 1 else 0

    mean_crossings = np.sum((signal[:-1] - mean_value) * (signal[1:] - mean_value) < 0) if signal.size > 1 else 0

    fft_values = np.fft.rfft(signal)
    power = np.abs(fft_values) ** 2
    power /= np.sum(power) + 1e-12
    spectral_entropy = -np.sum(power * np.log(power + 1e-12))

    feature_values = [mean_value, std_value, median_value, variance_value, rms_value, 
                      average_deviation_value, skewness_value, kurtosis_value, iqr_value,
                      zero_crossings, mean_crossings, spectral_entropy]

    return feature_values

def compute_features(dataset, window_duration=5.0, overlap=0.5, reload=True):
    """Extract features per window for acceleration, gyroscope, and magnetometer modules.

    Parameters
    ----------
    dataset : matrix, shape (n_samples, 13)
        Raw dataset aligned with variable arrays.
    variables_modules : matrix, shape (3, n_samples)
        Modules aligned with dataset rows.
    window_duration : float, optional
        Duration of each window in seconds (default=5.0).
    overlap : float, optional
        Overlap ratio between consecutive windows (default=0.5).

    Returns
    -------
    features : matrix, shape (n_windows, n_features)
        Feature matrix, one row per valid window.
    labels : matrix, shape (n_windows, 3)
        Integer matrix: (activity_label, participant_id, device_id) for each window."""
    
    if reload:
        try:
            features = np.load("cache/features.npy", allow_pickle=True)
            labels = np.load("cache/features_labels.npy", allow_pickle=True)
            return features, labels
        except FileNotFoundError:
            pass 

    # Compute features if not reloading or files not found
    windows = sliding_windows(dataset, window_duration, overlap)

    features = []
    labels = []

    # For each window, extract features for each axis
    for (start_idx, end_idx, activity, participant, device) in windows:
        window_data = dataset[start_idx:end_idx]

        # Acceleration x, y, z
        acc_x = window_data[:, 1]
        acc_y = window_data[:, 2]
        acc_z = window_data[:, 3]

        # Gyroscope x, y, z
        gyro_x = window_data[:, 4]
        gyro_y = window_data[:, 5]
        gyro_z = window_data[:, 6]

        # Magnetometer x, y, z
        mag_x = window_data[:, 7]
        mag_y = window_data[:, 8]
        mag_z = window_data[:, 9]

        # Extract features for each axis
        acc_x_features = extract_features(acc_x)
        acc_y_features = extract_features(acc_y)
        acc_z_features = extract_features(acc_z)
        gyro_x_features = extract_features(gyro_x)
        gyro_y_features = extract_features(gyro_y)
        gyro_z_features = extract_features(gyro_z)
        mag_x_features = extract_features(mag_x)
        mag_y_features = extract_features(mag_y)
        mag_z_features = extract_features(mag_z)

        # Combine all features (order: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z)
        combined_features = (
            acc_x_features + acc_y_features + acc_z_features +
            gyro_x_features + gyro_y_features + gyro_z_features +
            mag_x_features + mag_y_features + mag_z_features
        )

        # Append labels and features
        participant = int(dataset[start_idx, 12])
        labels.append((activity, participant, device))
        features.append(combined_features)

    features = np.array(features)
    labels = np.array(labels)

    if reload:
        np.save("cache/features.npy", features)
        np.save("cache/features_labels.npy", labels)

    return features, labels

def zscore_normalization(features, mean_values=None, std_values=None, return_parameters=False):
    """Z-score normalization of feature matrix.

    Parameters
    ----------
    features : matrix, shape (n_samples, n_features)
        Numeric feature matrix.
    mean_values : array, shape (n_features), optional
        Pre computed mean values for each feature (default=None).
    std_values : array, shape (n_features), optional
        Pre computed standard deviation values for each feature (default=None).

    Returns
    -------
    features : matrix, shape (n_samples, n_features)
        Z-score normalized feature matrix."""

    # If mean and std provided, use them, else compute from data
    if mean_values is not None and std_values is not None:
        features = features.astype(float, copy=True)
        features = (features - mean_values) / std_values
    else:
        mean_values = np.mean(features, axis=0)
        std_values = np.std(features, axis=0, ddof=1)
        std_values[std_values == 0] = 1.0  
        features = (features - mean_values) / std_values

    # Return normalized features and optionally parameters
    if return_parameters:
        return features, mean_values, std_values
    else:
        return features

# --- Exercise 4.3: PCA ---

def compute_pca(dataset, pca_object=None):
    """Perform PCA and return projected data and explained variance ratio.

    Parameters
    ----------
    features : matrix, shape (n_samples, n_features)
        Input feature matrix.
    pca_object : PCA object or None, optional
        Pre-fitted PCA object to use for transformation (default=None).

    Returns
    -------
    transformed : np.ndarray
        Transformed data (scores) for each sample.
    explained_variance_ratio : np.ndarray or None
        Variance ratio explained by each component (None if transform=True)."""
    
    # If no PCA object provided, fit new PCA, else transform using existing
    if pca_object is None:
        n_components = min(dataset.shape)
        pca_object = PCA(n_components=n_components)
        pca = pca_object.fit_transform(dataset)
        explained_variance_ratio = pca_object.explained_variance_ratio_
        return pca, explained_variance_ratio, pca_object
    else:
        pca = pca_object.transform(dataset)
        return pca

# --- Exercise 4.4: PCA Analysis  ---

def analyse_pca(explained_variance_ratio):
    """Plot explained variance ratio and cumulative variance from PCA.

    Parameters
    ----------
    explained_variance_ratio : array, shape (n_components)
        Per-component explained variance ratio from fitted PCA."""

    # Compute cumulative variance
    cumulative = np.cumsum(explained_variance_ratio)

    # Plot explained variance and cumulative variance
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6, label="Explained Variance Ratio")
    plt.plot(range(1, len(cumulative) + 1), cumulative, color='red', marker='o', label="Cumulative Variance")

    plt.axhline(y=0.75, color='green', linestyle='--', label=f"{int(0.75*100)}% Variance")

    n_components = np.argmax(cumulative >= 0.75) + 1
    plt.axvline(x=n_components, color='purple', linestyle='--', label=f"{n_components} Components")

    plt.title("PCA Analysis")
    plt.xlabel("Component Number")
    plt.ylabel("Variance Explained")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Exercise 4.5: Fisher Scores and ReliefF ---

def fisher(features, labels):
    """Compute Fisher scores for features and print the top-ranked feature names.

    Fisher score is the ratio of between-class to within-class variance for each feature.

    Parameters
    ----------
    features : matrix, shape (n_samples, n_features)
        Feature matrix.
    labels : matrix, shape (n_samples, 3)
        First column is activity label."""

    labels = np.array(labels)
    n_features = features.shape[1]
    activity_labels = labels[:, 0]
    activity_ids = np.unique(activity_labels)
    overall_mean = np.nanmean(features, axis=0)

    nominator = np.zeros(n_features)
    denominator = np.zeros(n_features)

    # For each activity, compute contributions to nominator and denominator
    for activity in activity_ids:

        activity_mask = activity_labels == activity
        activity_data = features[activity_mask]
        activity_count = np.sum(~np.isnan(activity_data), axis=0)
        activity_mean = np.nanmean(activity_data, axis=0)

        diff = np.where(np.isnan(activity_mean - overall_mean), 0, activity_mean - overall_mean)
        nominator += activity_count * (diff ** 2)
        denominator += np.nansum((activity_data - activity_mean) ** 2, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.where(denominator > 0, nominator / denominator, 0)

    # Sort features by Fisher score
    sorted_idx = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_idx]

    print_and_log("\n========== Fisher Score ==========")
    for i in range(min(10, n_features)):
        name = FEATURES_NAMES[sorted_idx[i]] if FEATURES_NAMES else f"feature_{sorted_idx[i]}"
        print_and_log(f"{i+1:02d}. {name:>20s}  |  score = {sorted_scores[i]:.4f}")

def relief(dataset, labels, n_samples=500, top_n=10, print_output=True):
    """Classic Relief feature ranking using single nearest hit/miss per sample.
    
    Parameters
    ----------
    dataset : matrix, shape (n_samples, n_features)
        Feature matrix.
    labels : matrix, shape (n_samples, 3)
        First column is activity label.
    n_samples : int, optional
        Number of random samples to use for Relief (default=500).
    top_n : int, optional
        Number of top features to print (default=10).
    print_output : bool, optional
        Whether to print the top features (default=True).

    Returns
    -------
    top_indices : array, shape (top_n)
        Indices of the top_n ranked features."""
    
    n_total = dataset.shape[0]
    n_features = dataset.shape[1]
    sample_indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)
    sample_features = dataset[sample_indices]
    sample_labels = labels[sample_indices, 0]

    nominator = np.zeros(n_features)

    for i, instance in enumerate(sample_features):
        same_mask = sample_labels == sample_labels[i]
        diff_mask = sample_labels != sample_labels[i]

        # Exclude the instance itself for hit
        same_indices = np.where(same_mask)[0]
        same_indices = same_indices[same_indices != i]
        diff_indices = np.where(diff_mask)[0]

        if same_indices.size > 0:
            hit_idx = same_indices[np.argmin(np.linalg.norm(sample_features[same_indices] - instance, axis=1))]
            hit_diff = np.abs(sample_features[hit_idx] - instance)
        else:
            hit_diff = np.zeros(n_features)

        if diff_indices.size > 0:
            miss_idx = diff_indices[np.argmin(np.linalg.norm(sample_features[diff_indices] - instance, axis=1))]
            miss_diff = np.abs(sample_features[miss_idx] - instance)
        else:
            miss_diff = np.zeros(n_features)

        nominator += miss_diff - hit_diff

    scores = nominator / n_samples
    sorted_indexes = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indexes]

    top_indices = sorted_indexes[:top_n]

    if print_output:
        print_and_log("\n========== Relief ==========")
        for rank, idx in enumerate(top_indices):
            name = FEATURES_NAMES[idx]
            print_and_log(f"{rank+1:02d}. {name:>20s}  |  score = {sorted_scores[rank]:.4f} (index {idx})")

    return top_indices