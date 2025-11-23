"""
Feature extraction, statistical tests and dimensionality reduction utilities.
File used for full exercise 4.

This module contains functions used in the project for:
- running normality and group significance tests across activities,
- creating sliding windows from raw sensor data (both index- and timestamp-based),
- extracting time-domain and simple spectral features from windows,
- normalizing feature matrices (z-score),
- performing PCA and visualizing explained variance,
- ranking features using Fisher score and a ReliefF procedures.

Column conventions expected in `dataset` arrays used by windowing functions:
- Column 0: device id 
- Column 10: timestamp in milliseconds (for timestamp-based windowing)
- Column 11: activity id 
- Column 12: participant id 
"""

from log import print_and_log
from outliers import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, f_oneway, kruskal
from sklearn.decomposition import PCA

FEATURES_NAMES = [
    "Acceleration Mean", "Acceleration Std", "Acceleration Median", "Acceleration Variance", "Acceleration RMS", "Acceleration Average Deviation", "Acceleration Skewness", 
    "Acceleration Kurtosis", "Acceleration IQR", "Acceleration Zero Crossing Rate", "Acceleration Mean Crossing Rate", "Acceleration Spectral Entropy",
    "Gyroscope Mean", "Gyroscope Std", "Gyroscope Median", "Gyroscope Variance", "Gyroscope RMS", "Gyroscope Average Deviation", "Gyroscope Skewness", 
    "Gyroscope Kurtosis", "Gyroscope IQR", "Gyroscope Zero Crossing Rate", "Gyroscope Mean Crossing Rate", "Gyroscope Spectral Entropy",
    "Magnetometer Mean", "Magnetometer Std", "Magnetometer Median", "Magnetometer Variance", "Magnetometer RMS", "Magnetometer Average Deviation", "Magnetometer Skewness", 
    "Magnetometer Kurtosis", "Magnetometer IQR", "Magnetometer Zero Crossing Rate", "Magnetometer Mean Crossing Rate", "Magnetometer Spectral Entropy"
]

# --- Exercise 4.1: Statistical Tests ---

def normality_and_significance(dataset, variables_modules, alpha=0.05):
    """
    Run normality checks per activity and perform a group-level significance test.

    For each variable in `variables_modules`, splits by activity and performs Kolmogorov-Smirnov normality test on z-scored samples.
    If all groups pass normality, uses one-way ANOVA; otherwise, uses Kruskal-Wallis test.

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw dataset matrix. Column 11 must contain activity id.
    variables_modules : iterable of (str, np.ndarray)
        Iterable of (name, modules) pairs, modules aligned with dataset rows.
    alpha : float, optional
        Significance threshold (default=0.05).
    """

    print_and_log("--- Normality and Significance Tests ---")
    for name, modules in zip(VARIABLE_NAMES, variables_modules):
        print_and_log(f"\n--- Variable: {name} ---\n")

        normality_results = {}
        activity_groups = []

        for activity in range(1, 17):
            activity_modules = modules[dataset[:, 11] == activity]
            if len(activity_modules) < 2:
                print_and_log(f"Activity {activity}: Not enough data")
                continue

            z_values = (activity_modules - np.mean(activity_modules)) / np.std(activity_modules)
            stat, p_value = kstest(z_values, 'norm')
            normality_results[activity] = p_value
            activity_groups.append(activity_modules)

            mean_val = np.mean(activity_modules)
            normal_str = "Normal" if p_value > alpha else "Not normal"
            print_and_log(f"Activity {activity}: mean = {mean_val:.4f}, p = {p_value:.4f} -> {normal_str}")

        if not activity_groups:
            print_and_log("No sufficient data for any activity.")
            continue

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

def _sliding_windows(dataset, window_duration=5.0, overlap=0.5):
    """
    Create sliding windows using timestamps, enforcing label/device/participant continuity.

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw data matrix with timestamp and label columns.
    window_duration : float, optional
        Window length in seconds (default=5.0).
    overlap : float, optional
        Fractional overlap between consecutive windows (default=0.5).

    Returns
    -------
    windows : list of tuples
        Each tuple is (start_idx, end_idx, activity_id, participant_id).
    """

    activity_ids = np.asarray(dataset[:, 11]).astype(int)
    device_ids = np.asarray(dataset[:, 0]).astype(int)
    participant_ids = np.asarray(dataset[:, 12]).astype(int)
    timestamps = np.asarray(dataset[:, 10]).astype(float)

    window_duration_ms = int(window_duration * 1000)
    n = len(dataset)
    windows = []
    start = 0
    while start < n:
        activity_ref = activity_ids[start]
        device_ref = device_ids[start]
        participant_ref = participant_ids[start]
        t_start = timestamps[start]

        end = start
        while (end < n and
               activity_ids[end] == activity_ref and
               device_ids[end] == device_ref and
               participant_ids[end] == participant_ref and
               timestamps[end] - t_start < window_duration_ms):
            end += 1

        if end - start > 1:
            windows.append((start, end, activity_ref, participant_ref))

        # Overlap
        if overlap > 0:
            step = int((end - start) * (1.0 - overlap))
            start += max(1, step)
        else:
            start = end

    return windows

def _extract_window_features(signal):
    """
    Compute time-domain and simple spectral features for a 1-D signal window.

    Features (in order): mean, std, median, variance, rms, average deviation, skewness, kurtosis, IQR, zero crossing rate, mean crossing rate, spectral entropy.

    Parameters
    ----------
    signal : np.ndarray
        1-D numeric vector of signal samples for a single window.

    Returns
    -------
    feature_values : list of float
        List of 12 features extracted from signal.
    """

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

def extract_features(dataset, variables_modules, window_duration=5.0, overlap_ratio=0.5):
    """
    Extract features per window for acceleration, gyroscope, and magnetometer modules.

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw dataset aligned with variable arrays (timestamps & labels).
    variables_modules : iterable of (str, np.ndarray)
        Iterable of (name, modules) pairs, modules aligned with dataset rows.
    window_duration : float, optional
        Window duration in seconds (default=5.0).
    overlap_ratio : float, optional
        Fractional overlap between windows (default=0.5).

    Returns
    -------
    features : np.ndarray, shape (n_windows, n_features)
        Z-score normalized feature matrix, one row per valid window.
    labels : np.ndarray, shape (n_windows, 2)
        Integer matrix: (activity_label, participant_id) for each window.
    """

    windows = _sliding_windows(dataset, window_duration, overlap_ratio)

    features = []
    labels = []

    acceleration_modules = variables_modules[0]
    gyroscope_modules = variables_modules[1]
    magnetic_modules = variables_modules[2]

    for (start_idx, end_idx, activity_label, participant) in windows:
        acc_window = acceleration_modules[start_idx:end_idx]
        gyro_window = gyroscope_modules[start_idx:end_idx]
        mag_window = magnetic_modules[start_idx:end_idx]

        if acc_window.size == 0 or gyro_window.size == 0 or mag_window.size == 0:
            continue

        acc_features = _extract_window_features(acc_window)
        gyro_features = _extract_window_features(gyro_window)
        mag_features = _extract_window_features(mag_window)

        combined_features = acc_features + gyro_features + mag_features

        participant = int(dataset[start_idx, 12]) 
        labels.append((activity_label, participant))
        features.append(combined_features)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels

def zscore_normalization(features, mean_values=None, std_values=None, return_parameters=False):
    """
    Column-wise z-score normalization of feature matrix.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        Numeric feature matrix.
    mean_values : np.ndarray or None, optional
        Precomputed mean values for each feature (default=None).
    std_values : np.ndarray or None, optional
        Precomputed std deviation values for each feature (default=None).
    return_parameters : bool, optional
        If True, also return computed mean and std values (default=False).

    Returns
    -------
    features : np.ndarray, shape (n_samples, n_features)
        Z-score normalized feature matrix.
    """

    if mean_values is not None and std_values is not None:
        features = features.astype(float, copy=True)
        features = (features - mean_values) / std_values
    else:
        mean_values = np.mean(features, axis=0)
        std_values = np.std(features, axis=0, ddof=1)
        std_values[std_values == 0] = 1.0  
        features = (features - mean_values) / std_values

    if return_parameters:
        return features, mean_values, std_values
    else:
        return features

# --- Exercise 4.3: PCA ---

def compute_pca(dataset, n_components=36, pca_object=None):
    """Perform principal component analysis (PCA) and return projected data and explained variance ratio.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        Input feature matrix.
    n_components : int or None, optional
        Number of principal components to compute. If None, uses min(features.shape).
    pca_object : sklearn.decomposition.PCA or None
        Pre-fitted PCA object to use for transformation (if transform=True).
    transform : bool, optional
        If False, fit PCA on features; if True, use pca_object to transform.

    Returns
    -------
    transformed : np.ndarray
        Transformed data (scores) for each sample.
    explained_variance_ratio : np.ndarray or None
        Variance ratio explained by each component (None if transform=True)."""
    
    if pca_object is None:
        n_components = n_components or min(dataset.shape)
        pca_object = PCA(n_components=n_components)
        pca = pca_object.fit_transform(dataset)
        explained_variance_ratio = pca_object.explained_variance_ratio_
        return pca, explained_variance_ratio, pca_object
    else:
        pca = pca_object.transform(dataset)
        return pca

# --- Exercise 4.4: PCA Analysis  ---

def analyse_pca(explained_variance_ratio):
    """
    Plot explained variance ratio and cumulative variance from PCA.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Per-component explained variance ratio from fitted PCA.
    """

    cumulative = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio,
            alpha=0.6, label="Explained Variance Ratio")
    plt.plot(range(1, len(cumulative) + 1), cumulative,
             color='red', marker='o', label="Cumulative Variance")

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
    """
    Compute Fisher scores for features and print the top-ranked feature names.

    Fisher score is the ratio of between-class to within-class variance for each feature.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    labels : np.ndarray, shape (n_samples, 2)
        First column is activity label.

    Returns
    -------
    None
        Prints top 10 feature names and scores.
    """

    labels = np.array(labels)
    n_features = features.shape[1]
    activity_labels = labels[:, 0]
    classes = np.unique(activity_labels)
    overall_mean = np.nanmean(features, axis=0)

    nominator = np.zeros(n_features)
    denominator = np.zeros(n_features)

    for cls in classes:
        cls_mask = activity_labels == cls
        cls_data = features[cls_mask]
        cls_count = np.sum(~np.isnan(cls_data), axis=0)
        cls_mean = np.nanmean(cls_data, axis=0)

        diff = np.where(np.isnan(cls_mean - overall_mean), 0, cls_mean - overall_mean)
        nominator += cls_count * (diff ** 2)
        denominator += np.nansum((cls_data - cls_mean) ** 2, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.where(denominator > 0, nominator / denominator, 0)

    sorted_idx = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_idx]

    print_and_log("\n========== Fisher Score ==========")
    for i in range(min(10, n_features)):
        name = FEATURES_NAMES[sorted_idx[i]] if FEATURES_NAMES else f"feature_{sorted_idx[i]}"
        print_and_log(f"{i+1:02d}. {name:>20s}  |  score = {sorted_scores[i]:.4f}")


def relief(dataset, labels, n_neighbors=50, n_samples=500, top_n=10, print_output=True):
    """
    Approximate ReliefF feature ranking using random sampling.

    For each of n_samples randomly chosen instances, computes average distance to n_neighbors hits (same class) and misses (different class).
    The difference (miss - hit) is accumulated for each feature; higher scores indicate stronger discrimination.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    labels : np.ndarray, shape (n_samples, 2)
        Class labels; first column is class label.
    n_neighbors : int, optional
        Number of neighbors for hit/miss estimation (default=50).
    n_samples : int, optional
        Number of random samples to draw (default=500).
    top_n : int, optional
        Number of top features to return (default=10).

    Returns
    -------
    top_indices : np.ndarray
        Indices of top n features sorted by ReliefF score.
    """
    
    n_total = dataset.shape[0]
    n_features = dataset.shape[1]
    sample_idx = np.random.choice(n_total, min(n_samples, n_total), replace=False)
    X_sample = dataset[sample_idx]
    y_sample = np.array(labels)[sample_idx, 0]

    nominator = np.zeros(n_features)

    for i, x in enumerate(X_sample):
        same_mask = y_sample == y_sample[i]
        diff_mask = y_sample != y_sample[i]

        same_idx = np.random.choice(np.where(same_mask)[0], min(n_neighbors, np.sum(same_mask)), replace=True)
        diff_idx = np.random.choice(np.where(diff_mask)[0], min(n_neighbors, np.sum(diff_mask)), replace=True)

        hit_diff = np.mean(np.abs(X_sample[same_idx] - x), axis=0)
        miss_diff = np.mean(np.abs(X_sample[diff_idx] - x), axis=0)

        nominator += miss_diff - hit_diff

    scores = nominator / n_samples
    sorted_idx = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_idx]

    top_indices = sorted_idx[:top_n]

    if print_output:
        print_and_log("\n========== ReliefF ==========")
        for rank, idx in enumerate(top_indices):
            name = FEATURES_NAMES[idx]
            print_and_log(f"{rank+1:02d}. {name:>20s}  |  score = {sorted_scores[rank]:.4f} (index {idx})")

    return top_indices