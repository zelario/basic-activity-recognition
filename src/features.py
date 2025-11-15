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

Column conventions expected in `data` arrays used by windowing functions:
- Column 0: device id 
- Column 10: timestamp in milliseconds (for timestamp-based windowing)
- Column 11: activity id 
- Column 12: participant id 
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from scipy.stats import kstest, f_oneway, kruskal
from sklearn.decomposition import PCA

# --- Exercise 4.1: Statistical Tests ---

def normality_and_significance(data, variables_modules, alpha=0.05):
    """Run normality checks per activity and a group-level significance test.

    For each variable provided in `variables_modules` the function splits the
    variable modules by activity (using column 11 of `data`) and performs a
    Kolmogorov-Smirnov test for normality on z-scored samples of each
    activity. If all activity groups pass the normality test (p > `alpha`),
    a one-way ANOVA is used to test for differences between activities; 
    otherwise the non-parametric Kruskal-Wallis test is used.

    Parameters
    ----------
    data : matrix, shape (n_samples, 13)
        Raw dataset matrix. Column 11 must contain the activity id for each sample.
    variables_modules : iterable of (str, array)
        Iterable of pairs (name, modules) where `modules` is a 1-D array of
        values aligned with `data` rows representing the variable to test.
    alpha : float
        Significance threshold for tests (default 0.05)."""

    print("--- Normality and Significance Tests ---")
    for name, modules in variables_modules:
        print(f"\n--- Variable: {name} ---\n")

        normality_results = {}
        activity_groups = []

        for activity in range(1, 17):
            activity_modules = modules[data[:, 11] == activity]
            if len(activity_modules) < 2:
                print(f"Activity {activity}: Not enough data")
                continue

            z_values = (activity_modules - np.mean(activity_modules)) / np.std(activity_modules)
            stat, p_value = kstest(z_values, 'norm')
            normality_results[activity] = p_value
            activity_groups.append(activity_modules)

            mean_val = np.mean(activity_modules)
            normal_str = "Normal" if p_value > alpha else "Not normal"
            print(f"Activity {activity}: mean = {mean_val:.4f}, p = {p_value:.4f} -> {normal_str}")

        if not activity_groups:
            print("No sufficient data for any activity.")
            continue

        if all(p > alpha for p in normality_results.values()):
            test_stat, test_p = f_oneway(*activity_groups)
            test_name = "ANOVA"
        else:
            test_stat, test_p = kruskal(*activity_groups)
            test_name = "Kruskal-Wallis"

        print(f"\nTest used: {test_name}")
        print(f"Statistic = {test_stat:.4f} | p-value = {test_p:.4f}")
        if test_p < alpha:
            print("Result: Significant differences between activities")
        else:
            print("Result: No significant differences between activities")


# --- Exercise 4.2: Feature Extraction ---

def _sliding_windows(data, fs, window_duration=5.0, overlap=0.5):
    """Create index-based sliding windows where label/device/participant are constant.

    Uses numpy's sliding_window_view to generate candidate windows of 
    `window_duration * fs` samples and selects windows in which the
    activity label (column 11), device id (column 0) and participant id
    (column 12) remain constant throughout the window.

    Parameters
    ----------
    data : matrix, shape (n_samples, 13)
        Raw data matrix with columns for device, timestamp, label, participant.
    fs : float
        Sampling frequency (Hz).
    window_duration : float
        Window duration in seconds.
    overlap : float
        Fractional overlap in [0, 1).

    Returns
    -------
    windows : list of tuples
        Each tuple is (start_idx, end_idx, activity_id, device_id, participant_id)."""

    activity_ids = np.asarray(data[:, 11]).astype(int)
    device_ids = np.asarray(data[:, 0]).astype(int)
    participant_ids = np.asarray(data[:, 12]).astype(int)
    window_size = int(round(window_duration * fs))
    step = max(1, int(round(window_size * (1.0 - overlap))))

    activity_windows = sliding_window_view(activity_ids, window_shape=window_size)
    participant_windows = sliding_window_view(participant_ids, window_shape=window_size)
    device_windows = sliding_window_view(device_ids, window_shape=window_size)

    starts = np.arange(0, activity_windows.shape[0], step)
    activity_candidates = activity_windows[starts]
    device_candidates = device_windows[starts]
    participant_candidates = participant_windows[starts]

    mask_activity = np.all(activity_candidates == activity_candidates[:, :1], axis=1)
    mask_device = np.all(device_candidates == device_candidates[:, :1], axis=1)
    mask_participant = np.all(participant_candidates == participant_candidates[:, :1], axis=1)
    valid_mask = mask_activity & mask_device & mask_participant

    valid_starts = starts[valid_mask]

    windows = [
        (int(s), int(s + window_size), int(activity_ids[s]), int(device_ids[s]), int(participant_ids[s]))
        for s in valid_starts
    ]

    return windows

def _sliding_windows_timestamp(data, fs, window_duration=5.0, overlap=0.5):
    """Create sliding windows using timestamps and enforce label/device/participant continuity.

    The function uses the timestamp column to limit window duration. 
    It also ensures all samples in a window share the same activity 
    label (col 11), device id (col 0) and participant id (col 12).

    Parameters
    ----------
    data : matrix, shape (n_samples, 13)
        Raw data matrix with timestamp and label columns.
    fs : float
        Sampling frequency (Hz) - kept for compatibility with other windowing functions.
    window_duration : float
        Window length in seconds.
    overlap : float
        Fractional overlap between consecutive windows.

    Returns
    -------
    windows : list of tuples
        Each tuple is (start_idx, end_idx, activity_id, device_id, participant_id)."""

    activity_ids = np.asarray(data[:, 11]).astype(int)
    device_ids = np.asarray(data[:, 0]).astype(int)
    participant_ids = np.asarray(data[:, 12]).astype(int)
    timestamps = np.asarray(data[:, 10]).astype(float)

    window_duration_ms = int(window_duration * 1000)
    n = len(data)
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
            windows.append((start, end, activity_ref, device_ref, participant_ref))

        # Overlap
        if overlap > 0:
            step = int((end - start) * (1.0 - overlap))
            start += max(1, step)
        else:
            start = end

    return windows

def _extract_window_features(signal):
    """Compute time-domain and simple spectral features for a 1-D signal.

    The returned features (in order) are:
    mean, std (sample), median, variance (sample), rms, average_deviation,
    skewness, kurtosis (excess), iqr, zero_crossing_rate, mean_crossing_rate,
    spectral_entropy.

    Parameters
    ----------
    signal : array
        1-D numeric vector containing signal samples for a single window.

    Returns
    -------
    feature_values : list of float
        A list of 12 numerical features extracted from `signal`."""

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


def _zscore_normalization(features, eps=1e-12):
    """Column-wise z-score normalization.

    Parameters
    ----------
    features : matrix, shape (n_samples, n_features)
        Numeric feature matrix.
    eps : float
        Small threshold to detect near-zero standard deviations.

    Returns
    -------
    features : matrix, shape (n_samples, n_features)
        Z-score normalized feature matrix (float)."""

    features = features.astype(float, copy=True)
    mean_values = np.nanmean(features, axis=0)
    std_values = np.nanstd(features, axis=0, ddof=1)
    std_values = np.where(std_values < eps, 1.0, std_values)
    features = (features - mean_values) / std_values
    return features


def extract_features(data, variables_modules, fs, window_duration=5.0, overlap_ratio=0.5):
    """Extract features per window for acceleration, gyroscope and magnetometer.

    The function windowizes the `data` using `sliding_windows` and
    computes the same set of features for the module signals of the
    three variables. Feature columns are named with prefixes `acc_`,
    `gyro_`, `mag_` followed by the base feature names.

    Parameters
    ----------
    data : matrix, shape (n_samples, 13)
        Raw dataset aligned with the variable arrays (timestamps & labels).
    variables_modules : iterable of (str, array)
        Iterable of pairs (name, modules) where `modules` is a 1-D array of
        values aligned with `data` rows representing the variable to test.
    fs : float
        Sampling frequency in Hz.
    window_duration : float
        Window duration in seconds.
    overlap_ratio : float
        Fractional overlap between windows.

    Returns
    -------
    features : matrix, shape (n_windows, n_features)
        Z-score normalized feature matrix with one row per valid window.
    labels : matrix, shape (n_windows, 2)
        Integer matrix with (activity_label, participant_id) for each window.
    feature_names : list
        List of strings naming each column in `features`."""

    base_feature_names = [
        "mean", "std", "median", "variance", "rms", "average_deviation",
        "skewness", "kurtosis", "iqr", "zero_crossing_rate", "mean_crossing_rate", "spectral_entropy"
    ]

    feature_names = [f"acc_{name}" for name in base_feature_names] + \
                    [f"gyro_{name}" for name in base_feature_names] + \
                    [f"mag_{name}" for name in base_feature_names]

    windows = _sliding_windows_timestamp(data, fs, window_duration, overlap_ratio)

    features = []
    labels = []

    acceleration_modules = variables_modules[0][1]
    gyroscope_modules = variables_modules[1][1]
    magnetic_modules = variables_modules[2][1]

    for (start_idx, end_idx, activity_label, device_id, participant) in windows:
        acc_window = acceleration_modules[start_idx:end_idx]
        gyro_window = gyroscope_modules[start_idx:end_idx]
        mag_window = magnetic_modules[start_idx:end_idx]

        if acc_window.size == 0 or gyro_window.size == 0 or mag_window.size == 0:
            continue

        acc_features = _extract_window_features(acc_window)
        gyro_features = _extract_window_features(gyro_window)
        mag_features = _extract_window_features(mag_window)

        combined_features = acc_features + gyro_features + mag_features

        participant = int(data[start_idx, 12]) 
        labels.append((activity_label, participant))
        features.append(combined_features)

    features = np.array(features, dtype=float)
    labels = np.array(labels, dtype=int) 

    features = _zscore_normalization(features)

    return features, labels, feature_names

# --- Exercise 4.3: PCA ---

def compute_pca(features, n_components=None):
    """Perform principal component analysis and return projected data.

    Parameters
    ----------
    features : matrix, shape (n_samples, n_features)
        Input feature matrix.
    n_components : int or None
        Number of principal components to compute. If None, uses `min(features.shape)`.

    Returns
    -------
    pca_matrix : matrix, shape (n_samples, n_components)
        The transformed data (scores) of shape (n_samples, n_components).
    explained_variance_ratio : array 
        Array containing the variance ratio explained by each component."""

    n_components = n_components or min(features.shape)

    pca = PCA(n_components=n_components)
    pca_matrix = pca.fit_transform(features)

    explained_variance_ratio = pca.explained_variance_ratio_

    return pca_matrix, explained_variance_ratio

# --- Exercise 4.4: PCA Analysis  ---

def analyse_pca(explained_variance_ratio):
    """Plot the explained variance ratio and cumulative variance from PCA.

    Parameters
    ----------
    explained_variance_ratio : array
        Per-component explained variance ratio as provided by a fitted PCA."""

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

def fisher(features, labels, feature_names):
    """Compute Fisher scores for features and return the top-ranked names.

    Fisher score is computed as the ratio of between-class variance to
    within-class variance for each feature.

    Parameters
    ----------
    features : matrix, shape (n_samples, n_features)
        Feature matrix where rows correspond to samples.
    labels : matrix, shape (n_samples, 2)
        For activity recognition, the first column is treated as the activity label.
    feature_names : list of str 
        Names used for pretty printing of top features.

    Returns
    -------
    top10 : list of str
        List with up to 10 feature names sorted by descending Fisher score."""

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

    top10 = []

    print("\n========== Fisher Score ==========")
    for i in range(min(10, n_features)):
        name = feature_names[sorted_idx[i]] if feature_names else f"feature_{sorted_idx[i]}"
        print(f"{i+1:02d}. {name:>20s}  |  score = {sorted_scores[i]:.4f}")
        top10.append(name)

    return top10


def relief(features, labels, feature_names, n_neighbors=50, n_samples=500):
    """Approximate ReliefF feature ranking using random sampling.

    This is a simplified and efficient approximation of ReliefF. A subset 
    of `n_samples` examples is chosen and for each chosen instance the 
    average distance to `n_neighbors` "hits"(same-class) and "misses" 
    (different-class) is computed. The accumulated difference (miss - hit) 
    across samples produces a score for each feature: larger positive values 
    indicate stronger discrimination.

    Parameters
    ----------
    features : matrix, shape (n_samples, n_features)
        Feature matrix.
    labels : matrix, shape (n_samples, 2)
        Class labels as a 1-D array or an (n_samples, 2) array where the first
        column is the class label.
    feature_names : list of str
        Names for printing.
    n_neighbors : int
        Number of neighbours used to estimate hit/miss distances.
    n_samples : int
        Number of random samples to draw for the approximation.

    Returns
    -------
    top10 : list of str
        Top 10 feature names sorted by the approximate ReliefF score."""
    
    n_total = features.shape[0]
    n_features = features.shape[1]
    sample_idx = np.random.choice(n_total, min(n_samples, n_total), replace=False)
    X_sample = features[sample_idx]
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

    top10 = []

    print("\n========== ReliefF ==========\n")
    for i in range(min(10, n_features)):
        name = feature_names[sorted_idx[i]] if feature_names else f"feature_{sorted_idx[i]}"
        print(f"{i+1:02d}. {name:>20s}  |  score = {sorted_scores[i]:.4f}")
        top10.append(name)
    
    return top10