import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from scipy.stats import kruskal, kstest
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# --- Exercise 4.1: Statistical Tests ---

def choose_and_test_method(data, modules, alpha=0.05):
    normality_results = {}
    for activity in range(1, 17):
        activity_modules = modules[data[:, 11] == activity]
        z_values = (activity_modules - np.mean(activity_modules)) / np.std(activity_modules)
        stat, p_value = kstest(z_values, "norm")
        normality_results[activity] = (stat, p_value)


    p_values = [p for (_, p) in normality_results.values() if not np.isnan(p)]
    percentage_normal = 100.0 * sum(p > alpha for p in p_values) / len(p_values) if p_values else 0.0

    activity_groups = [modules[data[:, 11] == activity]
                       for activity in range(1, 17)
                       if np.sum(data[:, 11] == activity) > 1]


    stat, p_value = kruskal(*activity_groups)

    return stat, p_value, percentage_normal

# --- Exercise 4.2: Feature Extraction ---

def sliding_windows(data, fs, window_duration=5.0, overlap=0.5):
    labels = np.asarray(data[:, 11]).astype(int)      
    device_ids = np.asarray(data[:, 0]).astype(int)   

    window_size = int(round(window_duration * fs))
    step = max(1, int(round(window_size * (1.0 - overlap))))

    label_windows = sliding_window_view(labels, window_shape=window_size)
    device_windows = sliding_window_view(device_ids, window_shape=window_size)

    starts = np.arange(0, label_windows.shape[0], step)
    label_candidates = label_windows[starts]
    device_candidates = device_windows[starts]

    mask_activity = np.all(label_candidates == label_candidates[:, :1], axis=1)
    mask_device = np.all(device_candidates == device_candidates[:, :1], axis=1)
    valid_mask = mask_activity & mask_device

    valid_starts = starts[valid_mask]

    windows = [
        (int(s), int(s + window_size), int(labels[s]), int(device_ids[s]))
        for s in valid_starts
    ]

    return windows

def extract_window_features(signal):

    # Mean value
    mean_value = np.mean(signal)

    # Standard deviation
    std_value = np.std(signal, ddof=1) if signal.size > 1 else 0.0

    # Median value
    median_value = np.median(signal)

    # Variance
    variance_value = np.var(signal, ddof=1) if signal.size > 1 else 0.0

    # Root Mean Square (RMS)
    rms_value = np.sqrt(np.mean(signal ** 2))

    # Averaged Deviation
    average_deviation_value = np.mean(np.abs(signal - mean_value))

    # Skewness
    skewness_value = (np.mean((signal - mean_value) ** 3) / (std_value ** 3)) if std_value > 0 else 0.0

    # Kurtosis
    kurtosis_value = (np.mean((signal - mean_value) ** 4) / (std_value ** 4)) - 3 if std_value > 0 else 0.0

    # Interquartile Range (IQR)
    iqr_value = np.percentile(signal, 75) - np.percentile(signal, 25)

    # Zero Crossing Rate (ZCR)
    zero_crossings = np.sum(np.sign(signal[:-1]) * np.sign(signal[1:]) < 0) if signal.size > 1 else 0

    # Mean Crossing Rate (MCR)
    mean_crossings = np.sum((signal[:-1] - mean_value) * (signal[1:] - mean_value) < 0) if signal.size > 1 else 0

    # Spectral Entropy
    fft_values = np.fft.rfft(signal)
    power = np.abs(fft_values) ** 2
    power /= np.sum(power) + 1e-12
    spectral_entropy = -np.sum(power * np.log(power + 1e-12))

    feature_values = [mean_value, std_value, median_value, variance_value, rms_value, 
                      average_deviation_value, skewness_value, kurtosis_value, iqr_value,
                      zero_crossings, mean_crossings, spectral_entropy]

    return feature_values


def zscore_normalization(features, eps=1e-12):
    features = features.astype(float, copy=True)
    mean_values = np.nanmean(features, axis=0)
    std_values = np.nanstd(features, axis=0, ddof=1)
    std_values = np.where(std_values < eps, 1.0, std_values)
    features = (features - mean_values) / std_values
    return features


def extract_features(data, acceleration_modules, magnetic_modules, gyroscope_modules,
                     fs, window_duration=5.0, overlap_ratio=0.5):

    base_feature_names = [
        "mean", "std", "median", "variance", "rms", "average_deviation",
        "skewness", "kurtosis", "iqr", "zero_crossing_rate", "mean_crossing_rate", "spectral_entropy"
    ]

    feature_names = [f"acc_{name}" for name in base_feature_names] + \
                    [f"gyro_{name}" for name in base_feature_names] + \
                    [f"mag_{name}" for name in base_feature_names]

    windows = sliding_windows(data, fs, window_duration, overlap_ratio)

    features = []
    labels = []
    devices = []
    valid_windows, discarded_windows = 0, 0

    for (start_idx, end_idx, activity_label, device_id) in windows:
        acc_window = acceleration_modules[start_idx:end_idx]
        gyro_window = gyroscope_modules[start_idx:end_idx]
        mag_window = magnetic_modules[start_idx:end_idx]

        if acc_window.size == 0 or gyro_window.size == 0 or mag_window.size == 0:
            discarded_windows += 1
            continue

        acc_features = extract_window_features(acc_window)
        gyro_features = extract_window_features(gyro_window)
        mag_features = extract_window_features(mag_window)

        combined_features = acc_features + gyro_features + mag_features
        features.append(combined_features)
        labels.append(activity_label)
        devices.append(device_id)
        valid_windows += 1

    features = np.array(features, dtype=float)
    labels = np.array(labels, dtype=int)
    devices = np.array(devices, dtype=int)

    features = zscore_normalization(features)

    return features, labels, feature_names

# --- Exercise 4.3: PCA ---

def pca(features, n_components=None):

    n_components = n_components or min(features.shape)

    pca = PCA(n_components=n_components)
    projected_data = pca.fit_transform(features)

    explained_variance_ratio = pca.explained_variance_ratio_

    return {
        "projected_data": projected_data,
        "components": pca.components_,
        "explained_variance_ratio": explained_variance_ratio,
        "pca_model": pca,
        "features": features
    }


# --- Exercise 4.4: PCA analysis  ---

def pca_analysis(pca_results):
    explained_ratio = pca_results["explained_variance_ratio"]
    cumulative = np.cumsum(explained_ratio)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_ratio) + 1), explained_ratio,
            alpha=0.6, label="Variância explicada")
    plt.plot(range(1, len(cumulative) + 1), cumulative,
             color='red', marker='o', label="Variância acumulada")

    plt.title("PCA — Variância explicada por componente")
    plt.xlabel("Número de componentes principais")
    plt.ylabel("Proporção da variância explicada")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Exercise 4.5: Fisher Scores and ReliefF ---

def fisher(feature_matrix, labels, feature_names=None, top_features=10):

    labels = np.array(labels)
    n_features = feature_matrix.shape[1]
    unique_labels = np.unique(labels)

    overall_mean = np.nanmean(feature_matrix, axis=0)
    numerator = np.zeros(n_features, dtype=float)
    denominator = np.zeros(n_features, dtype=float)

    for c in unique_labels:
        class_mask = labels == c
        class_data = feature_matrix[class_mask]

        n_c_feature = np.sum(~np.isnan(class_data), axis=0)

        class_mean = np.nanmean(class_data, axis=0)

        diff = class_mean - overall_mean
        diff = np.where(np.isnan(diff), 0.0, diff)

        numerator += n_c_feature * (diff ** 2)

        denom_feature = np.nansum((class_data - class_mean) ** 2, axis=0)
        denominator += denom_feature

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.where(denominator > 0, numerator / denominator, 0.0)

    sorted_idxs = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_idxs]

    print("\n========== Fisher Score ==========\n")
    for i in range(min(top_features, n_features)):
        idx = sorted_idxs[i]
        name = feature_names[idx] if feature_names is not None else f"feature_{idx}"
        print(f"{i+1:02d}. {name:>20s}  |  score = {sorted_scores[i]:.4f}")

def relief(feature_matrix, labels, feature_names=None, top_features=10, n_neighbors=50, n_samples=500):
    n_samples_total = feature_matrix.shape[0]
    n_features = feature_matrix.shape[1]

    sample_indices = np.random.choice(n_samples_total, min(n_samples, n_samples_total), replace=False)
    X_sample = feature_matrix[sample_indices]
    y_sample = labels[sample_indices]

    scores = np.zeros(n_features)

    for i, x_i in enumerate(X_sample):
        same_class_mask = y_sample == y_sample[i]
        diff_class_mask = y_sample != y_sample[i]

        same_class_idx = np.random.choice(np.where(same_class_mask)[0], min(n_neighbors, sum(same_class_mask)), replace=True)
        diff_class_idx = np.random.choice(np.where(diff_class_mask)[0], min(n_neighbors, sum(diff_class_mask)), replace=True)

        hit_diff = np.mean(np.abs(X_sample[same_class_idx] - x_i), axis=0)
        miss_diff = np.mean(np.abs(X_sample[diff_class_idx] - x_i), axis=0)

        scores += miss_diff - hit_diff

    scores /= n_samples

    sorted_idxs = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_idxs]

    print("\n========== ReliefF ==========\n")
    for i in range(min(top_features, n_features)):
        idx = sorted_idxs[i]
        name = feature_names[idx] if feature_names is not None else f"feature_{idx}"
        print(f"{i+1:02d}. {name:>20s}  |  score = {sorted_scores[i]:.4f}")
