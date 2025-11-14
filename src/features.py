import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from scipy.stats import kstest, f_oneway, kruskal
from sklearn.decomposition import PCA

# --- Exercise 4.1: Statistical Tests ---

def normality_and_significance(data, variables_data, alpha=0.05):
    print("\n--- Normality and Significance Tests ---\n")
    for name, modules in variables_data:
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

def sliding_windows(data, fs, window_duration=5.0, overlap=0.5):
    labels = np.asarray(data[:, 11]).astype(int)      
    device_ids = np.asarray(data[:, 0]).astype(int)   
    participants = np.asarray(data[:, 12]).astype(int)  
    window_size = int(round(window_duration * fs))
    step = max(1, int(round(window_size * (1.0 - overlap))))

    label_windows = sliding_window_view(labels, window_shape=window_size)
    participant_windows = sliding_window_view(participants, window_shape=window_size)
    device_windows = sliding_window_view(device_ids, window_shape=window_size)

    starts = np.arange(0, label_windows.shape[0], step)
    label_candidates = label_windows[starts]
    device_candidates = device_windows[starts]
    participant_candidates = participant_windows[starts]

    mask_activity = np.all(label_candidates == label_candidates[:, :1], axis=1)
    mask_device = np.all(device_candidates == device_candidates[:, :1], axis=1)
    mask_participant = np.all(participant_candidates == participant_candidates[:, :1], axis=1)
    valid_mask = mask_activity & mask_device & mask_participant

    valid_starts = starts[valid_mask]

    windows = [
        (int(s), int(s + window_size), int(labels[s]), int(device_ids[s]), int(participants[s]))
        for s in valid_starts
    ]

    return windows

def sliding_windows_timestamp(data, fs, window_duration=5.0, overlap=0.5):
    labels = np.asarray(data[:, 11]).astype(int)
    device_ids = np.asarray(data[:, 0]).astype(int)
    participants = np.asarray(data[:, 12]).astype(int)
    timestamps = np.asarray(data[:, 10]).astype(float)

    window_duration_ms = int(window_duration * 1000)
    n = len(data)
    windows = []
    start = 0
    while start < n:
        activity_ref = labels[start]
        device_ref = device_ids[start]
        participant_ref = participants[start]
        t_start = timestamps[start]

        end = start
        while (end < n and
               labels[end] == activity_ref and
               device_ids[end] == device_ref and
               participants[end] == participant_ref and
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

    windows = sliding_windows_timestamp(data, fs, window_duration, overlap_ratio)

    features = []
    labels = []
    valid_windows, discarded_windows = 0, 0

    for (start_idx, end_idx, activity_label, device_id, participant) in windows:
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

        participant = int(data[start_idx, 12])  # participant ID is in column 12
        labels.append((activity_label, participant))
        features.append(combined_features)
        valid_windows += 1

    features = np.array(features, dtype=float)
    labels = np.array(labels, dtype=int) 

    features = zscore_normalization(features)

    return features, labels, feature_names

# --- Exercise 4.3: PCA ---

def pca(features, n_components=None):

    n_components = n_components or min(features.shape)

    pca = PCA(n_components=n_components)
    pca_matrix = pca.fit_transform(features)

    explained_variance_ratio = pca.explained_variance_ratio_

    return pca_matrix, explained_variance_ratio

# --- Exercise 4.4: PCA analysis  ---

def pca_analysis(explained_variance_ratio):
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

    print("\n========== Fisher Score ==========\n")
    for i in range(min(10, n_features)):
        name = feature_names[sorted_idx[i]] if feature_names else f"feature_{sorted_idx[i]}"
        print(f"{i+1:02d}. {name:>20s}  |  score = {sorted_scores[i]:.4f}")
        top10.append(name)

    return top10


def relief(features, labels, feature_names, n_neighbors=50, n_samples=500):

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