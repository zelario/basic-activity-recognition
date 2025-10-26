import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, kruskal, kstest
from sklearn.decomposition import PCA
from skfeature.function.similarity_based import fisher_score
from sklearn.metrics import mean_squared_error
from skrebate import ReliefF

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

    use_anova = percentage_normal >= 80.0 and len(activity_groups) >= 2

    if use_anova:
        stat, p_value = f_oneway(*activity_groups)
        method = "ANOVA"
    else:
        stat, p_value = kruskal(*activity_groups)
        method = "Kruskal-Wallis"

    return method, stat, p_value, percentage_normal

# --- Exercise 4.2: Feature Extraction ---

def sliding_windows(labels, fs, window_duration=5.0, overlap=0.5):
    labels = np.asarray(labels)
    window_size = int(round(window_duration * fs))

    step = max(1, int(round(window_size * (1.0 - overlap))))

    windows = sliding_window_view(labels, window_shape=window_size)

    starts = np.arange(0, windows.shape[0], step)
    candidate_windows = windows[starts]

    mask = np.all(candidate_windows == candidate_windows[:, :1], axis=1)
    valid_starts = starts[mask]

    out = [(int(s), int(s + window_size), int(labels[s])) for s in valid_starts]
    return out

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
                                     fs, window_duration=5.0, overlap_ratio=0.5, normalize_zscore=True):
    
    base_feature_names = [
        "mean", "std", "median", "variance", "rms", "average_deviation", "skewness", "kurtosis", "iqr",
        "zero_crossing_rate", "mean_crossing_rate", "spectral_entropy"
    ]

    feature_names = [f"acc_{name}" for name in base_feature_names] + \
                    [f"gyro_{name}" for name in base_feature_names] + \
                    [f"mag_{name}" for name in base_feature_names]

    activity_labels = data[:, 11].astype(int)
    windows = sliding_windows(activity_labels, fs, window_duration, overlap_ratio)

    features = []
    labels = []

    valid_windows, discarded_windows = 0, 0
    for (start_idx, end_idx, activity_label) in windows:
        acceleration_window = acceleration_modules[start_idx:end_idx]
        gyroscope_window = gyroscope_modules[start_idx:end_idx]
        magnetic_window = magnetic_modules[start_idx:end_idx]

        if acceleration_window.size == 0 or gyroscope_window.size == 0 or magnetic_window.size == 0:
            discarded_windows += 1
            continue

        acc_features = extract_window_features(acceleration_window)
        gyro_features = extract_window_features(gyroscope_window)
        mag_features = extract_window_features(magnetic_window)

        combined_features = acc_features + gyro_features + mag_features
        features.append(combined_features)
        labels.append(activity_label)
        valid_windows += 1

    if len(features) == 0:
        print("Nenhuma janela válida encontrada.")
        return np.empty((0, len(feature_names))), np.array([]), feature_names

    features = np.array(features, dtype=float)
    labels = np.array(labels, dtype=int)

    if normalize_zscore:
        features = zscore_normalization(features)

    return features, labels, feature_names

# --- Exercise 4.3: PCA ---

def pca(features, labels=None, n_components=2):

    pca = PCA(n_components=n_components)

    projected_data = pca.fit_transform(features)

    explained_variance_ratio = pca.explained_variance_ratio_
    
    print("\n--- PCA (scikit-learn) ---")
    for i, var in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {var*100:.2f}% da variância explicada")
    
    cumulative_variance = np.cumsum(explained_variance_ratio)
    num_for_90 = np.searchsorted(cumulative_variance, 0.90) + 1
    print(f"Número de componentes para >=90% de variância: {num_for_90}")

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, n_components+1), explained_variance_ratio*100, marker='o')
    plt.xlabel("Componente Principal")
    plt.ylabel("Variância Explicada (%)")
    plt.title("PCA — Scree Plot")
    plt.tight_layout()
    plt.show()
    
    if projected_data.shape[1] >= 2:
        plt.figure(figsize=(6, 5))
        if labels is None:
            plt.scatter(projected_data[:, 0], projected_data[:, 1], s=10, alpha=0.7)
        else:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                plt.scatter(projected_data[mask, 0], projected_data[mask, 1], s=12, alpha=0.7,
                            label=f"Atividade {label}")
            plt.legend(markerscale=1.5, fontsize=8, ncol=2)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA — PC1 vs PC2")
        plt.tight_layout()
        plt.show()
    
    return projected_data, pca.components_, explained_variance_ratio

# --- Exercise 4.4: PCA analysis ---

def pca_analysis(features, feature_names=None, variance_threshold=0.75, instant_index=0, verbose=True):
    """
    Performs PCA on already normalized features (z-score done earlier in extract_features).

    1) Computes PCA with all components.
    2) Determines how many components explain >= variance_threshold (e.g. 75%).
    3) Returns compressed representation for a chosen instant and reconstruction.
    4) Prints useful info about explained variance and reconstruction quality.
    """

    if features.size == 0:
        raise ValueError("Matriz de features vazia.")

    # --- features are already normalized ---
    z_features = features.copy()

    # --- PCA with all possible components ---
    n_components_full = min(z_features.shape)
    pca = PCA(n_components=n_components_full)
    projected = pca.fit_transform(z_features)
    explained_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_ratio)

    # --- number of components for threshold ---
    num_components_for_threshold = int(np.searchsorted(cumulative, variance_threshold) + 1)

    if verbose:
        print("\n--- PCA: explicação de variância ---")
        for i, (er, cum) in enumerate(zip(explained_ratio, cumulative)):
            print(f"PC{i+1:02d}: {er*100:6.3f}%   |  acumulada: {cum*100:6.3f}%")
        print(f"\nNúmero de componentes necessárias para >= {variance_threshold*100:.1f}%: {num_components_for_threshold}")

    # --- extract compressed vector for chosen instant ---
    if instant_index < 0 or instant_index >= z_features.shape[0]:
        raise IndexError("instant_index fora do intervalo (0 .. n_samples-1).")

    K = num_components_for_threshold
    compressed_instant = projected[instant_index, :K]

    # --- reconstruct (in z-score space) ---
    components_K = pca.components_[:K, :]
    scores_K = compressed_instant.reshape(1, -1)
    approx_z = np.dot(scores_K, components_K).reshape(-1)

    # since features are already normalized, reconstruction = approx_z
    approx_original = approx_z
    original_instant = z_features[instant_index, :]

    mse = mean_squared_error(original_instant, approx_original)

    if verbose:
        print(f"\nInstante escolhido: {instant_index}")
        if feature_names is not None:
            top_features_names = feature_names[:min(10, len(feature_names))]
            print("Exemplo (primeiras features) — valor original vs reconstruído (aprox.):")
            for i, name in enumerate(top_features_names):
                print(f"  {name:30s} | orig = {original_instant[i]: .4f}  | recon = {approx_original[i]: .4f}")
        print(f"\nMSE de reconstrução para o instante {instant_index}: {mse:.6g}")
        print(f"Tamanho da compressão: {K} componentes (de {features.shape[1]} features)")

    results = {
        "z_features": z_features,
        "pca_model": pca,
        "explained_ratio": explained_ratio,
        "cumulative_explained": cumulative,
        "num_components_for_threshold": num_components_for_threshold,
        "compressed_instant": compressed_instant,
        "approx_original_instant": approx_original,
        "original_instant": original_instant,
        "reconstruction_mse": mse,
        "K": K
    }

    return results

# --- Exercise 4.5: Fisher Scores and ReliefF ---

from skfeature.function.similarity_based import fisher_score
from skrebate import ReliefF
import numpy as np
import matplotlib.pyplot as plt

def fisher_and_relief(feature_matrix, labels, feature_names=None, top_features=10, n_neighbors=100, show_plot=True):

    # Fisher Score
    scores_fisher = fisher_score.fisher_score(feature_matrix, labels)
    sorted_idx_fisher = np.argsort(scores_fisher)[::-1]
    sorted_scores_fisher = scores_fisher[sorted_idx_fisher]

    print("\n========== Fisher Score ==========")
    for i in range(min(top_features, len(sorted_idx_fisher))):
        idx = sorted_idx_fisher[i]
        name = feature_names[idx] if feature_names is not None and idx < len(feature_names) else f"feature_{idx}"
        print(f"{i+1:02d}. {name:>20s}  |  score = {sorted_scores_fisher[i]:.4f}")

    # ReliefF
    relief = ReliefF(n_neighbors=n_neighbors)
    relief.fit(feature_matrix, labels)
    scores_relief = relief.feature_importances_
    sorted_idx_relief = np.argsort(scores_relief)[::-1]
    sorted_scores_relief = scores_relief[sorted_idx_relief]

    print("\n========== ReliefF ==========")
    for i in range(min(top_features, len(sorted_idx_relief))):
        idx = sorted_idx_relief[i]
        name = feature_names[idx] if feature_names is not None and idx < len(feature_names) else f"feature_{idx}"
        print(f"{i+1:02d}. {name:>20s}  |  score = {sorted_scores_relief[i]:.4f}")

    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(range(top_features), sorted_scores_fisher[:top_features])
        plt.xticks(range(top_features),
                   [feature_names[i] if feature_names else f"f{i}" for i in sorted_idx_fisher[:top_features]],
                   rotation=45, ha='right')
        plt.ylabel("Fisher Score")
        plt.title("Top Features — Fisher Score")

        plt.subplot(1, 2, 2)
        plt.bar(range(top_features), sorted_scores_relief[:top_features])
        plt.xticks(range(top_features),
                   [feature_names[i] if feature_names else f"f{i}" for i in sorted_idx_relief[:top_features]],
                   rotation=45, ha='right')
        plt.ylabel("ReliefF Score")
        plt.title("Top Features — ReliefF")

        plt.tight_layout()
        plt.show()
    
    return {"fisher": {"scores": scores_fisher, "ranking": sorted_idx_fisher}, "relief": {"scores": scores_relief, "ranking": sorted_idx_relief}}
