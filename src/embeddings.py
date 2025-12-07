"""
Embeddings computation and pairing utilities.
File used in Exercise 2.

This module provides functions for:
- Loading a pretrained feature encoder from a public repository
- Resampling raw sensor data for embedding extraction
- Computing embeddings for sensor data using a sliding window approach
- Checking pairing between embeddings, features, and labels
"""

from features import *
from log import *

import torch
import numpy as np

########################### PROVIDED CODE ###########################

def load_model():
  ''' Loads the model from the github repo and obtains just the feature encoder. '''

  repo = 'OxWearables/ssl-wearables'
  # class_num não interessa para extrair features; mas o hub pede este arg
  model = torch.hub.load(repo, 'harnet5', class_num=5, pretrained=True)
  model.eval()

  # Passo crucial: ficar só com a parte auto-supervisionada
  # O README diz que há um 'feature_extractor' (pré-treinado) e um 'classifier' (não treinado). :contentReference[oaicite:14]{index=14}
  feature_encoder = model.feature_extractor
  feature_encoder.to("cpu")
  feature_encoder.eval()

  return feature_encoder

def resample_to_30hz_5s(acc_xyz, fs_in_hz):
    """Resample raw accelerometer data to 30 Hz over a 5-second window.

    Parameters
    ----------
    acc_xyz : np.ndarray, shape (N, 3)
        Raw accelerometer data (m/s^2 or g), sampled at fs_in_hz.
    fs_in_hz : float
        Original sampling frequency of the input data.

    Returns
    -------
    acc_resampled : np.ndarray, shape (M, 3)
        Resampled accelerometer data at 30 Hz for a 5-second window.
    fs_target : float
        Target sampling frequency (30.0 Hz)."""
    
    fs_target = 30.0
    win_size = 5 # in seconds
    t_in = np.arange(acc_xyz.shape[0]) / fs_in_hz
    t_out = np.arange(0, win_size, 1.0/fs_target)

    acc_resampled = np.zeros((len(t_out), 3), dtype=np.float32)
    for axis in range(3):
        acc_resampled[:, axis] = np.interp(t_out, t_in, acc_xyz[:, axis])

    return acc_resampled

# ======================== END OF PROVIDED CODE ========================

# --- Exercise 2.1: Embeddings Computing ---

def compute_embeddings(dataset, fs=51.5, window_duration=5.0, overlap_ratio=0.5):
    """Compute embeddings for the entire dataset using a sliding window approach.
    Ensures window alignment with traditional feature extraction for direct comparison.

    Parameters
    ----------
    data : np.ndarray
        Raw dataset matrix containing sensor data and metadata.
    fs : float, optional
        Sampling frequency of the data (default=51.5).
    window_duration : float, optional
        Duration of each sliding window in seconds (default=5.0).
    overlap_ratio : float, optional
        Fractional overlap between consecutive windows (default=0.5).
    batch_size : int, optional
        Batch size for processing embeddings (default=32).

    Returns
    -------
    embeddings : np.ndarray, shape (n_windows, n_embeddings)
        Computed embeddings for each valid window.
    labels : np.ndarray, shape (n_windows, 2)
        Labels for each window: [activity, participant]."""
    
    try:
        embeddings = np.load("cache/embeddings.npy", allow_pickle=True)
        embeddings_labels = np.load("cache/embedding_labels.npy", allow_pickle=True)
        return embeddings, embeddings_labels
    
    except FileNotFoundError:

        # Use the same sliding window function to ensure pairing with features pca and labels
        windows = sliding_windows(dataset, window_duration, overlap_ratio)

        resampled_windows = []
        labels = []

        # Iterate over windows, extract acc raw data, and resample
        for (start_idx, end_idx, activity, participant, device) in windows:
            raw_acc_segment = dataset[start_idx:end_idx, 1:4] 
            resampled_segment = resample_to_30hz_5s(raw_acc_segment, fs)
            resampled_windows.append(resampled_segment)
            labels.append([activity, participant, device])

        feature_encoder = load_model()
      
        embeddings_list = []
      
        # Reshape segments to [n_segments, dimensions(xyz), time]
        x_all = np.transpose(np.array(resampled_windows), (0, 2, 1))
        print(x_all.shape)
      
        batch_size = 5
        with torch.no_grad():
            for i in range(0, x_all.shape[0], batch_size):
                xb = torch.from_numpy(x_all[i:i+batch_size]).float().to("cpu")
                eb = feature_encoder(xb)
                embeddings_list.append(eb.cpu().numpy())

        # Concatenate results and create labels array
        embeddings = np.concatenate(embeddings_list, axis=0)
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1])
        print(embeddings.shape)

        labels = np.array(labels)

        np.save("cache/embeddings.npy", embeddings)
        np.save("cache/embedding_labels.npy", labels)

        return embeddings, labels
    
def check_pairing(embeddings, features, feature_labels, embedding_labels):
    """
    Check the pairing between embeddings, features, and labels by printing sample indices and values.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_windows, n_embeddings)
        Embeddings matrix for all windows.
    features : np.ndarray, shape (n_windows, n_features)
        Feature matrix for all windows.
    labels : np.ndarray, shape (n_windows, 2)
        Labels array for all windows: [activity, participant].

    Returns
    -------
    None
        Prints sample indices and corresponding values to stdout.
    """

    if embedding_labels is None: 
        return

    print_and_log("\n--- Checking pairing between embeddings, features, and labels ---\n")

    print_and_log(f"Features matrix shape: {features.shape}")
    print_and_log(f"Embeddings matrix shape: {embeddings.shape}")
    print_and_log(f"Feature labels shape: {feature_labels.shape}")
    print_and_log(f"Embedding labels shape: {embedding_labels.shape}")

    for feature_label, embedding_label in zip(feature_labels, embedding_labels):
        if not np.array_equal(feature_label, embedding_label):
            print_and_log(f"Label mismatch found: Feature label {feature_label}, Embedding label {embedding_label}")
            return
    print_and_log("\nAll labels match between features and embeddings.")
