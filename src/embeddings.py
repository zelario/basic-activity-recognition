import torch
import numpy as np

from features import _sliding_windows

########################### PROVIDED CODE ###########################

def load_model():
    """
    Load the feature extraction model from the OxWearables GitHub repository and return the feature encoder.

    Returns
    -------
    model : torch.nn.Module
        The feature encoder model with the final classification layer replaced by identity.
    """
    repo = 'OxWearables/ssl-wearables'
    # class_num não interessa para extrair features; mas o hub pede este arg
    model = torch.hub.load(repo, 'harnet5', class_num=5, pretrained=True)
    # Replace the final classification layer with an identity layer to get the embeddings
    model.fc = torch.nn.Identity()
    model.eval()
    model.to("cpu")
    return model

def resample_to_30hz_5s(acc_xyz, fs_in_hz):
    """
    Resample raw accelerometer data to 30 Hz over a 5-second window.

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
        Target sampling frequency (30.0 Hz).
    """
    fs_target = 30.0
    win_size = 5 # in seconds
    t_in = np.arange(acc_xyz.shape[0]) / fs_in_hz
    t_out = np.arange(0, win_size, 1.0/fs_target)

    acc_resampled = np.zeros((len(t_out), 3), dtype=np.float32)
    for axis in range(3):
        acc_resampled[:, axis] = np.interp(t_out, t_in, acc_xyz[:, axis])

    return acc_resampled

def compute_embeddings(data, fs=51.5, window_duration=5.0, overlap_ratio=0.5, batch_size=32):
    """
    Compute embeddings for the entire dataset using a sliding window approach.
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
        Labels for each window: [activity, participant].
    """
    
    try:
      embeddings = np.load("data/embeddings.npy", allow_pickle=True)
      return embeddings
    
    except FileNotFoundError:

      # Use the same sliding window function to ensure pairing with features pca and labels
      windows = _sliding_windows(data, window_duration, overlap_ratio)

      resampled_windows = []
      labels = []

      # Iterate over windows, extract acc raw data, and resample
      for (start_idx, end_idx, activity, participant) in windows:
          raw_acc_segment = data[start_idx:end_idx, 1:4]  # Acc data in cols 1, 2, 3
          resampled_segment = resample_to_30hz_5s(raw_acc_segment, fs)
          resampled_windows.append(resampled_segment)
          labels.append([activity, participant])

      # Load the feature extraction model
      feature_encoder = load_model()
      
      # Reshape segments to [n_segments, dimensions(xyz), time]
      x_all = np.transpose(np.array(resampled_windows), (0, 2, 1))
      
      # Process in batches to get embeddings
      embeddings_list = []
      with torch.no_grad():
          for i in range(0, x_all.shape[0], batch_size):
              xb = torch.from_numpy(x_all[i:i+batch_size]).float().to("cpu")
              eb = feature_encoder(xb)
              embeddings_list.append(eb.cpu().numpy())

      # Concatenate results and create labels array
      embeddings = np.concatenate(embeddings_list, axis=0)
      labels = np.array(labels)

      np.save("data/embeddings.npy", embeddings)

      return embeddings

