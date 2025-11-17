import torch
import numpy as np

from features import _sliding_windows

########################### PROVIDED CODE ###########################

def load_model():
  ''' Loads the model from the github repo and obtains just the feature encoder. '''

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
    acc_xyz: np.ndarray shape (N, 3) em m/s^2 (ou g), amostrado a fs_in_hz (float)
    devolve:
      acc_resampled: np.ndarray shape (M, 3) já a 30 Hz
      fs_target: 30.0
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
    """Computes embeddings for the entire dataset using a sliding window approach.

    This function ensures that the windows used for embeddings are the same as
    those used for traditional feature extraction, allowing for a direct comparison.

    Parameters
    ----------
    data : np.ndarray
        The raw dataset matrix.
    fs : float
        The sampling frequency of the data.
    window_duration : float
        The duration of the sliding window in seconds.
    overlap_ratio : float
        The fractional overlap between windows.
    batch_size : int
        The batch size for processing embeddings.

    Returns
    -------
    embeddings : np.ndarray
        The computed embeddings for each valid window, shape (n_windows, n_embeddings).
    labels : np.ndarray
        The labels for each window, shape (n_windows, 2)."""
    
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

