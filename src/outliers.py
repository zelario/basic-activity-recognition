
"""
Outlier detection and clustering utilities.
File used for full exercise 3.

This module contains functions used in the project for:
- computing vector modules for sensor data,
- visualizing module distributions via boxplots,
- detecting outliers using IQR and z-score methods,
- visualizing outliers for selected activities/devices,
- clustering samples using KMeans and DBSCAN,
- plotting clusters and highlighting outliers in 3D feature space.

Column conventions expected in `data` arrays:
- Column 0: device id
- Column 11: activity id
- Column 12: participant id
"""

from log import print_and_log
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

ACTIVITY_NAMES = [
    "Stand",
    "Sit",
    "Sit and Talk",
    "Walk",
    "Walk and Talk",
    "Climb Stair",
    "Climb Stair and talk",
    "Stand -> Sit",
    "Sit -> Stand",
    "Stand -> Sit and talk",
    "Sit -> Stand and talk",
    "Stand -> Walk",
    "Walk -> Stand",
    "Stand -> Climb Stairs, Stand -> Climb Stairs and Talk",
    "Climb Stairs -> Walk",
    "Climb Stairs and Talk -> Walk and Talk"
]

VARIABLE_NAMES = [
    "Acceleration (|Acc|)",
    "Angular Velocity (|Gyro|)",
    "Magnetic Field (|Mag|)"
]

# --- Exercise 3.1: Module Computation ---


def compute_modules(dataset):
    """
    Compute vector magnitudes (modules) for acceleration, gyroscope, and magnetometer.

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw dataset matrix. Columns 1:4, 4:7, 7:10 are x/y/z for Acc, Gyro, Mag.

    Returns
    -------
    variables_modules : list of (str, np.ndarray)
        List of (name, modules) for Acc, Gyro, Mag vector magnitudes.
    """
    
    acc_modules = np.linalg.norm(dataset[:, 1:4], axis=1)
    gyro_modules = np.linalg.norm(dataset[:, 4:7], axis=1)
    mag_modules = np.linalg.norm(dataset[:, 7:10], axis=1)

    variables_modules = [ acc_modules ,gyro_modules, mag_modules]

    return np.array(variables_modules)

def boxplot_modules(dataset, variables_modules):
    """
    Plot boxplots of variable modules for each activity and selected device.

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : list of (str, np.ndarray)
        List of (name, modules) for Acc, Gyro, Mag vector magnitudes.
    """
    
    print_and_log("\n--- Boxplots of Variable Modules ---")
    device_choice = int(input("\nPick a device (1-5): "))

    for name, modules in zip(VARIABLE_NAMES, variables_modules):
        activities_col = dataset[:, 11]
        devices_col = dataset[:, 0]
        activities_number = np.arange(1, 17)
        box_data = []
        positions = []

        for act in activities_number:
            vals = modules[(activities_col == act) & (devices_col == device_choice)]
            if vals.size > 0:
                box_data.append(vals)
                positions.append(act)
        plt.figure(figsize=(10, 6))

        if box_data:
            plt.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True, manage_ticks=False)

        plt.xticks(activities_number, activities_number)
        plt.title(f"{name} modules - Device {device_choice}")
        plt.xlabel("Activity")
        plt.ylabel(f"{name} module")
        plt.xlim(0.4, 16.6)
        plt.tight_layout()
        plt.show()

# --- Exercise 3.2: Outlier Densities via IQR ---

def outlier_density_iqr(dataset, variables_modules):
    """
    Compute outlier densities per activity using IQR method (right wrist only).

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : list of (str, np.ndarray)
        List of (name, modules) for Acc, Gyro, Mag vector magnitudes.
    """
    
    for variable_name, variable_modules in zip(VARIABLE_NAMES, variables_modules):
        print_and_log(f"\n--- {variable_name} Outlier Densities via IQR ---\n")
        right_wrist_mask = dataset[:, 0] == 2
        activities = dataset[right_wrist_mask, 11]
        variable_modules_right = variable_modules[right_wrist_mask]
        activities_number = np.arange(1, 17)
        densities = np.zeros(16)

        for idx, act in enumerate(activities_number):
            activity_mask = activities == act
            act_modules = variable_modules_right[activity_mask]
            q1, q3 = np.percentile(act_modules, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = np.logical_or(act_modules < lower, act_modules > upper)
            density = np.mean(outliers) * 100
            densities[idx] = density
            print_and_log(f"Activity {act}: {density:.2f}% ({np.sum(outliers)}/{act_modules.size})")

# --- Exercise 3.3 and 3.4: Outlier Detection via Z-Score ---

def outlier_density_z_score(dataset, variables_modules, k):
    """
    Compute outlier densities per activity using z-score method (right wrist only).

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : list of (str, np.ndarray)
        List of (name, modules) for Acc, Gyro, Mag vector magnitudes.
    k : float
        Z-score threshold for outlier detection.
    """
    
    print_and_log("\n--- Outlier Detection via Z-Score ---")

    for variable_name, variable_modules in zip(VARIABLE_NAMES, variables_modules):
        print_and_log(f"\n--- {variable_name} Outlier Densities via Z-Score ---\n")
        right_wrist_mask = dataset[:, 0] == 2
        activities = dataset[right_wrist_mask, 11]
        variable_modules_right = variable_modules[right_wrist_mask]

        for activity in range(1, 17):
            activity_mask = activities == activity
            act_vals = variable_modules_right[activity_mask]
            n = act_vals.size

            if n == 0:
                print_and_log(f"Activity {activity}: 0.00% (0/0)")
                continue
            std = np.std(act_vals)

            if std == 0 or not np.isfinite(std):
                density = 0.0
                outlier_count = 0

            else:
                z_scores = (act_vals - np.mean(act_vals)) / std
                outlier_mask = np.abs(z_scores) > k
                outlier_count = int(outlier_mask.sum())
                density = float(outlier_mask.mean() * 100.0)
            print_and_log(f"Activity {activity}: {density:.2f}% ({outlier_count}/{n})")

def _z_score_indices(dataset, var_modules, activity, k):
    """
    Return indices of outliers in a variable module for a given activity using z-score.

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw dataset matrix.
    var_modules : np.ndarray
        Variable module values (Acc, Gyro, Mag).
    activity : int
        Activity id to filter.
    k : float
        Z-score threshold for outlier detection.

    Returns
    -------
    outlier_idxs : np.ndarray
        Indices of outlier samples in the filtered activity.
    """
    
    activity_mask = dataset[:, 11] == activity
    activity_data = var_modules[activity_mask]
    mean = np.mean(activity_data)
    std = np.std(activity_data)
    z_scores = (activity_data - mean) / std
    outlier_idxs = np.where(np.abs(z_scores) > k)[0]
    return outlier_idxs

def plot_zscore_outliers(dataset, variables_modules, k):
    """
    Plot z-score outliers for a selected activity for each variable module.

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : list of (str, np.ndarray)
        List of (name, modules) for Acc, Gyro, Mag vector magnitudes.
    k : float
        Z-score threshold for outlier detection.
    """
    
    print_and_log("\n--- Z-Score Outlier Visualization ---\n")
    activity_id = int(input("Pick an activity to highlight outliers (1-16): "))

    for variable_name, variable_modules in zip(VARIABLE_NAMES, variables_modules):
        outlier_idxs = _z_score_indices(dataset, variable_modules, activity_id, k=k)
        activity_mask = dataset[:, 11] == activity_id
        activity_data = variable_modules[activity_mask]
        idxs = np.arange(activity_data.size)
        plt.figure(figsize=(10, 4))
        plt.scatter(idxs, activity_data, color='blue', label='Normal')
        plt.scatter(idxs[outlier_idxs], activity_data[outlier_idxs], color='red', label='Outlier')
        plt.title(f"Normal and outlier samples - {variable_name} - Activity {activity_id}")
        plt.xlabel("Sample Index")
        plt.ylabel("Variable")
        plt.legend()
        plt.show()

def remove_outliers(dataset, variables_modules, k):
    """
    Remove z-score outliers from the dataset for all variable modules.

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : list of (str, np.ndarray)
        List of (name, modules) for Acc, Gyro, Mag vector magnitudes.
    k : float
        Z-score threshold for outlier detection.

    Returns
    -------
    cleaned_dataset : np.ndarray
        Dataset with outlier samples removed.
    """
    
    outlier_mask = np.zeros(dataset.shape[0], dtype=bool)

    for variable_modules in variables_modules:
        for activity in range(1, 17):
            activity_mask = dataset[:, 11] == activity
            activity_outlier_idxs = _z_score_indices(dataset, variable_modules, activity, k=k)
            full_outlier_idxs = np.where(activity_mask)[0][activity_outlier_idxs]
            outlier_mask[full_outlier_idxs] = True

    # Remove outliers from dataset and all variable modules
    cleaned_dataset = dataset[~outlier_mask]
    cleaned_modules = [variable_modules[~outlier_mask] for variable_modules in variables_modules]
    return cleaned_dataset, cleaned_modules

# --- Exercise 3.6 and 3.7: Clustering ---

def kmeans(variables_modules, n_clusters):
    """
    Run KMeans clustering for each variable module and print cluster centers/counts.

    Parameters
    ----------
    variables_modules : list of (str, np.ndarray)
        List of (name, modules) for Acc, Gyro, Mag vector magnitudes.
    n_clusters : int
        Number of clusters to use for KMeans.
    """
    
    n_clusters = int(input("\n--- K-Means Clustering ---\n\nChoose the number of clusters: (e.g., 2, 3, 4): "))

    for variable_name, variable_modules in zip(VARIABLE_NAMES, variables_modules):
        print_and_log(f"\n--- {variable_name} K-Means Results ---\n")
        reshaped_data = np.array(variable_modules).reshape(-1, 1)
        kmeans = KMeans(n_clusters, random_state=42)
        kmeans.fit(reshaped_data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        for i in range(n_clusters):
            count = np.sum(labels == i)
            print_and_log(f"Cluster {i}: center = {centers[i][0]:.3f}, samples = {count}")

def plot_kmeans_clusters(dataset, variables_modules, n_clusters=3):
    """
    Plot KMeans clusters and highlight IQR outliers in 3D feature space for selected activity/device.

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : list of (str, np.ndarray)
        List of (name, modules) for Acc, Gyro, Mag vector magnitudes.
    n_clusters : int, optional
        Number of clusters for KMeans (default=3).
    """
    
    activity_choice = int(input("\nPick an activity to highlight outliers (1-16): "))
    device_choice = int(input("Pick a device to highlight outliers (1-5): "))

    acc_modules = variables_modules[0]
    gyro_modules = variables_modules[1]
    mag_modules = variables_modules[2]

    device_mask = dataset[:, 0] == device_choice
    activity_mask = dataset[:, 11] == activity_choice
    combined_mask = device_mask & activity_mask

    activity_idxs = np.where(combined_mask)[0]
    filtered_data_3d = np.column_stack([acc_modules, gyro_modules, mag_modules])[activity_idxs]
    outlier_mask = np.zeros(len(filtered_data_3d), dtype=bool)

    for i in range(3):
        var_data = filtered_data_3d[:, i]
        if var_data.size == 0:
            continue
        q1, q3 = np.percentile(var_data, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask |= (var_data < lower) | (var_data > upper)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(filtered_data_3d)
    cluster_labels = kmeans.labels_
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_clusters))

    for i, cluster in enumerate(unique_clusters):
        cluster_mask = cluster_labels == cluster
        cluster_mask[outlier_mask] = False
        ax.scatter(
            filtered_data_3d[cluster_mask, 0],
            filtered_data_3d[cluster_mask, 1],
            filtered_data_3d[cluster_mask, 2],
            s=50, alpha=0.6, color=colors(i),
            label=f'Cluster {cluster}'
        )
        
    ax.scatter(
        filtered_data_3d[outlier_mask, 0],
        filtered_data_3d[outlier_mask, 1],
        filtered_data_3d[outlier_mask, 2],
        color='red', s=50, marker='o', label='Outlier'
    )
    ax.set_xlabel('|Acceleration|')
    ax.set_ylabel('|Angular Velocity|')
    ax.set_zlabel('|Magnetic Field|')
    plt.title(f'KMeans clusters with outliers - Activity {activity_choice} - Device {device_choice}')
    ax.legend()
    plt.show()


def plot_dbscan_clusters(dataset, variables_modules, eps=0.5, min_samples=5):
    """
    Plot DBSCAN clusters and highlight outliers in 3D feature space for selected activity/device.

    Parameters
    ----------
    dataset : np.ndarray, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : list of (str, np.ndarray)
        List of (name, modules) for Acc, Gyro, Mag vector magnitudes.
    eps : float, optional
        DBSCAN epsilon parameter (default=0.5).
    min_samples : int, optional
        DBSCAN min_samples parameter (default=5).
    """
    
    print_and_log(f"\n--- DBSCAN Clustering ---")

    acc_modules = variables_modules[0]
    gyro_modules = variables_modules[1]
    mag_modules = variables_modules[2]

    activity_choice = int(input("\nPick an activity to highlight outliers (1-16): "))
    device_choice = int(input("Pick a device to highlight outliers (1-5): "))

    mask = (dataset[:, 0] == device_choice) & (dataset[:, 11] == activity_choice)
    X = np.column_stack([acc_modules, gyro_modules, mag_modules])[mask]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = np.sum(labels == -1)

    print_and_log(f"\nEstimated number of clusters: {n_clusters}")
    print_and_log(f"Number of outliers: {n_outliers}\n")
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.arange(len(unique_labels)))
    for i, cluster in enumerate(unique_labels):
        cluster_mask = labels == cluster
        color = 'red' if cluster == -1 else colors[i]
        ax.scatter(
            X[cluster_mask, 0], X[cluster_mask, 1], X[cluster_mask, 2],
            s=50, alpha=0.7, color=color,
            label='Outliers' if cluster == -1 else f'Cluster {cluster}'
        )

    ax.set_xlabel('|Acceleration|')
    ax.set_ylabel('|Angular Velocity|')
    ax.set_zlabel('|Magnetic Field|')
    ax.set_title(f'DBSCAN Clusters - Activity {activity_choice} - Device {device_choice}')
    ax.legend()
    plt.show()
