
"""
Outlier detection and clustering utilities.
File used for module A exercise 3.

This module contains functions used in the project for:
- computing vector modules for sensor data,
- visualizing module distributions via boxplots,
- detecting outliers using IQR and z-score methods,
- visualizing outliers for selected activities/devices,
- clustering samples using KMeans and DBSCAN,
- plotting clusters and highlighting outliers in 3D feature space.

Column conventions expected in `dataset` matrix:
- Column 0: device id
- Column 11: activity id
- Column 12: participant id
"""

from log import *

import numpy as np
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

# ---  3.1: Module Computation ---

def compute_modules(dataset):
    """Compute vector modules for acceleration, gyroscope, and magnetometer.

    Parameters
    ----------
    dataset : matrix, shape (n_samples, 13)
        Raw dataset matrix. Columns 1:4, 4:7, 7:10 are x/y/z for Acc, Gyro, Mag.

    Returns
    -------
    variables_modules : matrix, shape (3, n_samples)
        Matrix where each row corresponds to the module of Acc, Gyro, Mag vectors."""
    
    # Compute vector modules
    acc_modules = np.linalg.norm(dataset[:, 1:4], axis=1)
    gyro_modules = np.linalg.norm(dataset[:, 4:7], axis=1)
    mag_modules = np.linalg.norm(dataset[:, 7:10], axis=1)

    variables_modules = [acc_modules, gyro_modules, mag_modules]

    return np.array(variables_modules)

def boxplot_modules(dataset, variables_modules):
    """Plot boxplots of variable modules for each activity and selected device.

    Parameters
    ----------
    dataset : matrix, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : matrix, shape (3, n_samples)
        Matrix where each row corresponds to the module of Acc, Gyro, Mag vectors."""
    
    print_and_log("\n--- Boxplots of Variable Modules ---")
    device_choice = int(input("\nPick a device (1-5): "))

    # Plot boxplots for each variable modules
    for name, modules in zip(VARIABLE_NAMES, variables_modules):
        activities = dataset[:, 11]
        devices = dataset[:, 0]
        activities_number = np.arange(1, 17)
        box_data = []
        positions = []

        for activity in activities_number:
            values = modules[(activities == activity) & (devices == device_choice)]
            if values.size > 0:
                box_data.append(values)
                positions.append(activity)
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
    """Compute outlier densities per activity using IQR method (right wrist only).

    Parameters
    ----------
    dataset : matrix, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : matrix, shape (3, n_samples)
        Matrix where each row corresponds to the module of Acc, Gyro, Mag vectors."""
    
    # For each varible
    for variable_name, variable_modules in zip(VARIABLE_NAMES, variables_modules):
        print_and_log(f"\n--- {variable_name} Outlier Densities via IQR ---\n")
        right_wrist_mask = dataset[:, 0] == 2
        activities = dataset[right_wrist_mask, 11]
        variable_modules_right = variable_modules[right_wrist_mask]
        activities_number = np.arange(1, 17)
        densities = np.zeros(16)

        # For each activity, compute outlier density
        for index, activity in enumerate(activities_number):
            activity_mask = activities == activity
            activity_modules = variable_modules_right[activity_mask]
            q1, q3 = np.percentile(activity_modules, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = np.logical_or(activity_modules < lower, activity_modules > upper)
            density = np.mean(outliers) * 100
            densities[index] = density
            print_and_log(f"Activity {activity}: {density:.2f}% ({np.sum(outliers)}/{activity_modules.size})")

def remove_outliers(dataset):
    """Remove z-score outliers from the dataset for all variable modules.

    Parameters
    ----------
    dataset : matrix, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : matrix, shape (3, n_samples)
        Matrix where each row corresponds to the module of Acc, Gyro, Mag vectors.

    Returns
    -------
    cleaned_dataset : matrix
        Dataset with outlier samples removed."""
    
    variables_modules = compute_modules(dataset)
    
    outlier_mask = np.zeros(dataset.shape[0], dtype=bool)

    # For each variable module, detect outliers per activity
    for variable_modules in variables_modules:
        for activity in range(1, 17):
            activity_mask = dataset[:, 11] == activity
            activity_indexes = np.where(activity_mask)[0]
            activity_data = variable_modules[activity_mask]
            q1, q3 = np.percentile(activity_data, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_idxs = np.where((activity_data < lower) | (activity_data > upper))[0]
            full_outlier_indexes = activity_indexes[outlier_idxs]
            outlier_mask[full_outlier_indexes] = True

    # Remove outliers from dataset
    cleaned_dataset = dataset[~outlier_mask]

    return cleaned_dataset

# --- Exercise 3.3 and 3.4: Outlier Detection via Z-Score ---

def outlier_density_z_score(dataset, variables_modules, k=3):
    """Compute outlier densities per activity using z-score method (right wrist only).

    Parameters
    ----------
    dataset : matrix, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : matrix, shape (3, n_samples)
        Matrix where each row corresponds to the module of Acc, Gyro, Mag vectors.
    k : float, optional
        Z-score threshold for outlier detection. Default is 3."""
    
    print_and_log("\n--- Outlier Detection via Z-Score ---")

    # For each variable
    for variable_name, variable_modules in zip(VARIABLE_NAMES, variables_modules):
        print_and_log(f"\n--- {variable_name} Outlier Densities via Z-Score ---\n")
        right_wrist_mask = dataset[:, 0] == 2
        activities = dataset[right_wrist_mask, 11]
        variable_modules_right = variable_modules[right_wrist_mask]

        # For each activity, compute outlier density
        for activity in range(1, 17):
            activity_mask = activities == activity
            activity_modules = variable_modules_right[activity_mask]
            n = activity_modules.size

            if n == 0:
                print_and_log(f"Activity {activity}: 0.00% (0/0)")
                continue
            std = np.std(activity_modules)

            if std == 0 or not np.isfinite(std):
                density = 0.0
                outlier_count = 0

            else:
                z_scores = (activity_modules - np.mean(activity_modules)) / std
                outlier_mask = np.abs(z_scores) > k
                outlier_count = int(outlier_mask.sum())
                density = float(outlier_mask.mean() * 100.0)

            print_and_log(f"Activity {activity}: {density:.2f}% ({outlier_count}/{n})")

def plot_zscore_outliers(dataset, variables_modules, k=3):
    """Plot z-score outliers for a selected activity for each variable module.

    Parameters
    ----------
    dataset : matrix, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : matrix, shape (3, n_samples)
        Matrix where each row corresponds to the module of Acc, Gyro, Mag vectors.
    k : float
        Z-score threshold for outlier detection. Default is 3."""
    
    print_and_log("\n--- Z-Score Outlier Visualization ---\n")
    activity = int(input("Pick an activity to highlight outliers (1-16): "))

    # For each variable module, plot outliers for the selected activity
    for variable_name, variable_modules in zip(VARIABLE_NAMES, variables_modules):

        # Compute z-scores and identify outliers
        activity_mask = dataset[:, 11] == activity
        activity_data = variable_modules[activity_mask]
        mean = np.mean(activity_data)
        std = np.std(activity_data)
        z_scores = (activity_data - mean) / std
        outlier_indexes = np.where(np.abs(z_scores) > k)[0]

        # Plot normal samples and outliers
        activity_mask = dataset[:, 11] == activity
        activity_data = variable_modules[activity_mask]
        activity_indexes = np.arange(activity_data.size)
        plt.figure(figsize=(10, 4))
        plt.scatter(activity_indexes, activity_data, color='blue', label='Normal')
        plt.scatter(activity_indexes[outlier_indexes], activity_data[outlier_indexes], color='red', label='Outlier')
        plt.title(f"Normal and outlier samples - {variable_name} - Activity {activity}")
        plt.xlabel("Sample Index")
        plt.ylabel("Variable")
        plt.legend()
        plt.show()

# --- Exercise 3.6 and 3.7: Clustering ---

def kmeans(variables_modules, n_clusters):
    """Run KMeans clustering for each variable module and print cluster centers/counts.

    Parameters
    ----------
    variables_modules : matrix, shape (3, n_samples)
        Matrix where each row corresponds to the module of Acc, Gyro, Mag vectors.
    n_clusters : int
        Number of clusters to use for KMeans.
    """
    
    n_clusters = int(input("\n--- K-Means Clustering ---\n\nChoose the number of clusters: (e.g., 2, 3, 4): "))

    # For each variable
    for variable_name, variable_modules in zip(VARIABLE_NAMES, variables_modules):
        print_and_log(f"\n--- {variable_name} K-Means Results ---\n")

        # Fit KMeans model
        reshaped_data = np.array(variable_modules).reshape(-1, 1)
        kmeans = KMeans(n_clusters, random_state=42)
        kmeans.fit(reshaped_data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        for i in range(n_clusters):
            count = np.sum(labels == i)
            print_and_log(f"Cluster {i}: center = {centers[i][0]:.3f}, samples = {count}")

def plot_kmeans_clusters(dataset, variables_modules, n_clusters=3):
    """Plot KMeans clusters and highlight IQR outliers in 3D feature space for selected activity/device.

    Parameters
    ----------
    dataset : matrix, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : matrix, shape (3, n_samples)
        Matrix where each row corresponds to the module of Acc, Gyro, Mag vectors.
    n_clusters : int, optional
        Number of clusters for KMeans (default=3)."""
    
    activity_choice = int(input("\nPick an activity to highlight outliers (1-16): "))
    device_choice = int(input("Pick a device to highlight outliers (1-5): "))

    # Get variables modules
    acc_modules = variables_modules[0]
    gyro_modules = variables_modules[1]
    mag_modules = variables_modules[2]

    # Create masks for selected device and activity
    device_mask = dataset[:, 0] == device_choice
    activity_mask = dataset[:, 11] == activity_choice
    combined_mask = device_mask & activity_mask

    # Filter data for selected activity/device
    activity_indexes = np.where(combined_mask)[0]
    filtered_data_3d = np.column_stack([acc_modules, gyro_modules, mag_modules])[activity_indexes]
    outlier_mask = np.zeros(len(filtered_data_3d), dtype=bool)

    # For each variable
    for i in range(3):

        # Calculate IQR and identify outliers for each variable
        variable_data = filtered_data_3d[:, i]
        if variable_data.size == 0:
            continue
        q1, q3 = np.percentile(variable_data, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask |= (variable_data < lower) | (variable_data > upper)

    # Fit KMeans and plot clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(filtered_data_3d)
    cluster_labels = kmeans.labels_
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_clusters))

    # Plot each cluster and outliers
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
    """Plot DBSCAN clusters and highlight outliers in 3D feature space for selected activity/device.

    Parameters
    ----------
    dataset : matrix, shape (n_samples, 13)
        Raw dataset matrix.
    variables_modules : matrix, shape (3, n_samples)
        Matrix where each row corresponds to the module of Acc, Gyro, Mag vectors.
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

    # Filter data for selected activity/device
    mask = (dataset[:, 0] == device_choice) & (dataset[:, 11] == activity_choice)
    X = np.column_stack([acc_modules, gyro_modules, mag_modules])[mask]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = np.sum(labels == -1)

    print_and_log(f"\nEstimated number of clusters: {n_clusters}")
    print_and_log(f"Number of outliers: {n_outliers}\n")
    
    # Plot clusters and outliers
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
