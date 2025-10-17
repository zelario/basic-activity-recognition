import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from mpl_toolkits.mplot3d import Axes3D

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

def variable_module(data, variable):
    if variable == "Acceleration": idx = 1  
    elif variable == "Angular Velocity": idx = 4   
    elif variable == "Magnetic Field": idx = 7
    return np.linalg.norm(data[:, idx:idx+3], axis=1)

def boxplot_variable(data, var_modules, variable, device):
    activities_col = data[:, 11]
    devices_col = data[:, 0]
    activities_number = np.arange(1, 17)

    box_data = []
    positions = []
    for act in activities_number:
        vals = var_modules[(activities_col == act) & (devices_col == device)]
        if vals.size > 0:
            box_data.append(vals)
            positions.append(act)

    plt.figure(figsize=(10, 6))
    if box_data:
        plt.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True, manage_ticks=False)

    plt.xticks(activities_number, activities_number)
    plt.title(f"{variable} modules - Device {device}")
    plt.xlabel("Activity")
    plt.ylabel(f"{variable} module")
    plt.xlim(0.4, 16.6)
    plt.tight_layout()
    plt.show()

def outlier_density(data, var_modules):

    right_wrist_mask = data[:, 0] == 2
    activities = data[right_wrist_mask, 11]
    var_modules_right = var_modules[right_wrist_mask]
    activities_number = np.arange(1, 17)
    densities = np.zeros(16) 

    print("\n--- Outlier Densities by Activity (Right Wrist Sensor only): ---")
    for idx, act in enumerate(activities_number):
        activity_mask = activities == act
        act_modules = var_modules_right[activity_mask]
        q1, q3 = np.percentile(act_modules, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = np.logical_or(act_modules < lower, act_modules > upper)
        density = np.mean(outliers) * 100  
        densities[idx] = density
        print(f"Activity {act}: {density:.2f}% ({np.sum(outliers)}/{act_modules.size})")

def z_score(data, var_modules, activity, k):
    activity_mask = data[:, 11] == activity
    activity_data = var_modules[activity_mask]
    mean = np.mean(activity_data)
    std = np.std(activity_data)
    z_scores = (activity_data - mean) / std 
    outlier_idxs = np.where(np.abs(z_scores) > k)[0]
    print(f"\n---Outliers detected on activity {activity} by z-score: ---\n", outlier_idxs)
    return outlier_idxs

def plot_zscore_outliers(data, var_modules, variable_name, activity, outlier_idxs):
    activity_mask = data[:, 11] == activity
    activity_data = var_modules[activity_mask]
    idxs = np.arange(activity_data.size)

    plt.figure(figsize=(10, 4))
    plt.scatter(idxs, activity_data, color='blue', label='Normal')
    plt.scatter(idxs[outlier_idxs], activity_data[outlier_idxs], color='red', label='Outlier')
    plt.title(f"Normal and outlier samples - {variable_name} - Activity {activity}")
    plt.xlabel("Sample Index")
    plt.ylabel("Variable")
    plt.legend()
    plt.show()

def kmeans(var_modules, n_clusters):

    reshaped_data = np.array(var_modules).reshape(-1, 1)

    kmeans = KMeans(n_clusters, random_state=42)
    kmeans.fit(reshaped_data)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    for i in range(n_clusters):
        count = np.sum(labels == i)
        print(f"Cluster {i}: center = {centers[i][0]:.3f}, samples = {count}")

    return labels, centers

def plot_clusters(acc_centers, gyro_centers, mag_centers):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    n_clusters = min(len(acc_centers), len(gyro_centers), len(mag_centers))

    for i in range(n_clusters):
        ax.scatter(acc_centers[i], gyro_centers[i], mag_centers[i],
                   s=100, label=f'Cluster {i}')

    ax.set_xlabel('|Acceleration|')
    ax.set_ylabel('|Angular Velocity|')
    ax.set_zlabel('|Magnetic Field|')
    plt.title('Clusters')
    ax.legend()
    plt.show()
