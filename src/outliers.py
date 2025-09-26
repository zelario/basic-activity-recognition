import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from load import *

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

def boxplot_variable(data, trans_data, variable):

    activities = data[:, 11] 
    activities_number = np.arange(1, 17)
    box_data = [trans_data[activities == act] for act in activities_number]

    plt.figure(figsize=(10, 6))
    plt.boxplot(box_data, labels=activities_number)
    plt.title(f"{variable} modules")
    plt.xlabel("Activity")
    plt.ylabel(f"{variable} module")
    plt.show()

def outlier_density(data, var_trans_data):

    right_wrist_mask = data[:, 0] == 2
    activities = data[right_wrist_mask, 11]
    var_trans_data_right = var_trans_data[right_wrist_mask]
    activities_number = np.arange(1, 17)
    densities = np.zeros(16) 

    print("\n--- Outlier Densities by Activity (Right Wrist Sensor only): ---")
    for idx, act in enumerate(activities_number):
        activity_mask = activities == act
        act_modules = var_trans_data_right[activity_mask]
        q1, q3 = np.percentile(act_modules, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = np.logical_or(act_modules < lower, act_modules > upper)
        density = np.mean(outliers) * 100  
        densities[idx] = density
        print(f"Activity {act}: {density:.2f}% ({np.sum(outliers)}/{act_modules.size})")

def z_score(data, var_trans_data, activity, k):
    activity_mask = data[:, 11] == activity
    activity_data = var_trans_data[activity_mask]
    mean = np.mean(activity_data)
    std = np.std(activity_data)
    z_scores = (activity_data - mean) / std 
    outlier_idxs = np.where(np.abs(z_scores) > k)[0]
    print(f"\n---Outliers detected on activity {activity} by z-score: ---\n", outlier_idxs)
    return outlier_idxs

def plot_zscore_outliers(data, trans_data, activity, outlier_idxs):
    activity_mask = data[:, 11] == activity
    activity_data = trans_data[activity_mask]
    idxs = np.arange(activity_data.size)

    plt.figure(figsize=(10, 4))
    plt.scatter(idxs, activity_data, color='blue', label='Normal')
    plt.scatter(idxs[outlier_idxs], activity_data[outlier_idxs], color='red', label='Outlier')
    plt.title(f"Normal and outlier samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Variable")
    plt.legend()
    plt.show()