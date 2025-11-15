from load import *
from outliers import *
from features import *
from augmentation import *

def partA ():

    # --- Exercise 1: Load Part Data ---

    part_data = load_part_data(0)

    # --- Exercise 2: Load Full Data ---

    data = load_data()

    # --- Exercise 3.1: Variable Modules ---
    acc_modules = variable_module(data, "Acceleration")
    mag_modules = variable_module(data, "Magnetic Field")
    gyro_modules = variable_module(data, "Angular Velocity") 

    variables_modules = [
		("Acceleration (|Acc|)", acc_modules),
		("Angular Velocity (|Gyro|)", gyro_modules),
		("Magnetic Field (|Mag|)", mag_modules)
	]

    print("\n--- Boxplots of Variable Modules ---")
    device_choice = int(input("\nPick a device (1-5): "))
    for name, modules in variables_modules:
        boxplot_variable(data, modules, name, device_choice)

    # --- Exercise 3.2: Outlier Densities via IQR ---

    print("\n--- Acceleration Outlier Densities via IQR ---\n")
    outlier_density_iqr(data, acc_modules)
    print("\n--- Magnetic Field Outlier Densities via IQR ---\n")
    outlier_density_iqr(data, mag_modules)
    print("\n--- Angular Velocity Outlier Densities via IQR ---\n")
    outlier_density_iqr(data, gyro_modules)

    # --- Exercise 3.3 and 3.4: Outlier Detection via Z-Score ---

    print("\n--- Outlier Detection via Z-Score ---\n")

    k=3

    activity_choice = int(input("Pick an activity (1-16): "))

    print("\n--- Acceleration Outliers via Z-Score ---\n")
    outlier_density_z_score(data, acc_modules, k)
    acc_outlier_idxs = z_score(data, acc_modules, activity_choice, k)

    print("\n--- Magnetic Field Outliers via Z-Score ---\n")
    outlier_density_z_score(data, mag_modules, k)
    mag_outlier_idxs = z_score(data, mag_modules, activity_choice, k)

    print("\n--- Angular Velocity Outliers via Z-Score ---\n")
    outlier_density_z_score(data, gyro_modules, k)
    gyro_outlier_idxs = z_score(data, gyro_modules, activity_choice, k)

    plot_zscore_outliers(data, acc_modules, "Acceleration", activity_choice, acc_outlier_idxs)
    plot_zscore_outliers(data, mag_modules, "Magnetic Field", activity_choice, mag_outlier_idxs)
    plot_zscore_outliers(data, gyro_modules, "Angular Velocity", activity_choice, gyro_outlier_idxs)

    # --- Exercise 3.6 and 3.7: Clustering ---

    n_clusters = int(input("\n--- K-Means Clustering ---\n\nChoose the number of clusters: (e.g., 2, 3, 4): "))
    print(f"\n--- Acceleration K-Means Results ---\n")
    acc_labels, acc_centers = kmeans(acc_modules, n_clusters)
    print(f"\n--- Magnetic Field K-Means Results ---\n")
    mag_labels, mag_centers = kmeans(mag_modules, n_clusters)
    print(f"\n--- Angular Velocity K-Means Results ---\n")
    gyro_labels, gyro_centers = kmeans(gyro_modules, n_clusters)

    activity_choice = int(input("\nPick an activity to highlight outliers (1-16): "))
    device_choice = int(input("Pick a device to highlight outliers (1-5): "))

    plot_kmeans_clusters(data, activity_choice, device_choice, acc_modules, gyro_modules, mag_modules, n_clusters=n_clusters)
    
    plot_dbscan_clusters(data, activity_choice, device_choice, acc_modules, gyro_modules, mag_modules)

    # --- Exercise 4.1: Statistical Tests ---

    alpha = 0.05
    normality_and_significance(data, variables_modules, alpha)

    # --- Exercise 4.2: Feature Extraction ---
    
    features, labels, feature_names = extract_features(data, variables_modules, fs=51.5, window_duration=5.0, overlap_ratio=0.5)

    # --- Exercise 4.3: PCA ---

    n_components = 36
    pca, explained_variance_ratio = pca(features, n_components)

    # --- Exercise 4.4: PCA Analysis ---

    pca_analysis(explained_variance_ratio)

    # --- Exercise 4.5: Fisher Scores and ReliefF ---

    fisher(features, labels, feature_names)
    relief(features, labels, feature_names=feature_names, n_neighbors=100)

def partB ():

    # --- Pre game data loading and preprocessing ---

    features, pca, scores, labels = reload_data()

    features, pca, labels = discard_activities(features, pca, labels)

if __name__ == "__main__":

    #--- Run Part A Exercises ---

    #partA()

    #--- Run Part B Exercises ---

    partB()