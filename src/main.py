import numpy as np
import matplotlib as mpl
from load import *
from outliers import *
from mpl_toolkits.mplot3d import Axes3D  
from features import *

if __name__ == "__main__":

    #Exercicio 2

    '''
    part_data = load_part_data(0)
    print("\nParticipant 0 data:\n")
    print(part_data)
    '''

    #Exercicio 3.1

    data = load_data_csv()
    acc_modules = variable_module(data, "Acceleration")
    mag_modules = variable_module(data, "Magnetic Field")
    gyro_modules = variable_module(data, "Angular Velocity")

    variables_data = [
		("Acceleration (|Acc|)", acc_modules),
		("Angular Velocity (|Gyro|)", gyro_modules),
		("Magnetic Field (|Mag|)", mag_modules)
	]

    print("\n--- Boxplots of Variable Modules ---")
    device_choice = int(input("\nPick a device (1-5): "))
    for name, modules in variables_data:
        boxplot_variable(data, modules, name, device_choice)

    #Exercicio 3.2

    outlier_density(data, acc_modules)

    #Exercicio 3.3 e 3.4

    print("\n--- Outlier Detection via Z-Score ---\n")

    k=3

    activity_choice = int(input("Pick an activity (1-16): "))

    acc_outlier_idxs = z_score(data, acc_modules, activity_choice, k)
    mag_outlier_idxs = z_score(data, mag_modules, activity_choice, k)
    gyro_outlier_idxs = z_score(data, gyro_modules, activity_choice, k)

    plot_zscore_outliers(data, acc_modules, "Acceleration", activity_choice, acc_outlier_idxs)
    plot_zscore_outliers(data, mag_modules, "Magnetic Field", activity_choice, mag_outlier_idxs)
    plot_zscore_outliers(data, gyro_modules, "Angular Velocity", activity_choice, gyro_outlier_idxs)

    #Exercicio 3.5

    ''' blablablabla'''

    #Exercicio 3.6 e 3.7

    n_clusters = int(input("\n--- K-Means Clustering ---\n\nChoose the number of clusters: (e.g., 2, 3, 4): "))
    print(f"\n--- Acceleration K-Means Results ---\n")
    acc_labels, acc_centers = kmeans(acc_modules, n_clusters)
    print(f"\n--- Magnetic Field K-Means Results ---\n")
    mag_labels, mag_centers = kmeans(mag_modules, n_clusters)
    print(f"\n--- Angular Velocity K-Means Results ---\n")
    gyro_labels, gyro_centers = kmeans(gyro_modules, n_clusters)

    activity_choice = int(input("\nPick an activity to highlight outliers (1-16): "))
    device_choice = int(input("Pick a device to highlight outliers (1-5): "))

    plot_clusters(data, activity_choice, device_choice, 
                                    acc_modules, gyro_modules, mag_modules, 
                                    n_clusters=n_clusters)

    #Exercicio 4.1

    print("\n--- Statistical Tests - Kruskal-Wallis ---")
    alpha=0.05
    for name, modules in variables_data:
        stat, p, pct_norm = choose_and_test_method(data, modules, alpha)

        print(f"\n- {name}")
        print(f"Normality (KS): {pct_norm}%  p > {alpha}")
        print(f"Statistic = {stat} | p = {p}")

    # Exercicio 4.2
    features, labels, feature_names = extract_features(
        data, acc_modules, mag_modules, gyro_modules,
        fs=50.0, window_duration=5.0, overlap_ratio=0.5)

    # Exercicio 4.3

    projected_data, components, explained_variance = pca(features, labels)

    # Exercicio 4.4

    analysis = pca_analysis(features, feature_names, variance_threshold=0.75, instant_index=5)

    # Exercicio 4.5

    fisher(features, labels, feature_names, top_features=10)
    relief(features, labels, feature_names=feature_names, top_features=10, n_neighbors=100)

