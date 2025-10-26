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

    print("\n--- Boxplot de variáveis ---\n")

    data = load_data_polars()
    acc_modules = variable_module(data, "Acceleration")
    mag_modules = variable_module(data, "Magnetic Field")
    gyro_modules = variable_module(data, "Angular Velocity")

    variables_data = [
		("Aceleração (|Acc|)", acc_modules),
		("Velocidade Angular (|Gyro|)", gyro_modules),
		("Campo Magnético (|Mag|)", mag_modules)
	]

    device_choice = int(input("Escolha o dispositivo (1-5): "))
    for name, modules in variables_data:
        boxplot_variable(data, modules, name, device_choice)

    #Exercicio 3.2

    outlier_density(data, acc_modules)

    #Exercicio 3.3 e 3.4

    print("\n--- Detecção de Outliers via Z-Score ---\n")

    k=3

    activity_choice = int(input("Escolha a atividade (1-16): "))

    acc_outlier_idxs = z_score(data, acc_modules, activity_choice, k)
    mag_outlier_idxs = z_score(data, mag_modules, activity_choice, k)
    gyro_outlier_idxs = z_score(data, gyro_modules, activity_choice, k)

    plot_zscore_outliers(data, acc_modules, "Acceleration", activity_choice, acc_outlier_idxs)
    plot_zscore_outliers(data, mag_modules, "Magnetic Field", activity_choice, mag_outlier_idxs)
    plot_zscore_outliers(data, gyro_modules, "Angular Velocity", activity_choice, gyro_outlier_idxs)

    #Exercicio 3.5

    ''' blablablabla'''

    #Exercicio 3.6 e 3.7

    n_clusters = int(input("\n--- K-Means Clustering ---\nEscolha o número de clusters (e.g., 2, 3, 4): "))
    print(f"\n--- Acceleration K-Means Results ---")
    acc_labels, acc_centers = kmeans(acc_modules, n_clusters)
    print(f"\n--- Magnetic Field K-Means Results ---")
    mag_labels, mag_centers = kmeans(mag_modules, n_clusters)
    print(f"\n--- Angular Velocity K-Means Results ---")
    gyro_labels, gyro_centers = kmeans(gyro_modules, n_clusters)

    activity_choice = int(input("Escolha a atividade para destacar outliers (1-16): "))

    plot_clusters_activity_outliers(
        data, acc_modules, gyro_modules, mag_modules,
        activity_choice, acc_outlier_idxs, gyro_outlier_idxs, mag_outlier_idxs
    )

    #Exercicio 4.1
    
    print("\n--- Analise Estatistica ---\n")
    alpha=0.05
    for name, modules in variables_data:
        method, stat, p, pct_norm = choose_and_test_method(data, modules, alpha)

        print(f"\n--- {name} ---")
        print(f"Normalidade (KS): {pct_norm}% dos grupos têm p > {alpha}")
        print(f"Método aplicado: {method}")
        print(f"Estatística = {stat} | p = {p}")

        if p < alpha:
            print("-Diferenças significativas entre atividades.")
        else:
            print("-Sem diferenças significativas entre atividades.")

    # Exercicio 4.2
    features, labels, feature_names = extract_features(
        data, acc_modules, mag_modules, gyro_modules,
        fs=50.0, window_duration=5.0, overlap_ratio=0.5, normalize_zscore=True
    )

    # Exercicio 4.3

    projected_data, components, explained_variance = pca(features, labels, n_components=2)

    # Exercicio 4.4

    analysis = pca_analysis(features, feature_names, variance_threshold=0.75, instant_index=5)

    # Exercicio 4.5
    fisher_relief = fisher_and_relief(features, labels, feature_names, top_features=10)

