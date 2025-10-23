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

    device_choice = int(input("Escolha o dispositivo (1-5): "))
    boxplot_variable(data, acc_modules, "Acceleration", device_choice)
    boxplot_variable(data, mag_modules, "Magnetic Field", device_choice)
    boxplot_variable(data, gyro_modules, "Angular Velocity", device_choice)

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

    plot_clusters(acc_centers, gyro_centers, mag_centers)

    #Exercicio 4.1
    
    print("\n--- Statistical Analysis per Activity ---\n")
    alpha=0.05
    variaveis = [
		("Aceleração (|Acc|)", acc_modules),
		("Velocidade Angular (|Gyro|)", gyro_modules),
		("Campo Magnético (|Mag|)", mag_modules)
	]
    for nome, dados in variaveis:
        metodo, stat, p, pct_norm = choose_and_test(data, dados, alpha)
        
        print(f"\n--- {nome} ---")
        print(f"Normalidade (KS): {pct_norm}% dos grupos têm p > {alpha}")
        print(f"Método aplicado: {metodo}")
        print(f"Estatística = {stat} | p = {p}")
        
        if p < alpha:
            print("-Diferenças significativas entre atividades.")
        else:
            print("-Sem diferenças significativas entre atividades.")
            
     #Exercicio 4.2
    
    X, y, feat_names = run_42_extract_features(
    data, acc_modules, mag_modules, gyro_modules, fs=50.0, win_s=5.0, overlap=0.5, zscore=True
