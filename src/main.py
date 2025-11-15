from load import *
from outliers import *
from features import *
from augmentation import *

def partA ():

    # --- Exercise 1: Load Part Data ---

    '''part_data = load_part_data(0)'''

    # --- Exercise 2: Load Full Data ---

    data = load_data()

    # --- Exercise 3.1: Variable Modules ---

    variables_modules = compute_modules(data) 
    boxplot_modules(data, variables_modules)

    # --- Exercise 3.2: Outlier Densities via IQR ---

    outlier_density_iqr(data, variables_modules)

    # --- Exercise 3.3 and 3.4: Outlier Detection via Z-Score ---

    k=3
    outlier_density_z_score(data, variables_modules, k)
    plot_zscore_outliers(data, variables_modules, k)

    # --- Exercise 3.6 and 3.7: Clustering ---

    n_clusters=3
    kmeans(variables_modules, n_clusters=n_clusters)
    plot_kmeans_clusters(data, variables_modules, n_clusters=n_clusters)
    plot_dbscan_clusters(data, variables_modules)

    # --- Exercise 4.1: Statistical Tests ---

    alpha = 0.05
    normality_and_significance(data, variables_modules, alpha)

    # --- Exercise 4.2: Feature Extraction ---
    
    features, labels, feature_names = extract_features(data, variables_modules, fs=51.5, window_duration=5.0, overlap_ratio=0.5)

    # --- Exercise 4.3: PCA ---

    n_components = 36
    pca, explained_variance_ratio = compute_pca(features, n_components)

    # --- Exercise 4.4: PCA Analysis ---

    analyse_pca(explained_variance_ratio)

    # --- Exercise 4.5: Fisher Scores and ReliefF ---

    fisher(features, labels, feature_names)
    relief(features, labels, feature_names=feature_names, n_neighbors=100)

def partB ():

    # --- Pre game data loading and preprocessing ---

    features, pca, scores, labels = reload_data()
    features, pca, labels = discard_activities(features, pca, labels)

    # --- Exercise 1.1: Analyse sample balance ---

    label_count = analyze_activity_balance(labels)

    # --- Exercise 1.2: Data Augmentation with SMOTE ---

    features, pca, labels, synthetic_rows = augment_activity_data(features, pca, labels)

    # --- Exercise 1.3: Visualize Synthetic vs Real Samples ---

    plot_synthetic_vs_real(features, labels, synthetic_rows)

if __name__ == "__main__":

    #--- Run Part A Exercises ---

    '''partA()'''

    #--- Run Part B Exercises ---

    partB()