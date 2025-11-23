from evaluation import *
from log import *
from load import *
from outliers import *
from features import *
from augmentation import *
from embeddings import *
from splitting import *
from model_learning import *


def partA ():

    '''# --- Exercise 1: Load Part Dataset ---

    part_dataset = load_part_dataset(0)'''

    # --- Exercise 2: Load Full Dataset ---

    dataset = load_data()

    # --- Exercise 3.1: Variable Modules ---

    variables_modules = compute_modules(dataset) 

    '''boxplot_modules(dataset, variables_modules)

    # --- Exercise 3.2: Outlier Densities via IQR ---

    outlier_density_iqr(dataset, variables_modules)

    # --- Exercise 3.3 and 3.4: Outlier Detection via Z-Score ---

    k=3
    outlier_density_z_score(dataset, variables_modules, k)
    plot_zscore_outliers(dataset, variables_modules, k)

    # --- Exercise 3.6 and 3.7: Clustering ---

    n_clusters=3
    kmeans(variables_modules, n_clusters=n_clusters)
    plot_kmeans_clusters(dataset, variables_modules, n_clusters=n_clusters)
    plot_dbscan_clusters(dataset, variables_modules)

    # --- Exercise 4.1: Statistical Tests ---

    alpha = 0.05
    normality_and_significance(dataset, variables_modules, alpha)'''

    # --- Exercise 4.2: Feature Extraction ---
    
    features, labels = extract_features(dataset, variables_modules, window_duration=5.0, overlap_ratio=0.5)
    features = zscore_normalization(features)

    # --- Exercise 4.3: PCA ---

    pca, explained_variance_ratio, _ = compute_pca(features)

    '''# --- Exercise 4.4: PCA Analysis ---

    analyse_pca(explained_variance_ratio)

    # --- Exercise 4.5: Fisher and ReliefF ---

    fisher(features, labels)
    relief(features, labels, n_neighbors=100)'''

def partB ():

    # --- Pre game dataset loading and preprocessing ---

    dataset, features, labels = reload_data()
    features, labels = discard_activities(features=features, labels=labels)
    
    '''# --- Exercise 1.1: Analyse sample balance ---

    analyze_activity_balance(labels)

    # --- Exercise 1.2: Data Augmentation with SMOTE ---

    synthetic_features = augment_activity_data(features, labels)

    # --- Exercise 1.3: Visualize Synthetic vs Real Samples ---

    plot_synthetic_vs_real(features, labels, synthetic_features)'''

    # --- Exercise 2.1: Embeddings ---

    embeddings, embedding_labels = compute_embeddings(dataset, fs=51.5, window_duration=5.0, overlap_ratio=0.5)
    embeddings, embedding_labels = discard_activities(embeddings=embeddings, labels=embedding_labels)

    check_pairing(embeddings, features, labels, embedding_labels)

    '''# --- Exercise 3.1: Mixed splitting ---

    train_dataset, validation_dataset, test_dataset = mixed_splitting(features, embeddings, labels)

    # --- Exercise 3.2: Participant-based splitting ---

    #train_dataset, validation_dataset, test_dataset = participant_splitting(features, embeddings, labels)

    # --- Exercise 3.4: Pipeline training and evaluation ---

    pipeline = prepare_pipeline(train_dataset, validation_dataset, test_dataset)

    # --- Exercise 4.1: knn classifier ---

    k=3
    #knn_model = my_knn_classifier(pipeline[0], scenario='a', k=k)
    knn_model = sklearn_knn_classifier(pipeline[0], scenario='a', k=k)
    metrics = validate_model(knn_model, pipeline[1], k=k)'''

    # --- Exercise 5.1: Hyperparameter Tuning ---

    hyperparemeter_tuning(features, embeddings, labels, k_values=[1, 3, 5, 7, 11, 13, 17, 19], n_splits=10)


if __name__ == "__main__":

    clear_and_print(f"\n============================= START AT {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} =============================")

    #--- Run Part A Exercises ---

    #partA()

    #--- Run Part B Exercises ---

    partB()

    print_and_log(f"\n============================= END AT {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} =============================\n")