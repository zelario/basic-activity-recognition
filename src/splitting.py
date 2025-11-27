from sklearn.model_selection import train_test_split
import numpy as np
from features import relief
from features import compute_pca
from features import zscore_normalization

# --- Exercise 3.1: Mixed participant splitting ---

def mixed_splitting(features, embeddings, labels):
    """Split features, embeddings, and labels into training, validate, and test sets using mixed participant splitting.
    Ensures consistent splits across all representations and stratifies by the first label column.

    Parameters
    ----------
    features : np.ndarray, shape (n_windows, n_features)
        Feature matrix for all windows.
    embeddings : np.ndarray, shape (n_windows, n_embedding_features)
        Embeddings matrix for all windows.
    labels : np.ndarray, shape (n_windows, n_label_features)
        Labels array for all windows.

    Returns
    -------
    train_dataset : tuple
        (features, embeddings, labels) for training set.
    validate_dataset : tuple or None
        (features, embeddings, labels) for validate set, or None if validate=False.
    test_dataset : tuple
        (features, embeddings, labels) for test set."""

    # First split: Train+Val and Test
    train_val_features, test_features, train_val_embeddings, test_embeddings, train_val_labels, test_labels = train_test_split(
        features, embeddings, labels, test_size=0.2, random_state=None, stratify=labels[:, 1])

    # Second split: Train and Val
    train_features, val_features, train_embeddings, validate_embeddings, train_labels, validate_labels = train_test_split(
        train_val_features, train_val_embeddings, train_val_labels, test_size=0.2, random_state=None, stratify=train_val_labels[:, 1])
    
    train_dataset = np.array(train_features), np.array(train_embeddings), np.array(train_labels)
    validate_dataset = np.array(val_features), np.array(validate_embeddings), np.array(validate_labels)
    test_dataset = np.array(test_features), np.array(test_embeddings), np.array(test_labels)

    split = {"train": train_dataset, "validate": validate_dataset, "test": test_dataset}

    return split

# --- Exercise 3.2: Participant-based splitting ---

def participant_splitting(features, embeddings, labels, train_n=9, validate_n=3, test_n=3):
    """Split features, embeddings, and labels into training, validate, and test sets by participant groups.
    Ensures no data leakage between sets by assigning unique participants to each split.

    Parameters
    ----------
    features : np.ndarray, shape (n_windows, n_features)
        Feature matrix for all windows.
    embeddings : np.ndarray, shape (n_windows, n_embedding_features)
        Embeddings matrix for all windows.
    labels : np.ndarray, shape (n_windows, n_label_features)
        Labels array for all windows. Assumes participant ID is in column 1.
    train_n : int, optional
        Number of participants in training set (default=9).
    validate_n : int, optional
        Number of participants in validate set (default=3).
    test_n : int, optional
        Number of participants in test set (default=3).
    validate : bool, optional
        Whether to create a validate split (default=True).

    Returns
    -------
    train_dataset : tuple
        (features, embeddings, labels) for training set.
    validate_dataset : tuple or None
        (features, embeddings, labels) for validate set, or None if validate=False.
    test_dataset : tuple
        (features, embeddings, labels) for test set."""

    random = np.random.default_rng(None)
    participants = np.unique(labels[:, 1])
    participants_shuffled = random.permutation(participants)

    # Split participants into Train, validate, and Test
    train_participants = participants_shuffled[:train_n]
    validate_participants = participants_shuffled[train_n:train_n+validate_n]
    test_participants = participants_shuffled[train_n+validate_n:train_n+validate_n+test_n]

    # Create masks for each split
    train_mask = np.isin(labels[:, 1], train_participants)
    validate_mask = np.isin(labels[:, 1], validate_participants)
    test_mask = np.isin(labels[:, 1], test_participants)

    # Create datasets
    train_dataset = np.array(features[train_mask]), np.array(embeddings[train_mask]), np.array(labels[train_mask])
    validate_dataset = np.array(features[validate_mask]), np.array(embeddings[validate_mask]), np.array(labels[validate_mask])
    test_dataset = np.array(features[test_mask]), np.array(embeddings[test_mask]), np.array(labels[test_mask])

    split = {"train": train_dataset, "validate": validate_dataset, "test": test_dataset}

    return split

# --- Exercise 3.3: Discussion on splitting strategies ---

'''A estratégia de divisão dentro de cada participante consiste em separar os dados de cada participante em conjuntos de treino, validação e teste. 
Desta forma, o modelo é treinado, validado e testado com dados de todos os participantes, ou seja, ele tem sempre exemplos de cada pessoa durante o 
treinamento. Isso pode superestimar o desempenho do modelo, pois ele pode aprender padrões específicos de cada participante e, assim, ter facilidade 
para reconhecer dados semelhantes no conjunto de teste.

Já a estratégia de divisão de participantes separa os participantes em grupos distintos para treino, validação e teste. O modelo é treinado apenas 
com dados de um grupo de participantes e testado com dados de pessoas que ele nunca viu antes. Esta abordagem é mais rigorosa e representa melhor o 
cenário real de uso, onde o modelo precisa generalizar para novos participantes. Portanto, a divisão entre participantes fornece uma estimativa mais 
confiável do desempenho do modelo quando aplicado a dados de um novo participante desconhecido, pois avalia a capacidade de generalização e evita 
que o modelo se beneficie de padrões individuais presentes no treino.'''

# --- Exercise 3.4: Pipeline preparation ---

def prepare_pipeline(split):
    """Prepare three feature transformation scenarios for train, validate, and test sets:
      a) All features/embeddings (no transformation)
      b) PCA-reduced features (retain 90% variance, fit on train only)
      c) ReliefF-selected top 15 features (fit on train only)

    All transformations (PCA, ReliefF) are fit ONLY on the training set and applied to validate and test sets.

    Parameters
    ----------
    train_dataset : tuple
        (features, embeddings, labels) for training set.
    validate_dataset : tuple
        (features, embeddings, labels) for validate set.
    test_dataset : tuple
        (features, embeddings, labels) for test set.

    Returns
    -------
    pipeline : np.ndarray, shape (3, 5)
        Array containing for each split (train, val, test):
        [all_features, pca_features, relief_features, embeddings, labels]"""

    # === Scenario a: All features/embeddings ===

    # Unpack datasets
    train_features_all, train_embeddings_all, train_labels = split["train"]
    validate_features_all, validate_embeddings_all, validate_labels = split["validate"]
    test_features_all, test_embeddings_all, test_labels = split["test"]

    # === Scenario b: PCA-reduced features and embeddings (90% variance) ===

    # Normalize training features and embeddings
    normalized_train_features, feature_means, feature_stds = zscore_normalization(train_features_all, return_parameters=True)
    normalized_train_embeddings, embedding_means, embedding_stds = zscore_normalization(train_embeddings_all, return_parameters=True)

    # Compute PCA on normalized training features and embeddings
    train_features_pca, explained_variance_features, pca_object_features = compute_pca(normalized_train_features)
    train_embeddings_pca, explained_variance_embeddings, pca_object_embeddings = compute_pca(normalized_train_embeddings)

    # Determine number of components to retain 90% variance
    cumulative_features = np.cumsum(explained_variance_features)
    n_components_90_features = np.argmax(cumulative_features >= 0.9) + 1

    cumulative_embeddings = np.cumsum(explained_variance_embeddings)
    n_components_90_embeddings = np.argmax(cumulative_embeddings >= 0.9) + 1

    # Reduce to selected number of components
    train_features_pca = train_features_pca[:, :n_components_90_features]
    train_embeddings_pca = train_embeddings_pca[:, :n_components_90_embeddings]

    # Apply normalization to validate set
    normalized_validate_features = zscore_normalization(validate_features_all, mean_values=feature_means, std_values=feature_stds)
    normalized_validate_embeddings = zscore_normalization(validate_embeddings_all, mean_values=embedding_means, std_values=embedding_stds)

    # Project validate set using PCA fitted on training set
    validate_features_pca = compute_pca(normalized_validate_features, pca_object=pca_object_features)
    validate_embeddings_pca = compute_pca(normalized_validate_embeddings, pca_object=pca_object_embeddings)

    validate_features_pca = validate_features_pca[:, :n_components_90_features]
    validate_embeddings_pca = validate_embeddings_pca[:, :n_components_90_embeddings]

    # Apply normalization to test set
    normalized_test_features = zscore_normalization(test_features_all, mean_values=feature_means, std_values=feature_stds)
    normalized_test_embeddings = zscore_normalization(test_embeddings_all, mean_values=embedding_means, std_values=embedding_stds)

    # Project test set using PCA fitted on training set
    test_features_pca = compute_pca(normalized_test_features, pca_object=pca_object_features)
    test_embeddings_pca = compute_pca(normalized_test_embeddings, pca_object=pca_object_embeddings)

    test_features_pca = test_features_pca[:, :n_components_90_features]
    test_embeddings_pca = test_embeddings_pca[:, :n_components_90_embeddings]
    
    # === Scenario c: ReliefF-selected top 15 features ===

    # Select top 15 features using ReliefF on normalized training features and embeddings
    top_15_features_indices = relief(normalized_train_features, train_labels, top_n=15, print_output=False)
    top_15_embeddings_indices = relief(normalized_train_embeddings, train_labels, top_n=15, print_output=False)

    # Apply feature selection to all splits
    train_features_relief = normalized_train_features[:, top_15_features_indices]
    validate_features_relief = normalized_validate_features[:, top_15_features_indices]
    test_features_relief = normalized_test_features[:, top_15_features_indices]

    train_embeddings_relief = normalized_train_embeddings[:, top_15_embeddings_indices]
    validate_embeddings_relief = normalized_validate_embeddings[:, top_15_embeddings_indices]
    test_embeddings_relief = normalized_test_embeddings[:, top_15_embeddings_indices]

    # Prepare pipeline dictionary
    pipeline = {
        "train": {
            "features": {
                "a": np.array(train_features_all),
                "b": np.array(train_features_pca),
                "c": np.array(train_features_relief)
            },
            "embeddings": {
                "a": np.array(train_embeddings_all),
                "b": np.array(train_embeddings_pca),
                "c": np.array(train_embeddings_relief)
            },
            "labels": np.array(train_labels)
        },
        "validate": {
            "features": {
                "a": np.array(validate_features_all),
                "b": np.array(validate_features_pca),
                "c": np.array(validate_features_relief)
            },
            "embeddings": {
                "a": np.array(validate_embeddings_all),
                "b": np.array(validate_embeddings_pca),
                "c": np.array(validate_embeddings_relief)
            },
            "labels": np.array(validate_labels)
        },
        "test": {
            "features": {
                "a": np.array(test_features_all),
                "b": np.array(test_features_pca),
                "c": np.array(test_features_relief)
            },
            "embeddings": {
                "a": np.array(test_embeddings_all),
                "b": np.array(test_embeddings_pca),
                "c": np.array(test_embeddings_relief)
            },
            "labels": np.array(test_labels)
        }
    }

    return pipeline