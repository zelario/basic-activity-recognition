from sklearn.model_selection import train_test_split
import numpy as np
from features import relief
from features import compute_pca
from features import zscore_normalization

# --- Exercise 3.1: Mixed participant splitting ---

def mixed_splitting(features, embeddings, labels):
    """Split features, embeddings, and labels into training, validation, and test sets using mixed participant splitting.
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
    validation_dataset : tuple
        (features, embeddings, labels) for validation set.
    test_dataset : tuple
        (features, embeddings, labels) for test set."""

    # First split: Train+Val and Test
    train_val_features, test_features, train_val_embeddings, test_embeddings, train_val_labels, test_labels = train_test_split(
        features, embeddings, labels, test_size=0.2, random_state=None, stratify=labels[:, 0]
    )

    # Second split: Train and Val
    train_features, val_features, train_embeddings, validation_embeddings, train_labels, validation_labels = train_test_split(
        train_val_features, train_val_embeddings, train_val_labels, test_size=0.25, random_state=None, stratify=train_val_labels[:, 0]
    )

    # Combine splits
    train_dataset = np.array(train_features), np.array(train_embeddings), np.array(train_labels)
    validation_dataset = np.array(val_features), np.array(validation_embeddings), np.array(validation_labels)
    test_dataset = np.array(test_features), np.array(test_embeddings), np.array(test_labels)

    return train_dataset, validation_dataset, test_dataset

# --- Exercise 3.2: Participant-based splitting ---

def participant_splitting(features, embeddings, labels, train_n=9, validation_n=3, test_n=3):
    """Split features, embeddings, and labels into training, validation, and test sets by participant groups.
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
    validation_n : int, optional
        Number of participants in validation set (default=3).
    test_n : int, optional
        Number of participants in test set (default=3).

    Returns
    -------
    train_dataset : tuple
        (features, embeddings, labels) for training set.
    validation_dataset : tuple
        (features, embeddings, labels) for validation set.
    test_dataset : tuple
        (features, embeddings, labels) for test set."""

    random = np.random.default_rng(None)

    # Get unique subject IDs
    participants = np.unique(labels[:, 1])

    # Shuffle subjects
    participants_shuffled = random.permutation(participants)

    # Assign subjects to splits
    train_participants = participants_shuffled[:train_n]
    validation_participants = participants_shuffled[train_n:train_n+validation_n]
    test_participants = participants_shuffled[train_n+validation_n:train_n+validation_n+test_n]
    
    # Create masks
    train_mask = np.isin(labels[:, 1], train_participants)
    validation_mask = np.isin(labels[:, 1], validation_participants)
    test_mask = np.isin(labels[:, 1], test_participants)

    # Combine splits
    train_dataset = np.array(features[train_mask])  , np.array(embeddings[train_mask]), np.array(labels[train_mask])
    validation_dataset = np.array(features[validation_mask]), np.array(embeddings[validation_mask]), np.array(labels[validation_mask])
    test_dataset = np.array(features[test_mask]), np.array(embeddings[test_mask]), np.array(labels[test_mask])

    return train_dataset, validation_dataset, test_dataset

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

def prepare_pipeline(train_dataset, validation_dataset, test_dataset):
    """Prepare three feature transformation scenarios for train, validation, and test sets:
      a) All features/embeddings (no transformation)
      b) PCA-reduced features (retain 90% variance, fit on train only)
      c) ReliefF-selected top 15 features (fit on train only)

    All transformations (PCA, ReliefF) are fit ONLY on the training set and applied to validation and test sets.

    Parameters
    ----------
    train_dataset : tuple
        (features, embeddings, labels) for training set.
    validation_dataset : tuple
        (features, embeddings, labels) for validation set.
    test_dataset : tuple
        (features, embeddings, labels) for test set.

    Returns
    -------
    pipeline : np.ndarray, shape (3, 5)
        Array containing for each split (train, val, test):
        [all_features, pca_features, relief_features, embeddings, labels]"""

    # Unpack datasets and set Scenario a: All features/embeddings
    train_features_all, train_embeddings, train_labels = train_dataset
    validation_features_all, validation_embeddings, validation_labels = validation_dataset
    test_features_all, test_embeddings, test_labels = test_dataset

    # Scenario b: PCA-reduced features (90% variance)
    train_features_pca, explained_variance_ratio = compute_pca(train_features_all, n_components=None)
    cumulative = np.cumsum(explained_variance_ratio)
    n_components_90 = np.argmax(cumulative >= 0.9) + 1
    train_features_pca = train_features_pca[:, :n_components_90]

    validation_features_pca, _ = compute_pca(validation_features_all, n_components=n_components_90)
    test_features_pca, _ = compute_pca(test_features_all, n_components=n_components_90)

    # Scenario c: ReliefF-selected top 15 features
    normalized_train_features = zscore_normalization(train_features_all)
    top_15_indices = relief(normalized_train_features, train_labels, top_n=15, print_output=False)

    train_features_relief = train_features_all[:, top_15_indices]
    validation_features_relief = validation_features_all[:, top_15_indices]
    test_features_relief = test_features_all[:, top_15_indices]

    pipeline = [
        [
            np.array(train_features_all),
            np.array(train_features_pca),
            np.array(train_features_relief),
            np.array(train_embeddings),
            np.array(train_labels)
        ],
        [
            np.array(validation_features_all),
            np.array(validation_features_pca),
            np.array(validation_features_relief),
            np.array(validation_embeddings),
            np.array(validation_labels)
        ],
        [
            np.array(test_features_all),
            np.array(test_features_pca),
            np.array(test_features_relief),
            np.array(test_embeddings),
            np.array(test_labels)
        ]
    ]
    return pipeline