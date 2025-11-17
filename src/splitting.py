from sklearn.model_selection import train_test_split
import numpy as np
from features import relief
from features import compute_pca

# --- Exercise 3.1: Mixed participant splitting ---

def mixed_splitting(features, embeddings, labels):
    """Splits features and embeddings into training, validation, and test sets
    while ensuring that the splits are consistent across both representations.

    Parameters
    ----------
    features : matrix, shape (n_windows, n_features)
        The feature matrix.
    embeddings : matrix, shape (n_windows, n_embedding_features)
        The embeddings matrix.
    labels : matrix, shape (n_windows, n_label_features)
        The labels array.

    Returns
    -------
    features_split : tuple of matrixes
        Training, validation, and test sets for features.
    embeddings_split : tuple of matrixes
        Training, validation, and test sets for embeddings.
    labels_split : tuple of matrixes
        Training, validation, and test labels."""

    # First split: Train+Val and Test
    train_val_features, test_features, train_val_embeddings, test_embeddings, train_val_labels, test_labels = train_test_split(
        features, embeddings, labels, test_size=0.2, random_state=None, stratify=labels[:, 0]
    )

    # Second split: Train and Val
    train_features, val_features, train_embeddings, validation_embeddings, train_labels, validation_labels = train_test_split(
        train_val_features, train_val_embeddings, train_val_labels, test_size=0.25, random_state=None, stratify=train_val_labels[:, 0]
    )

    # Combine splits
    train_dataset = train_features, train_embeddings, train_labels
    validation_dataset = val_features, validation_embeddings, validation_labels
    test_dataset = test_features, test_embeddings, test_labels

    return np.array(train_dataset), np.array(validation_dataset), np.array(test_dataset)

# --- Exercise 3.2: Participant-based splitting ---

def participant_splitting(features, embeddings, labels, train_n=9, validation_n=3, test_n=3):
    """Splits features and embeddings into training, validation, and test sets
    based on participants to ensure no data leakage between sets.

    Parameters
    ----------
    features : matrix, shape (n_windows, n_features)
        The feature matrix.
    embeddings : matrix, shape (n_windows, n_embedding_features)
        The embeddings matrix.
    labels : matrix, shape (n_windows, n_label_features)
        The labels array.
    train_subjects : int
        Number of subjects for training set.
    val_subjects : int
        Number of subjects for validation set.
    test_subjects : int
        Number of subjects for test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    features_split : tuple of matrixes
        Training, validation, and test sets for features.
    embeddings_split : tuple of matrixes
        Training, validation, and test sets for embeddings.
    labels_split : tuple of matrixes
        Training, validation, and test labels."""

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
    train_dataset = features[train_mask], embeddings[train_mask], labels[train_mask]
    validation_dataset = features[validation_mask], embeddings[validation_mask], labels[validation_mask]
    test_dataset = features[test_mask], embeddings[test_mask], labels[test_mask]

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
    """Prepares three scenarios for the dataset:
    a) All features/embeddings
    b) PCA-reduced features (90% variance)
    c) ReliefF-selected top 15 features
    All transformations are fit ONLY on the training set and applied to val/test.
    Returns three lists for train, val, test: each is (scenario_a, scenario_b, scenario_c)"""

    # Unpack datasets
    train_features, train_embeddings, train_labels = train_dataset
    validation_features, validation_embeddings, validation_labels = validation_dataset
    test_features, test_embeddings, test_labels = test_dataset

    # Scenario b: PCA-reduced features (90% variance)
    train_pca_scores, explained_variance_ratio = compute_pca(train_features, n_components=None)
    cumulative = np.cumsum(explained_variance_ratio)
    n_components_90 = np.argmax(cumulative >= 0.9) + 1
    train_features_pca = train_pca_scores[:, :n_components_90]

    validation_pca_scores, explained_variance_ratio = compute_pca(validation_features, n_components=n_components_90)
    cumulative = np.cumsum(explained_variance_ratio)
    n_components_90 = np.argmax(cumulative >= 0.9) + 1
    validation_features_pca = validation_pca_scores[:, :n_components_90]

    test_pca_scores, explained_variance_ratio = compute_pca(test_features, n_components=n_components_90)
    cumulative = np.cumsum(explained_variance_ratio)
    n_components_90 = np.argmax(cumulative >= 0.9) + 1
    test_features_pca = test_pca_scores[:, :n_components_90]

    # Scenario c: ReliefF-selected top 15 features
    top_15_indices = relief(train_features, train_labels, n_features=15)
    train_features_relief = train_features[:, top_15_indices]
    validation_features_relief = validation_features[:, top_15_indices]
    test_features_relief = test_features[:, top_15_indices]

    pipeline = [[train_features, train_features_pca, train_features_relief, train_embeddings, train_labels],
                [validation_features, validation_features_pca, validation_features_relief, validation_embeddings, validation_labels],
                [test_features, test_features_pca, test_features_relief, test_embeddings, test_labels]]
    
    return np.array(pipeline)

    




