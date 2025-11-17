from sklearn.model_selection import train_test_split
import numpy as np

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
    features_split = (train_features, val_features, test_features)
    embeddings_split = (train_embeddings, validation_embeddings, test_embeddings)
    labels_split = (train_labels, validation_labels, test_labels)

    return features_split, embeddings_split, labels_split

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
    features_split = (features[train_mask], features[validation_mask], features[test_mask])
    embeddings_split = (embeddings[train_mask], embeddings[validation_mask], embeddings[test_mask])
    labels_split = (labels[train_mask], labels[validation_mask], labels[test_mask])

    return features_split, embeddings_split, labels_split

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

def prepare_pipeline(features, embeddings, labels, splitting_strategy):
    """Prepares the data pipeline for model training and evaluation, including
    data splitting and scenario preparation.

    Parameters
    ----------
    features : matrix, shape (n_windows, n_features)
        The feature matrix.
    embeddings : matrix, shape (n_windows, n_embedding_features)
        The embeddings matrix.
    labels : matrix, shape (n_windows, n_label_features)
        The labels array.
    splitting_strategy : str
        The splitting strategy to use ('mixed' or 'participant').

    Returns
    -------
    pipeline : dict
        A dictionary containing the train, validation, and test sets for features,
        embeddings, and labels, as well as the selected scenarios."""

    # Split data
    if splitting_strategy == 'mixed':
        features_split, embeddings_split, labels_split = mixed_splitting(features, embeddings, labels)
    elif splitting_strategy == 'participant':
        features_split, embeddings_split, labels_split = participant_splitting(features, embeddings, labels)
    else:
        raise ValueError("Invalid splitting strategy. Choose 'mixed' or 'participant'.")

    # Prepare scenarios
    train_features, val_features, test_features = features_split
    train_embeddings, val_embeddings, test_embeddings = embeddings_split
    train_labels, val_labels, test_labels = labels_split

    # Feature names assumed to be available as a global variable
    global feature_names
    feature_names = np.array([f'feature_{i}' for i in range(train_features.shape[1])])

    scenarios = prepare_scenarios(train_features, val_features, test_features, train_labels, val_labels, test_labels, feature_names)

    pipeline = {
        'features': features_split,
        'embeddings': embeddings_split,
        'labels': labels_split,
        'scenarios': scenarios
    }

    return pipeline

def prepare_scenarios(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """
    Prepares three scenarios for the dataset:
    a) All features
    b) PCA-reduced features (90% variance)
    c) ReliefF-selected top 15 features
    All transformations are fit ONLY on the training set and applied to val/test.
    Returns a dict with keys 'all', 'pca', 'relief' and values as tuples (train, val, test, selected_feature_names)
    """
    from sklearn.decomposition import PCA
    # ReliefF implementation from your features.py
    from features import relief
    import numpy as np

    scenarios = {}

    # a) All features
    scenarios['all'] = (X_train, X_val, X_test, feature_names)

    # b) PCA-reduced features (fit on train, transform val/test)
    pca = PCA(n_components=None)
    pca.fit(X_train)
    # Find number of components for 90% variance
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = np.argmax(cumulative >= 0.9) + 1
    pca_90 = PCA(n_components=n_components_90)
    pca_90.fit(X_train)
    X_train_pca = pca_90.transform(X_train)
    X_val_pca = pca_90.transform(X_val)
    X_test_pca = pca_90.transform(X_test)
    scenarios['pca'] = (X_train_pca, X_val_pca, X_test_pca, [f'PC{i+1}' for i in range(n_components_90)])

    # c) ReliefF-selected top 15 features (fit on train, select/transform val/test)
    top15 = relief(X_train, y_train, feature_names)[:15]
    top15_idx = [feature_names.index(name) for name in top15]
    X_train_relief = X_train[:, top15_idx]
    X_val_relief = X_val[:, top15_idx]
    X_test_relief = X_test[:, top15_idx]
    scenarios['relief'] = (X_train_relief, X_val_relief, X_test_relief, top15)

    return scenarios
