from log import print_and_log
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

# ---Exercise 4.1: K Nearest Neighbors---

def knn_classifier(training_dataset, type="features", scenario='a', k=3):
    """Returns a knn model trained on the selected scenario.
    
    Parameters
    ----------
    training_dataset : matrix
        Training dataset containing all scenarios datasets.
    scenario : str
        Scenario to use: 'a', 'b', or 'c'.
    k : int
        Number of neighbors to consider.
    
    Returns
    -------
    predict : function
        Function that takes a validation dataset and returns predictions."""

    scenario_indices = {'a': 0, 'b': 1, 'c': 2}

    if type == "features":
        training_scenario_data = training_dataset[scenario_indices[scenario]]
    elif type == "embeddings":
        training_scenario_data = training_dataset[scenario_indices[scenario] + 3]

    def predict(validation_dataset):
        prediction_labels = []

        if type == "features":
            validation_scenario_data = validation_dataset[scenario_indices[scenario]]
        elif type == "embeddings":
            validation_scenario_data = validation_dataset[scenario_indices[scenario] + 3]

        # For each sample in validation, find k nearest neighbors in training
        for sample in validation_scenario_data:
            distances = np.linalg.norm(training_scenario_data - sample, axis=1)
            nn_indices = np.argsort(distances)[:k]
            nn_labels = training_dataset[4][nn_indices, 0]
            most_common = Counter(nn_labels).most_common(1)[0][0]
            prediction_labels.append(most_common)
        return np.array(prediction_labels)
    return predict

def sklearn_knn_classifier(training_dataset, type="features", scenario='a', k=3):
    """Returns a knn model trained on the selected scenario using scikit-learn's KNeighborsClassifier.
    
    Parameters
    ----------
    training_dataset : matrix
        Training dataset containing all scenarios datasets.
    scenario : str
        Scenario to use: 'a', 'b', or 'c'.
    k : int
        Number of neighbors to consider.
    
    Returns
    -------
    predict : function
        Function that takes a validation dataset and returns predictions."""

    scenario_indices = {'a': 0, 'b': 1, 'c': 2}
    
    if type == "features":
        training_scenario_data = training_dataset[scenario_indices[scenario]]
    elif type == "embeddings":
        training_scenario_data = training_dataset[scenario_indices[scenario] + 3]

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(training_scenario_data, training_dataset[4][:, 0])

    def predict(validation_dataset):

        if type == "features":
            validation_scenario_data = validation_dataset[scenario_indices[scenario]]
        elif type == "embeddings":
            validation_scenario_data = validation_dataset[scenario_indices[scenario] + 3]

        return model.predict(validation_scenario_data)

    return predict

# --- Exercise 4.2: Classification Metrics ---

def validate_model(knn_model, validation_dataset, k=3, print_output=True):
    """Evaluate knn classifier on validation data.

    Parameters
    ----------
    knn_model : function
        KNN model returned by knn_classifier.
    validation_dataset : matrix 
        Validation dataset containing all scenarios datasets.

    Returns
    -------
    metrics : dict
        Dictionary containing confusion matrix, accuracy, precision, recall, F1 score."""

    # Get predictions and true labels
    predicted_labels = knn_model(validation_dataset)
    true_labels = validation_dataset[4][:, 0]
    predicted_labels = np.array(predicted_labels)

    # Compute metrics
    confusion = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    metrics = {
        'confusion_matrix': confusion,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    if print_output:

        print_and_log(f"\n--- Validation Metrics for k = {k} ---\n")

        # Print confusion matrix with labels on axes
        print_and_log("Confusion Matrix:\n")
        print_and_log("         PREDICTED")
        print_and_log("      ", end="")
        for lbl in range(1, 8):
            print_and_log(f"{lbl:>5}", end="")
        print_and_log()
        real_label = "REAL"
        for row_idx, row in enumerate(confusion):
            letter = real_label[row_idx] if row_idx < len(real_label) else " "
            print_and_log(f"  {letter}  {row_idx+1:>2} ", end="")
            for val in row:
                print_and_log(f"{val:>5}", end="")
            print_and_log()
            
        print_and_log(f"\nAccuracy:  {accuracy:.4f}")
        print_and_log(f"Precision: {precision:.4f}")
        print_and_log(f"Recall:    {recall:.4f}")
        print_and_log(f"F1 Score:  {f1:.4f}")

    return metrics