from log import print_and_log
from model_learning import sklearn_knn_classifier, validate_model
from splitting import mixed_splitting, participant_splitting, prepare_pipeline
import numpy as np

def hyperparemeter_tuning(features, embeddings, labels, k_values=[1], n_splits=1):

    # Dictionary to store best k and its accuracy for each scenario and splitting method
    best_k_counts = {
        (method, type, scenario, k): [] for method in ['mixed', 'participant'] for type in ['features', 'embeddings'] for scenario in ['a', 'b', 'c'] for k in k_values
    }

    print_and_log(f"\n--- Hyperparameter Tuning over {n_splits} different splits ---\n", path="log/hyperparameter_tuning.log")

    # For each splitting method
    for method in ['mixed', 'participant']:

        splits = [mixed_splitting(features, embeddings, labels) for _ in range(n_splits)] if method == 'mixed' else [participant_splitting(features, embeddings, labels) for _ in range(n_splits)]

        # For each data type
        for type in ['features', 'embeddings']:

            # For each scenario
            for scenario in ['a', 'b', 'c']:

                # For n_splits or number of splits
                for i, split in enumerate(splits):

                    # Prepare dataset
                    pipeline = prepare_pipeline(split[0], split[1], split[2])

                    # For each k value
                    for k in k_values:

                        # Create KNN model
                        knn_model = sklearn_knn_classifier(pipeline[0], scenario=scenario, type=type, k=k)

                        # Validate model
                        metrics = validate_model(knn_model, pipeline[1], k=k, scenario=scenario, type=type, method=method)
                        accuracy = metrics['accuracy']

                        # Store the k and its accuracy
                        best_k_counts[(method, type, scenario, k)].append(accuracy)
                        print_and_log(f"Method: {method}, Type: {type}, Scenario: {scenario}, Split= {i+1}, k= {k}, Accuracy: {accuracy:.4f}", path="log/hyperparameter_tuning.log")

    print_and_log(f"\n--- Best k for each method, type of data and scenario across {n_splits} splits ---", path="log/hyperparameter_tuning.log")

    best_k={}

    # Group keys by (method, type, scenario)
    grouped = {}
    for key in best_k_counts:
        method, type, scenario, k = key
        group_key = (method, type, scenario)
        if group_key not in grouped:
            grouped[group_key] = {}
        grouped[group_key][k] = best_k_counts[key]

    for group_key in grouped:
        method, type, scenario = group_key
        k_acc_dict = grouped[group_key]
        k_mean_acc = {k: (sum(accs)/len(accs) if accs else 0) for k, accs in k_acc_dict.items()}
        best_k_value = max(k_mean_acc, key=lambda k: k_mean_acc[k])
        best_mean_accuracy = k_mean_acc[best_k_value]
        print_and_log(f"Method: {method}, Type: {type}, Scenario: {scenario} | Best k = {best_k_value}, Mean Accuracy = {best_mean_accuracy:.4f}")
        best_k[(method, type, scenario)] = (best_k_value, best_mean_accuracy)

    np.save('npy/best_k.npy', best_k)
    return best_k

def models_evaluation(features, embeddings, labels, best_k, n_splits=1):

    # List to store all metrics rows
    metrics = []

    print_and_log(f"\n--- Models Evaluation over {n_splits} splits for each model ---\n", path="log/hyperparameter_tuning.log")

    # For each splitting method
    for method in ['mixed', 'participant']:

        splits = [mixed_splitting(features, embeddings, labels, validate=False) for _ in range(n_splits)] if method == 'mixed' else [participant_splitting(features, embeddings, labels, validate=False) for _ in range(n_splits)]
        
        # For each data type
        for type in ['features', 'embeddings']:

            # For each scenario
            for scenario in ['a', 'b', 'c']:

                k = best_k.get(method, type, scenario)[0]

                # For n_splits
                for i, split in enumerate(splits):

                    pipeline = prepare_pipeline(split[0], split[1], split[2])

                    knn_model = sklearn_knn_classifier(pipeline[0], scenario=scenario, type=type, k=k)

                    iteration_metrics = validate_model(knn_model, pipeline[2], k=k, scenario=scenario, type=type, method=method)

                    metrics.append([
                        method, type, scenario, i+1, k,
                        iteration_metrics['accuracy'], iteration_metrics['precision'], iteration_metrics['recall'], iteration_metrics['f1_score']
                    ])
                    print_and_log(f"Method: {method}, Type: {type}, Scenario: {scenario}, Split= {i+1}, k= {k}, Accuracy: {iteration_metrics['accuracy']:.4f}", path="log/hyperparameter_tuning.log")

    metrics = np.array(metrics, dtype=object)
    np.save('npy/metrics.npy', metrics)

    # --- Summary printing ---
    print_and_log(f"\n--- Best k for each method, type of data and scenario across {n_splits} splits ---\n")
