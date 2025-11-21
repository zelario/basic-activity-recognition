from log import print_and_log
from model_learning import sklearn_knn_classifier, validate_model
from splitting import mixed_splitting, participant_splitting, prepare_pipeline

def hyperparemeter_tuning(features, embeddings, labels, k_values=[1], n_splits=1):

    # Dictionary to store best k and its accuracy for each scenario and splitting method
    best_k_counts = {
        (method, type, scenario, k): [] for method in ['mixed', 'participant'] for type in ['features', 'embeddings'] for scenario in ['a', 'b', 'c'] for k in k_values
    }

    print_and_log(f"\n--- Hyperparameter Tuning over {n_splits} repeats ---\n")

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
                        metrics = validate_model(knn_model, pipeline[1], k, print_output=False)
                        accuracy = metrics['accuracy']

                        # Store the k and its accuracy
                        best_k_counts[(method, type, scenario, k)].append(accuracy)
                        print_and_log(f"Method: {method}, Type: {type}, Scenario: {scenario}, Split= {i+1}, k= {k}, Accuracy: {accuracy:.4f}")

    best_overall_mean_accuracy = 0
    best_overall = None

    print_and_log(f"\n--- Best k for each method, type of data and scenario across {n_splits} splits ---")

    # Aggregate and report best k for each (method, type, scenario)
    best_overall_mean_accuracy = 0
    best_overall = None

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
        best_k = max(k_mean_acc, key=lambda k: k_mean_acc[k])
        best_mean_accuracy = k_mean_acc[best_k]
        print_and_log(f"\nMethod: {method}, Type: {type}, Scenario: {scenario} | Best k = {best_k}, Mean Accuracy = {best_mean_accuracy:.4f}")

        # Track best overall mean accuracy
        if best_mean_accuracy > best_overall_mean_accuracy:
            best_overall_mean_accuracy = best_mean_accuracy
            best_overall = (method, type, scenario, best_k)

    # Report best overall
    method, type, scenario, best_k = best_overall
    print_and_log(f"\n--- Best overall mean accuracy ---\n")
    print_and_log(f"Best mean accuracy: {best_overall_mean_accuracy:.4f} (k={best_k}, method={method}, type={type}, scenario={scenario})")