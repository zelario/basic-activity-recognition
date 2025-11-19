from log import print_and_log
from model_learning import sklearn_knn_classifier, validate_model
from splitting import mixed_splitting, participant_splitting, prepare_pipeline

def hyperparemeter_tuning(features, embeddings, labels, k_values, n_repeats=1):

    # Dictionary to store best k and its accuracy for each scenario and splitting method
    best_k_counts = {
        (split, scenario): [] for split in ['mixed', 'participant'] for scenario in ['a', 'b', 'c']
    }

    print_and_log(f"\n--- Hyperparameter Tuning over {n_repeats} repeats ---\n")

    # For each splitting method
    for split in ['mixed', 'participant']:

        # For each scenario
        for scenario in ['a', 'b', 'c']:

            # For each k value
            for k in k_values:
                accuracies = []

                # Compute accuracy over n_repeats
                for repeat in range(n_repeats):

                    # Perform splitting based on the current method
                    if split == 'mixed':
                        train_dataset, validation_dataset, test_dataset = mixed_splitting(features, embeddings, labels)
                    else:
                        train_dataset, validation_dataset, test_dataset = participant_splitting(features, embeddings, labels)
                    
                    pipeline = prepare_pipeline(train_dataset, validation_dataset, test_dataset)

                    knn_model = sklearn_knn_classifier(pipeline[0], scenario=scenario, k=k)
                    metrics = validate_model(knn_model, pipeline[1], k=k, print_output=False)

                    # Collect accuracy
                    accuracy = metrics['accuracy']
                    accuracies.append(accuracy)
                    print_and_log(f"Split: {split}, Scenario: {scenario}, k={k}, repeat {repeat+1}, accuracy: {accuracy:.4f}")

                # Store best k and its mean accuracy for this scenario/split
                mean_accuracy = sum(accuracies) / len(accuracies)
                best_k_counts[(split, scenario)].append([k, mean_accuracy])

    best_overall_mean_accuracy = 0
    best_overall = None

    print_and_log(f"\n--- Best k for each scenario and splitting method across {n_repeats} repeats ---")
    for key, k_list in best_k_counts.items():
        split, scenario = key
        k_values_only = [item[0] for item in k_list]
        accuracy_values_only = [item[1] for item in k_list]
        best_idx = accuracy_values_only.index(max(accuracy_values_only)) 
        best_k = k_values_only[best_idx]
        best_mean_accuracy = accuracy_values_only[best_idx]
        print_and_log(f"\nScenario {scenario}, {split} splitting: k = {best_k} (Mean Accuracy = {best_mean_accuracy:.4f})")

        # Track best overall mean accuracy
        if best_mean_accuracy > best_overall_mean_accuracy:
            best_overall_mean_accuracy = best_mean_accuracy
            best_overall = (split, scenario, best_k)


    split, scenario, best_k = best_overall
    print_and_log(f"\n--- Best overall mean accuracy ---")
    print_and_log(f"Best mean accuracy: {best_overall_mean_accuracy:.4f} (k={best_k}, split={split}, scenario={scenario})")