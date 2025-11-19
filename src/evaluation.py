from log import print_and_log
from model_learning import sklearn_knn_classifier, validate_model
from splitting import mixed_splitting, participant_splitting, prepare_pipeline

def hyperparemeter_tuning(features, embeddings, labels, k_values, n_repeats=1):

    # Dictionary to store best k and its accuracy for each scenario and splitting method
    best_k_counts = {
        (scenario, split): [] for split in ['mixed', 'participant'] for scenario in ['a', 'b', 'c']
    }

    print_and_log(f"\n--- Hyperparameter Tuning over {n_repeats} repeats ---\n")

    # For each splitting method
    for split in ['mixed', 'participant']:

        # For each scenario
        for scenario in ['a', 'b', 'c']:

            # For each repeat (number of splits for each method)
            for repeat in range(n_repeats):

                if split == 'mixed':
                    train_dataset, validation_dataset, test_dataset = mixed_splitting(features, embeddings, labels)
                else:
                    train_dataset, validation_dataset, test_dataset = participant_splitting(features, embeddings, labels)

                pipeline = prepare_pipeline(train_dataset, validation_dataset, test_dataset)
                best_accuracy = 0.0
                best_k = None

                # For each k value in k_values
                for k in k_values:
                    knn_model = sklearn_knn_classifier(pipeline[0], scenario=scenario, k=k)
                    metrics = validate_model(knn_model, pipeline[1], k=k, print_output=False)
                    accuracy = metrics['accuracy']

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_k = k

                best_k_counts[(scenario, split)].append([best_k, best_accuracy, repeat+1])
                print_and_log(f"Scenario {scenario}, {split} splitting, repeat {repeat+1}: k={best_k} (Accuracy: {best_accuracy:.4f})")

    # Compute and print overall best k frequency and mean accuracy
    print_and_log(f"\n--- Best k for each scenario and splitting method across {n_repeats} repeats ---")
    for key, k_list in best_k_counts.items():
        scenario, split = key

        # Extract only k values and accuracies from the list of [k, accuracy, repeat]
        k_values_only = [item[0] for item in k_list]
        accuracy_values_only = [item[1] for item in k_list]

        # Count frequency of each k and collect accuracies for each k
        frequency_dict = {}
        accuracy_dict = {}
        for k, acc in zip(k_values_only, accuracy_values_only):
            frequency_dict[k] = frequency_dict.get(k, 0) + 1
            if k not in accuracy_dict:
                accuracy_dict[k] = []
            accuracy_dict[k].append(acc)

        # Find the most common k
        most_common_k = max(frequency_dict, key=frequency_dict.get)
        frequency = frequency_dict[most_common_k]
        mean_accuracy = sum(accuracy_dict[most_common_k]) / len(accuracy_dict[most_common_k]) if accuracy_dict[most_common_k] else 0.0
        print_and_log(f"\nScenario {scenario}, {split} splitting: k = {most_common_k} (Frequency: {frequency}, Mean Accuracy = {mean_accuracy:.4f})")
        print_and_log(f"All best k values: {k_values_only}")