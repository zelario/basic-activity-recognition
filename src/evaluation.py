from log import print_and_log
from model_learning import sklearn_knn_classifier, validate_model
from splitting import mixed_splitting, participant_splitting, prepare_pipeline
from augmentation import augment_dataset
import numpy as np

def hyperparemeter_tuning(features, embeddings, labels, k_values=[1], n_splits=1):

    print_and_log(f"\n--- Hyperparameter Tuning over {n_splits} different splits ---\n", path="log/hyperparameter_tuning.log")
    print_and_log(f"\n--- Hyperparameter Tuning best k values results ---\n")

    metrics = []
    metrics_labels = [] 

    # For each splitting method
    for method in ['mixed', 'participant']:

        splits = [mixed_splitting(features, embeddings, labels) for _ in range(n_splits)] if method == 'mixed' else [participant_splitting(features, embeddings, labels) for _ in range(n_splits)]

        pipelines = [prepare_pipeline(split) for split in splits]

        # For each data type
        for type in ['features', 'embeddings']:

            # For each scenario
            for scenario in ['a', 'b', 'c']:

                # For n_splits or number of splits
                for i in range(n_splits):

                    # Split number label
                    split_number = i + 1 if method == 'mixed' else 10 + i + 1

                    # Get split pipeline
                    pipeline = pipelines[i]
                    
                    # Blank accuracy list for each k
                    k_accuracies = {(k): [] for k in k_values}

                    # For each k value
                    for k in k_values:

                        # Unpack datasets and labels from desired model
                        train_dataset = pipeline["train"][type][scenario]
                        train_labels = pipeline["train"]["labels"]

                        validate_dataset = pipeline["validate"][type][scenario]
                        validate_labels = pipeline["validate"]["labels"]

                        # Train knn model and validate
                        knn_model = sklearn_knn_classifier(train_dataset, train_labels, k=k)
                        iteration_metrics = validate_model(knn_model, validate_dataset, validate_labels, k=k, scenario=scenario, type=type, method=method)
                        accuracy = iteration_metrics['accuracy']

                        # Store accuracy for this k
                        k_accuracies[(k)].append(accuracy)
                        print_and_log(f"Method: {method}, Type: {type}, Scenario: {scenario}, Split= {split_number}, k= {k}, Accuracy: {accuracy:.4f}", path="log/hyperparameter_tuning.log")

                    best_k = max(k_accuracies, key=lambda key: np.mean(k_accuracies[key]))

                    # Retrain on combined train + validate, test on test set
                    combined_dataset = pipeline["combined"][type][scenario]
                    combined_labels = pipeline["combined"]["labels"]
                    test_dataset = pipeline["test"][type][scenario]
                    test_labels = pipeline["test"]["labels"]

                    # Final model training and evaluation
                    knn_model = sklearn_knn_classifier(combined_dataset, combined_labels, k=best_k)
                    iteration_metrics = validate_model(knn_model, test_dataset, test_labels, k=best_k, scenario=scenario, type=type, method=method)

                    # Keep metrics and labels
                    metrics.append((iteration_metrics['confusion_matrix'],
                                    iteration_metrics['accuracy'], 
                                    iteration_metrics['precision'], 
                                    iteration_metrics['recall'], 
                                    iteration_metrics['f1_score']))
                    metrics_labels.append((method, type, scenario, best_k))

                    print_and_log(f"Method: {method}, Type: {type}, Scenario: {scenario}, Split= {split_number}, k= {best_k}, Accuracy: {iteration_metrics['accuracy']:.4f}")
    
    metrics = np.array(metrics, dtype=object)
    metrics_labels = np.array(metrics_labels, dtype=object)
    np.save("npy/metrics.npy", metrics)
    np.save("npy/metrics_labels.npy", metrics_labels)

# --- Exercise 5.2: Results Report ---

'''
1. Matrizes de confusão e atividades mais difíceis de classificar :

Ao analisar as matrizes de confusão, nota-se que a maioria das atividades é bem classificada, especialmente para k=1 devido à menor remoção de outliers do dataset. No entanto, 
à medida que k aumenta, a accuracy diminui, indicando que algumas atividades se tornam mais difíceis de distinguir, provavelmente devido à sua semelhança. As atividades com 
valores mais baixos de precisão e recall são as que apresentam maior confusão, mas no geral, o desempenho é aceitável.

2. Melhor dataset: features ou embeddings:

Os resultados mostram que os embeddings apresentam melhor desempenho, especialmente no cenário a). Nos outros cenários, a diferença diminui, mas embeddings continuam ligeiramente 
superiores. Portanto, embeddings são o melhor dataset para classificação das atividades revelando que as features extraídas capturam inferiormente as características relevantes.

3. Seleção de features e impacto na performance:

O melhor desempenho ocorre para k=1, sugerindo que uma seleção mais restrita (menos vizinhos ou features mais relevantes) melhora a classificação. A remoção de outliers no início 
do processamento contribuiu para este resultado, tornando o k-NN com k=1 mais fiável. A seleção de features, por si só, não trouxe melhorias significativas em relação ao uso de 
embeddings, que já encapsulam informação relevante.
'''
