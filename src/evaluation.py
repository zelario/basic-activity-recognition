from log import print_and_log
from model_learning import sklearn_knn_classifier, validate_model
from splitting import mixed_splitting, participant_splitting, prepare_pipeline
import numpy as np

def hyperparemeter_tuning(features, embeddings, labels, k_values=[1], n_splits=1):

    print_and_log(f"\n--- Hyperparameter Tuning over {n_splits} different splits ---\n", path="log/hyperparameter_tuning.log")
    print_and_log(f"\n--- Hyperparameter Tuning best k values results ---\n")

    metrics = [] 

    # For each splitting method
    for method in ['mixed', 'participant']:

        splits = [mixed_splitting(features, embeddings, labels) for _ in range(n_splits)] if method == 'mixed' else [participant_splitting(features, embeddings, labels) for _ in range(n_splits)]

        # For each data type
        for type in ['features', 'embeddings']:

            # For each scenario
            for scenario in ['a', 'b', 'c']:

                k_accuracies = {(k): [] for k in k_values}

                # For n_splits or number of splits
                for i, split in enumerate(splits):

                    # Split number label
                    split_number = i + 1 if method == 'mixed' else 10 + i + 1

                    # Prepare dataset
                    pipeline = prepare_pipeline(split[0], split[1], split[2])

                    # For each k value
                    for k in k_values:

                        # Create KNN model
                        knn_model = sklearn_knn_classifier(pipeline[0], scenario=scenario, type=type, k=k)

                        # Validate model
                        iteration_metrics = validate_model(knn_model, pipeline[1], k=k, scenario=scenario, type=type, method=method)

                        metrics.append((method, type, scenario, split_number, k, 
                                        iteration_metrics['confusion_matrix'], iteration_metrics['accuracy'],
                                        iteration_metrics['precision'], iteration_metrics['recall'], iteration_metrics['f1_score']))

                        # Store the k and its accuracy
                        k_accuracies[(k)].append(iteration_metrics['accuracy'])
                        print_and_log(f"Method: {method}, Type: {type}, Scenario: {scenario}, Split= {split_number}, k= {k}, Accuracy: {iteration_metrics['accuracy']:.4f}", path="log/hyperparameter_tuning.log")

                # Get the best k value
                k_mean_accuracy = {k: (sum(accs)/len(accs) if accs else 0) for k, accs in k_accuracies.items()}
                best_k_value = max(k_mean_accuracy, key=lambda k: k_mean_accuracy[k])

                # Join train and validate sets for final model training
                combined_features_all = np.concatenate([pipeline[0][0], pipeline[1][0]], axis=0)
                combined_features_pca = np.concatenate([pipeline[0][1], pipeline[1][1]], axis=0)
                combined_features_relief = np.concatenate([pipeline[0][2], pipeline[1][2]], axis=0)
                combined_embeddings_all = np.concatenate([pipeline[0][3], pipeline[1][3]], axis=0)
                combined_embeddings_pca = np.concatenate([pipeline[0][4], pipeline[1][4]], axis=0)
                combined_embeddings_relief = np.concatenate([pipeline[0][5], pipeline[1][5]], axis=0)
                combined_labels =  np.concatenate([pipeline[0][6], pipeline[1][6]], axis=0)

                new_train_dataset = (combined_features_all, combined_features_pca, combined_features_relief,
                                     combined_embeddings_all, combined_embeddings_pca, combined_embeddings_relief,
                                     combined_labels)

                # Create final KNN model with the best k
                knn_model = sklearn_knn_classifier(new_train_dataset, scenario=scenario, type=type, k=best_k_value)

                # Final test
                iteration_metrics = validate_model(knn_model, pipeline[2], k=best_k_value, scenario=scenario, type=type, method=method)
                print_and_log(f"Method: {method}, Type: {type}, Scenario: {scenario}, k= {best_k_value}, Accuracy: {iteration_metrics['accuracy']:.4f}")
    
    metrics = np.array(metrics, dtype=object)
    np.save("npy/metrics.npy", metrics)

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
