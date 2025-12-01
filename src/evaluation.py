"""
Model evaluation, statistical tests and visualization utilities.
File used for modulo B exercise 5.

This file contains functions used in the project for:
- k-NN hyperparameter tuning and model selection,
- saving and loading model performance metrics,
- performing paired and independent hypothesis tests to compare models,
- visualizing performance distributions with KDE plots,
- reporting results and statistical significance.
"""

from log import print_and_log
from models import sklearn_knn_classifier, validate_model
from splitting import mixed_splitting, participant_splitting, prepare_pipeline
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from augmentation import *

MODEL_NAMES = [
    'Mixed-Features-A', 'Mixed-Embeddings-A',
    'Mixed-Features-B', 'Mixed-Embeddings-B',
    'Mixed-Features-C', 'Mixed-Embeddings-C',
    'Participant-Features-A', 'Participant-Embeddings-A',
    'Participant-Features-B', 'Participant-Embeddings-B',
    'Participant-Features-C', 'Participant-Embeddings-C'
]

def hyperparemeter_tuning(features, embeddings, labels, k_values=[1], n_splits=1):
    """Run k-NN hyperparameter tuning and model selection for all scenarios, types, and splitting methods.
    For each method (mixed, participant), type (features, embeddings), and scenario (a, b, c), performs n_splits random splits, 
    trains k-NN models for each k, selects the best k, retrains on combined train+validate, and evaluates on the test set. Saves all metrics to npy/metrics.npy.

    Parameters
    ----------
    features : matrix, shape (n_samples, n_features)
        Feature matrix for all samples.
    embeddings : matrix, shape (n_samples, n_features)
        Embedding matrix for all samples.
    labels : matrix, shape (n_samples, 3)
        Labels for all samples.
    k_values : list, optional
        List of k values to test (default=[1]).
    n_splits : int, optional
        Number of splits to perform (default=1)."""

    print_and_log(f"\n--- Hyperparameter Tuning over {n_splits} different splits ---\n", path="log/hyperparameter_tuning.log")
    print_and_log(f"\n--- Hyperparameter Tuning best k values results ---\n")

    metrics = {(method, type, scenario): [] for method in ['mixed', 'participant'] for type in ['features', 'embeddings'] for scenario in ['a', 'b', 'c']}

    # For each splitting method
    for method in ['mixed','participant']:

        splits = [mixed_splitting(features, embeddings, labels) for _ in range(n_splits)] if method == 'mixed' else [participant_splitting(features, embeddings, labels) for _ in range(n_splits)]

        pipelines = [prepare_pipeline(split) for split in splits]

        for pipeline in pipelines:
            train_dataset = pipeline["train"]
            augmented_train_dataset = augment_train_dataset(train_dataset)
            pipeline["train"] = augmented_train_dataset

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
                    metrics[(method, type, scenario)].append(iteration_metrics)

                    print_and_log(f"Method: {method}, Type: {type}, Scenario: {scenario}, Split= {split_number}, k= {best_k}, Accuracy: {iteration_metrics['accuracy']:.4f}")
    
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

O melhor desempenho ocorre para k=1, sugerindo que uma seleção mais restrita (menos vizinhos ou features mai s relevantes) melhora a classificação. A remoção de outliers no início 
do processamento contribuiu para este resultado, tornando o k-NN com k=1 mais fiável. A seleção de features, por si só, não trouxe melhorias significativas em relação ao uso de 
embeddings, que já encapsulam informação relevante.
'''

def paired_hypothesis_test(method, metrics=None, chosen_metric='accuracy'):
    """Perform paired hypothesis tests and KDE visualizations for all pairs of models within a method (mixed or participant).
    For each pair of models (type and scenario) in the selected method, performs a paired t-test and plots the distribution of the chosen metric for each model.
    Shows 15 pairwise comparisons in a grid of subplots.

    Parameters
    ----------
    method : str
        Splitting method ('mixed' or 'participant').
    metrics : dict, optional
        Dictionary of saved metrics (default: loads from npy/metrics.npy).
    chosen_metric : str, optional
        Metric to compare (default='accuracy')."""

    try:
        metrics = np.load("npy/metrics.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print_and_log("Metrics file not found. Please run hyperparameter_tuning() first.")
        return
    
    print_and_log(f"\n--- Paired hypothesis test for method: {method} using metric: {chosen_metric} ---\n")

    model_keys = [
        (method, 'features', 'a'),
        (method, 'embeddings', 'a'),
        (method, 'features', 'b'),
        (method, 'embeddings', 'b'),
        (method, 'features', 'c'),
        (method, 'embeddings', 'c')
    ]

    values = [[model_metrics[chosen_metric] for model_metrics in metrics[key]] for key in model_keys]

    pairs = []
    for i in range(len(model_keys)):
        for j in range(i+1, len(model_keys)):
            pairs.append((i, j))

    fig, axs = plt.subplots(5, 3, figsize=(9, 12))
    axs = axs.flatten()

    for subplot_idx, (i, j) in enumerate(pairs):
        ax = axs[subplot_idx]
        sns.kdeplot(values[i], label=MODEL_NAMES[i], fill=True, ax=ax)
        sns.kdeplot(values[j], label=MODEL_NAMES[j], fill=True, ax=ax)

        stat, p_value = ttest_rel(values[i], values[j])

        ax.text(0.95, 0.95, f'p={p_value:.4f}', transform=ax.transAxes, ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlabel(chosen_metric.capitalize(), fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.grid(True)
        ax.legend(loc='upper left', fontsize=7)

        print_and_log(f"{MODEL_NAMES[i]} vs {MODEL_NAMES[j]}: p-value = {p_value:.4f}")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle(f'{method.capitalize()} Model Comparisons (All Pairs)', y=0.995)
    plt.show()


def independent_hypothesis_test(metrics=None, chosen_metric='accuracy'):
    """Perform independent hypothesis tests and KDE visualizations comparing mixed vs participant versions of each model.
    For each scenario and type, performs an independent t-test and plots the distribution of the chosen metric for mixed and participant models.
    Shows 6 comparisons in subplots.

    Parameters
    ----------
    metrics : dict, optional
        Dictionary of saved metrics (default: loads from npy/metrics.npy).
    chosen_metric : str, optional
        Metric to compare (default='accuracy')."""

    try:
        metrics = np.load("npy/metrics.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print_and_log("Metrics file not found. Please run hyperparameter_tuning() first.")
        return

    print_and_log(f"\n--- Independent hypothesis test: Mixed vs Participant ---\n")

    scenarios = ['a', 'b', 'c']
    types = ['features', 'embeddings']

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = axs.flatten()

    for index, (scenario, type) in enumerate([(s, t) for s in scenarios for t in types]):

        mixed_index = index
        participant_index = index + 6
        mixed_name = MODEL_NAMES[mixed_index]
        participant_name = MODEL_NAMES[participant_index]

        mixed_values = [model_metrics[chosen_metric] for model_metrics in metrics[('mixed', type, scenario)]]
        participant_values = [model_metrics[chosen_metric] for model_metrics in metrics[('participant', type, scenario)]]

        stat, p_value = ttest_ind(mixed_values, participant_values)
        ax = axs[index]

        sns.kdeplot(mixed_values, label=mixed_name, fill=True, ax=ax)
        sns.kdeplot(participant_values, label=participant_name, fill=True, ax=ax)
        ax.set_xlabel(chosen_metric.capitalize(), fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.text(0.95, 0.95, f'p={p_value:.4f}', transform=ax.transAxes, ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True)

        print_and_log(f"{mixed_name} vs {participant_name}: p-value = {p_value:.4f}")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle('Mixed vs Participant Model Comparisons (All Versions)', y=0.995)
    plt.show()