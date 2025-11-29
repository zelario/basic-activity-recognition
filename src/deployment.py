

from features import *
from outliers import *
from embeddings import *


def my_model(sample_dataset):
    """Deploy my model to make predictions on a sample dataset."""

    sample_variables_modules = compute_modules(sample_dataset)
    sample_dataset, sample_variables_modules = remove_outliers(sample_dataset, sample_variables_modules)

    sample_features, sample_labels = extract_features(sample_dataset, sample_variables_modules)
    sample_embeddings, _ = compute_embeddings(sample_dataset)

    

    pass