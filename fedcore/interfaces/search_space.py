from hyperopt import hp

from fedcore.repository.constanst_repository import (
    PRUNING_NORMALIZE,
    PRUNING_REDUCTION,
    PRUNING_NORMS,
)

fedcore_search_space = {
    "pruning_model": {
        "importance": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [
                [
                    "MagnitudeImportance",
                    "BNScaleImportance",
                    "GroupNormImportance",
                    "GroupTaylorImportance",
                    "TaylorImportance",
                    "HessianImportance",
                ]
            ],
            "type": "categorical",
        },
        "importance_norm": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [PRUNING_NORMS],
            "type": "categorical",
        },
        "importance_reduction": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [PRUNING_REDUCTION],
            "type": "categorical",
        },
        "importance_normalize": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [PRUNING_NORMALIZE],
            "type": "categorical",
        },
        "pruning_ratio": {
            "hyperopt-dist": hp.uniform,
            "sampling-scope": [0.1, 1.0],
            "type": "continuous",
        },
        "pruning_iterations": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [[x for x in range(1, 5, 1)]],
            "type": "categorical",
        },
    }
}


def get_fedcore_search_space(self):
    parameters_per_operation = {
        "tfidf": {
            "ngram_range": {
                "hyperopt-dist": hp.choice,
                "sampling-scope": [[(1, 1), (1, 2), (1, 3)]],
                "type": "categorical",
            },
            "min_df": {
                "hyperopt-dist": hp.uniform,
                "sampling-scope": [0.0001, 0.1],
                "type": "continuous",
            },
            "max_df": {
                "hyperopt-dist": hp.uniform,
                "sampling-scope": [0.9, 0.99],
                "type": "continuous",
            },
        },
    }
    for key in fedcore_search_space:
        parameters_per_operation[key] = fedcore_search_space[key]

    if self.custom_search_space is not None:
        for operation in self.custom_search_space.keys():
            if self.replace_default_search_space:
                parameters_per_operation[operation] = self.custom_search_space[
                    operation
                ]
            else:
                for key, value in self.custom_search_space[operation].items():
                    parameters_per_operation[operation][key] = value

    return parameters_per_operation
