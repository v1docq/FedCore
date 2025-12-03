from hyperopt import hp
import numpy as np

from fedcore.repository.constant_repository import (
    PRUNING_NORMALIZE,
    PRUNING_REDUCTION,
    PRUNING_NORMS,
    PrunerImportances,
)
from fedcore.algorithm.low_rank.rank_pruning import SLRStrategiesEnum
from fedcore.algorithm.low_rank.decomposer import DECOMPOSERS

fedcore_search_space = {
    "pruning_model": {
        "importance": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [
                [importance.name for importance in PrunerImportances]
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
            "sampling-scope": [0.15, 0.95],
            "min": 0.15,
            "max": 0.95,
            "type": "continuous",
        }
        # "pruning_iterations": {
        #     "hyperopt-dist": hp.choice,
        #     "sampling-scope": [[x for x in range(1, 5, 1)]],
        #     "type": "categorical",
        # },
    },
    "low_rank_model": {
        "strategy": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [
                [strategy.name for strategy in SLRStrategiesEnum]
            ],
            "type": "categorical",
        },
        "decomposer": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [
                list(DECOMPOSERS.keys())
            ],
            "type": "categorical",
        },
        "rank": {
            "hyperopt-dist": hp.uniform,
            "sampling-scope": [0.1, 0.9],
            "min": 0.1,
            "max": 0.9,
            "type": "continuous",
        },
        "distortion_factor": {
            "hyperopt-dist": hp.uniform,
            "sampling-scope": [0.1, 0.9],
            "min": 0.1,
            "max": 0.9,
            "type": "continuous",
        },
        "non_adaptive_threshold": {
            "hyperopt-dist": hp.uniform,
            "sampling-scope": [0.1, 0.9],
            "min": 0.1,
            "max": 0.9,
            "type": "continuous",
        },
        "compose_mode": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [
                [
                    "one_layer",
                    "two_layers",
                    "three_layers"
                ]
            ],
            "type": "categorical",
        },
        "rank_prune_each": {
            # Positive values: frequency in epochs (e.g., 1=every epoch, 2=every 2 epochs)
            # Special value -1 means one-time pruning at the end
            # Note: 0 is not a valid value (must be -1 or positive integer >= 1)
            "hyperopt-dist": hp.choice,
            "sampling-scope": [[-1] + list(range(1, 101))],  # -1 (disabled) or 1-100 epochs (0 excluded)
            "type": "categorical",
        },
        "power": {
            # Default: 3 (from LowRankTemplate)
            # Power parameter for RandomizedSVD (used in torch.pow(G, power))
            # Integer parameter controlling matrix power iteration
            "hyperopt-dist": hp.randint,
            "sampling-scope": [1, 10],
            "min": 1,
            "type": "integer",
        },
        "random_init": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [
                [
                    "normal",
                    "uniform"
                ]
            ],
            "type": "categorical",
        },
    },
    "quantization_model": {
        "quant_type": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [
                [
                    "dynamic",
                    "static",
                    "qat"
                ]
            ],
            "type": "categorical",
        },
        "allow_emb": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [[True, False]],
            "type": "categorical",
        },
        "allow_conv": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [[True, False]],
            "type": "categorical",
        },
        "quant_each": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [[-1, 1, 2, 3, 5, 10]],
            "type": "categorical",
        },
        "prepare_qat_after_epoch": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [[1, 2, 3, 5]],
            "type": "categorical",
        },
    },
    "distillation_model": {
        "epochs": {
            "hyperopt-dist": hp.choice,
            "sampling-scope": [[5, 10, 15, 20, 25]],
            "type": "categorical",
        },
        "learning_rate": {
            "hyperopt-dist": hp.uniform,
            "sampling-scope": [1e-5, 1e-2],
            "min": 1e-5,
            "max": 1e-2,
            "type": "continuous",
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
                "min": 0.0001,
                "max": 0.1,
                "type": "continuous",
            },
            "max_df": {
                "hyperopt-dist": hp.uniform,
                "sampling-scope": [0.9, 0.99],
                "min": 0.9,
                "max": 0.99,
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
