from enum import Enum
import torch_pruning as tp


class AtomizedModel(Enum):
    PRUNER_MODELS = {'magnitude_pruner': tp.pruner.MagnitudePruner,
                     'group_norm_pruner': tp.pruner.GroupNormPruner,
                     'batch_norm_pruner': tp.pruner.BNScalePruner,
                     'growing_reg_pruner': tp.pruner.GrowingRegPruner}

    QUANTISATION_MODELS = {'magnitude_pruner': tp.pruner.MagnitudePruner,
                           'group_norm_pruner': tp.pruner.GroupNormPruner,
                           'batch_norm_pruner': tp.pruner.BNScalePruner,
                           'growing_reg_pruner': tp.pruner.GrowingRegPruner}


PRUNER_MODELS = AtomizedModel.PRUNER_MODELS.value
QUANTISATION_MODELS = AtomizedModel.QUANTISATION_MODELS.value
