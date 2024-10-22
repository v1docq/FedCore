from fedot.core.data.data import InputData

from fedcore.repository.constanst_repository import (
    CROSS_ENTROPY,
    MSE,
)


def _get_loss_metric(input_data: InputData):
    if input_data.task.task_type.value == "classification":
        loss_fn = CROSS_ENTROPY() if input_data.num_classes == 2 else CROSS_ENTROPY()
    elif input_data.task.task_type.value == "regression":
        loss_fn = MSE()
    else:
        loss_fn = None
    return loss_fn
