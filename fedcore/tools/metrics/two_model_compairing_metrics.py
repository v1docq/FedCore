from typing import Tuple
from sklearn.metrics import accuracy_score
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def compare_accuracy(model_before: nn.Module, model_after: nn.Module, data_loader: DataLoader, n_batches=10) -> Tuple:
    preds_before = []
    preds_after = []
    targets = []

    for i, batch in enumerate(data_loader):
        if i >= n_batches:
            break
        x, y = batch[0], batch[1]
        # forward
        out_b = model_before(x)
        out_a = model_after(x)
        preds_before.append(out_b.argmax(dim=1).numpy())
        preds_after.append(out_a.argmax(dim=1).numpy())
        targets.append(y.numpy())

    y_true = np.concatenate(targets)
    y_pred_b = np.concatenate(preds_before)
    y_pred_a = np.concatenate(preds_after)

    acc_before = accuracy_score(y_pred_b, y_true)
    acc_after = accuracy_score(y_pred_a, y_true)
    return acc_before, acc_after