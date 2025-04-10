# from collections import Counter
# import timeit
#
# import numpy as np
# from numpy.typing import NDArray
#
# NDArrayFloat = NDArray[np.float64]
#
#
# class KNN:
#     def __init__(self, k=3):
#         self.k = k
#
#     def fit(self, X_train: NDArrayFloat, y_train: NDArrayFloat) -> None:
#         self.X_train = X_train
#         self.y_train = y_train
#
#     def _sort_idx(self, dist):
#         k_indices = sorted(range(len(dist)), key=lambda i: dist[i])
#         # partition_sort сортировка top_k
#         k_indices = k_indices[:self.k]
#         k_nearest_labels = [self.y_train[i] for i in k_indices]
#         most_common = Counter(k_nearest_labels).most_common(1)
#         return most_common[0][0]
#
#     def predict(self, X_test: NDArrayFloat) -> NDArrayFloat:
#         predictions = []
#         n_samples_test = len(X_test)
#         #n_samples_train = len(self.X_train)
#         for i in range(n_samples_test):  # 1 цикл вычисление для каждого семпла в тесте
#             #dist = [self.euclidean_distance(X_test[i], self.X_train[j]) for j in range(n_samples_train)]
#             dist = self.euclidean_distance(test_sample=X_test[i],train_array=self.X_train)
#             # dist = list(map(lambda sample: self.euclidean_distance(test_sample=X_test[i],
#             #                                                        train_sample=sample), self.X_train))
#             # test_sample в np.array (train)
#             # 2 цикл вычисление dist попарно
#             # для каждого семпла в трейне (соотвественно здесь мы можем векторизовать наши вычисления фиксируя
#             # в качестве 2 входных аргументов - функцию вычисления dist и наш семпл из теста
#             # dist = list(map(lambda x:,self.X_train))
#
#            #most_common = self._sort_idx(dist)
#             predictions.append(self._sort_idx(dist)) # потенциально можно ли оптимизировать тут?
#         return predictions
#
#     def euclidean_distance(self, test_sample: NDArrayFloat, train_array: NDArrayFloat) -> NDArrayFloat:
#         #diff = test_sample - train_array
#         return np.sqrt(np.sum((test_sample - train_array) ** 2, axis=1)) # корень по факту не нужен из-за монотоности
#
#
# def generate_dataset(
#         n: int,
#         prior_class_probs=(0.6, 0.2, 0.2)) -> tuple[NDArrayFloat, NDArrayFloat]:
#     classes = [0, 1, 2]
#     class_means = [0, 10, 20]
#     prior_class_probs = [0.6, 0.2, 0.2]
#     x = np.zeros((n, 2))
#     y = np.random.choice(classes, size=n, replace=True, p=prior_class_probs)
#     y_class_counts = [(y == cls).sum() for cls in classes]
#     for cls, cls_count, cls_mean in zip(classes, y_class_counts, class_means):
#         x[y == cls] = np.random.normal(
#             loc=cls_mean,
#             scale=1.,
#             size=(cls_count, 2)
#         ).reshape(cls_count, 2)
#     return x, y
# if __name__ == "__main__":
#     x_train, y_train = generate_dataset(n=1000)
#     x_test, y_test = generate_dataset(n=100)
#
#     model = KNN(k=3)
#     model.fit(x_train, y_train)
#     time_in_sec = timeit.timeit(lambda: model.predict(x_test), number=100)
#     print(f"Time: {time_in_sec:.2E}")
#     y_pred = model.predict(x_test)
#     acc = (y_pred == y_test).mean()
#     print(f"Accuracy: {acc:.2f}")

from __future__ import annotations
from pathlib import Path

import torch
from typing import Literal, Protocol
from torch import nn
from torch import Tensor
# from torch import Dataloader
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pandas as pd

from fedcore.metrics.metric_impl import RMSE

NDArrayFloat = NDArray[np.float64]

class SimpleRegressionTorch(nn.Module):

    def __init__(self, input_dim:int,
                 output_dim:int,
                 depth:int):
        #self.fc = nn.
        self.activation = nn.ReLU
        for d in range(depth):
            nn.ModuleList()


    def forward(self, input_tensor):
        output = self.fc(input_tensor)
        output = self.activation(input_tensor)
        return output



class Dataset:
    def __init__(self, target: NDArrayFloat, x: NDArrayFloat, y: NDArrayFloat) -> None:
        self.target = target
        self.x: NDArrayFloat = x
        self.y: NDArrayFloat = y
        self.tensor_dataset = torch.concatenate([Tensor(x),Tensor(y)])

    @classmethod
    def from_dump(cls, path_to_dataset: Path) -> Dataset:
        df = pd.read_csv(path_to_dataset)
        return cls(
            target=df["target"].to_numpy(dtype=np.float64),
            x=df["x"].to_numpy(dtype=np.float64),
            y=df["y"].to_numpy(dtype=np.float64),
        )

    def plot(self) -> None:
        n_x = len(np.unique(self.x))
        n_y = len(np.unique(self.y))
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        conts = ax.pcolormesh(
            self.x.reshape(n_y, n_x)[0, :],
            self.y.reshape(n_y, n_x)[:, 0],
            self.target.reshape(n_y, n_x),
        )
        ax.set_xlabel(r"$x$", fontsize=12)
        ax.set_ylabel(r"$y$", fontsize=12)
        fig.colorbar(conts)
        plt.tight_layout()
        plt.show()

def train_loop(model,epochs,bs, optimizer, dataloader):
    criterion = RMSE()
    for epoch in epochs:
        for batch in dataloader:
            x,y = batch
            model.to()
            predict = model(x)
            loss = criterion(x,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model
def fit_predict(train_dataset: Dataset, test_dataset: Dataset) -> NDArrayFloat:
    pred = np.zeros_like(test_dataset.target)
    model = SimpleRegressionTorch(input_dim=train_dataset.tensor_dataset.shape[1],
                                  output_dim=1,
                                  depth=2)
    dataloader = Dataloder(dataset = train_dataset)

    train_loop()

    ##########################
    ### PUT YOUR CODE HERE ###
    ##########################

    return pred


if __name__ == "__main__":
    train_dataset = Dataset.from_dump(Path("./train_dataset.csv"))
    test_dataset = Dataset.from_dump(Path("./test_dataset.csv"))
    # target непрерывный - т.е. смотрим в сторону регрессионых задач
    # f(x1,x2) = y
    # в качестве модели просто нейронка на допустим 2 слоя с input_dim = 2, output_dim = 1
    # в качестве лосса - rmse
    train_dataset.plot()

    pred = fit_predict(train_dataset, test_dataset)
    residual_dataset = Dataset(
        target=test_dataset.target - pred,
        x=test_dataset.x,
        y=test_dataset.y,
    )
    residual_dataset.plot()

