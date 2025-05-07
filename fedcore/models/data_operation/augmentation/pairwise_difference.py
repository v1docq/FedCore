from itertools import chain

from typing import Optional, Union, List

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either
import torch.nn as nn
from torch import Tensor
import torch

from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.dataset.datasets_from_source import AbstractDataset

from fedcore.models.data_operation.hooks import PairwiseAugmentationHook
from fedcore.models.network_impl.base_nn_model import BaseNeuralForecaster, BaseNeuralModel
from fedcore.models.network_impl.hooks import BaseHook
from torch.utils.data import DataLoader


class PairwiseDifferenceEstimator:
    """
    Base class for Pairwise Difference Learning.
    """

    def __init__(self):
        self.map_dict = {}
        self.initial_classes = None
        self.device = default_device()

    def _create_label_dict(self, stacked_target: Tensor):
        self.map_dict = {}
        self.new_labels = torch.unique(stacked_target).cpu().detach().numpy()
        self.num_new_labels = len(self.new_labels)
        for diff_label, label_for_learning in zip(self.new_labels, range(self.num_new_labels)):
            self.map_dict.update({diff_label: label_for_learning})

    def convert_to_dataloader(self, augmented_input: List[tuple], batch_size: int = 8, task: str = 'classification'):
        stacked_features = torch.vstack([data_tuple[0] for idx, data_tuple in enumerate(augmented_input)])
        stacked_target = torch.hstack([data_tuple[1] for idx, data_tuple in enumerate(augmented_input)])
        if task == 'classification':
            map_func = lambda x: 0 if x == 0 else 1
            self.num_new_labels = 2
        else:
            map_func = lambda x: self.map_dict[x]
            if len(self.map_dict) == 0:
                self._create_label_dict(stacked_target)
        stacked_target = stacked_target.to('cpu').apply_(map_func).to(self.device)
        # replace "negative" labels for correct crossentropy eval
        train_dataset = AbstractDataset(data_source=(stacked_features, stacked_target))
        train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
        return train_dataloader

    def _eval_pair_diff(self, tensor1: Tensor, tensor2: Tensor, with_difference: bool = False):
        old_shape = list(tensor1.shape)
        samples = old_shape[0]
        # Add new axis
        tensor1_exp = tensor1.unsqueeze(1)  # (n, m, d)
        tensor2_exp = tensor2.unsqueeze(0)  # (n, m, d)
        # Diff calc
        diff = tensor1_exp - tensor2_exp  # create set of functions to eval "difference"?
        old_shape[0] = samples * samples
        diff = torch.reshape(diff, old_shape)  # (batch_size x batch_size x sample_shape(channel x width x height))
        # Create common feature space with original features
        return torch.concat([tensor1, diff], dim=0) if with_difference else diff

    def pair_input(self, X: Tensor, y: Tensor, with_difference: bool = False) -> Tensor:
        """
        Cross join с добавлением разностей между признаками.
        Возвращает тензор с парами и их разностями.
        """
        return (self._eval_pair_diff(X, X, with_difference), self._eval_pair_diff(y, y, with_difference))

    def pair_output(self,
                    y1: Tensor,
                    y2: Tensor) -> Tensor:
        """For regresion. beware this is different from regression this is b-a not a-b"""

        y_pair = torch.cross(y1, y2)
        y_pair_diff = y_pair[:, 1] - y_pair[:, 0]
        return y_pair_diff

    def pair_output_difference(self,
                               y1: Tensor,
                               y2: Tensor) -> Tensor:
        """For MultiClassClassification base on difference only"""

        # y_pair = pd.DataFrame(y1).merge(y2, how="cross")
        # y_pair_diff = (y_pair.iloc[:, 1] != y_pair.iloc[:, 0]).astype(int)
        y_pair = torch.cross(y1, y2)
        y_pair_diff = y_pair[y_pair[:, 1] != y_pair[:, 0]].to(torch.int8)
        return y_pair_diff

    # @staticmethod
    # def check_sample_weight(sample_weight: pd.Series, y_train: pd.Series) -> None:
    #     if sample_weight is None:
    #         pass
    #     elif isinstance(sample_weight, pd.Series):
    #         # check
    #         if len(sample_weight) != len(y_train):
    #             raise ValueError(
    #                 f'sample_weight size {len(sample_weight)} should be equal to the train size {len(y_train)}')
    #         if not sample_weight.index.equals(y_train.index):
    #             raise ValueError(
    #                 f'sample_weight and y_train must have the same index\n{sample_weight.index}\n{y_train.index}')
    #         if all(sample_weight.fillna(0) <= 0):
    #             raise ValueError(f'sample_weight are all negative/Nans.\n{sample_weight}')
    #
    #         # norm
    #         class_sums = np.bincount(y_train, sample_weight)
    #         sample_weight = sample_weight / class_sums[y_train.astype(int)]
    #     else:
    #         raise NotImplementedError()
    #
    # @staticmethod
    # def correct_sample_weight(sample_weight: pd.Series, y_train: pd.Series) -> pd.Series:
    #     if sample_weight is not None:
    #         sample_weight = sample_weight / sum(sample_weight)
    #         # norm
    #         # class_sums = np.bincount(y_train, sample_weight)
    #         # sample_weight = sample_weight / class_sums[y_train.astype(int)]
    #
    #     #     # if sample_weight.min() < 0:  # dolla weight change : improvement +0.0032 bof
    #     #     #     sample_weight = sample_weight - sample_weight.min()
    #     return sample_weight

    @staticmethod
    def predict(y_prob: Tensor, output_mode: str = 'default'):
        labels_output = output_mode.__contains__('label')
        return torch.argmax(y_prob, 1) if labels_output else y_prob


class PairwiseDifferenceModel:
    """PDL have a low chance of improvement compared to using directly parametric models like Logit, MLP. \
    To obtain an improvement, it is better to use a tree-based model like: ExtraTrees"""

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.num_classes = None
        self.proba_aggregate_method = params.get('aggregate_func', 'softmax')
        self.use_prior = params.get('use_prior', False)
        self.ft_params = params.get('learning_strategy_params', None)
        self._hooks = [PairwiseAugmentationHook]
        self.prior = None
        self.sample_weight_ = None
        self._init_empty_object()

    def __repr__(self):
        return 'PairwiseAugmentation'

    def _init_empty_object(self):
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        # add hooks
        self._on_epoch_end = []
        self._on_epoch_start = []

    def _init_data_object(self, input_data: InputData):
        self.device = default_device()
        self.num_classes = input_data.features.num_classes
        self.base_model = input_data.features.target
        self.task_type = input_data.task
        self.is_regression_task = any([self.task_type.task_type.value == 'regression',
                                       input_data.task.task_type.value.__contains__('forecasting')])
        try:
            self.initial_classes = np.unique(input_data.features.train_dataloader.dataset.targets)
        except Exception:
            self.initial_classes = np.unique(input_data.features.train_dataloader.dataset.dataset.targets)

        self.pdl_target = [i for i in self.initial_classes for rep in range(self.num_classes)]
        self.min_label_zero = True if 0 in self.initial_classes else False

    def _init_model(self):
        self.pde = PairwiseDifferenceEstimator()
        self.trainer = BaseNeuralModel(self.ft_params)
        self.trainer.model = self.base_model

    def _init_hooks(self):
        for hook_elem in chain(*self._hooks):
            hook: BaseHook = hook_elem.value
            hook = hook(self.ft_params, self.base_model)
            if hook._hook_place >= 0:
                self._on_epoch_end.append(hook)
            else:
                self._on_epoch_start.append(hook)

    def _estimate_prior(self):
        if self.prior is not None:
            return self
        # Calculate class priors
        target = pd.DataFrame(self.target)
        class_counts = target.value_counts()
        class_priors = class_counts / len(self.target)
        # Convert class priors to a dictionary
        self.prior = class_priors.sort_index().values

    def fit(self, input_data: InputData):
        self._init_data_object(input_data)
        self._init_model()
        # self._init_hooks()
        train_bs = input_data.features.train_dataloader.batch_size
        val_bs = input_data.features.val_dataloader.batch_size
        augmented_train_data = list(map(lambda batch: self.pde.pair_input(batch[0].to(self.device),
                                                                          batch[1].to(self.device)),
                                        input_data.features.train_dataloader))
        augmented_val_data = list(map(lambda batch: self.pde.pair_input(batch[0].to(self.device),
                                                                        batch[1].to(self.device)),
                                      input_data.features.train_dataloader))
        # augmented_val_data = list(map(lambda batch: self.pde.pair_input(batch[0].to(self.device),
        #                                                              batch[1].to(self.device)),
        #                            input_data.features.val_dataloader))

        input_data.features.train_dataloader = self.pde.convert_to_dataloader(augmented_input=augmented_train_data,
                                                                              batch_size=train_bs)
        input_data.features.val_dataloader = self.pde.convert_to_dataloader(augmented_input=augmented_val_data,
                                                                            batch_size=val_bs)
        # update output layer for model
        model_layers = list(self.trainer.model.modules())
        output_layer = model_layers[-1]
        self.trainer.model.fc = nn.Linear(output_layer.weight.shape[1], self.pde.num_new_labels)
        self.trainer.fit(input_data)
        return self

    def predict_similarity_samples(self, input_data: InputData, X_anchors=None) -> Tensor:
        """ For each input sample, output C probabilities for each N train pair.
        Beware that this function does not apply the weights at this level
        """

        test_bs = input_data.features.val_dataloader.batch_size
        augmented_test_data = list(map(lambda batch: self.pde.pair_input(batch[0].to(self.device),
                                                                         batch[1].to(self.device)),
                                       input_data.features.val_dataloader))
        input_data.features.val_dataloader = self.pde.convert_to_dataloader(augmented_input=augmented_test_data,
                                                                            batch_size=test_bs)

        predictions_proba_difference: Tensor = self.trainer.predict(input_data)
        # get mean probability valus for case where ab_diff = ba_diff
        # predictions_proba_similarity_ab = predictions_proba_difference.predict[:,:input_data.features.num_classes]
        # predictions_proba_similarity_ba = predictions_proba_difference.predict[:,input_data.features.num_classes-1:]
        # predictions_proba_similarity_ba = torch.flip(predictions_proba_similarity_ba,dims=[0,1])
        # predictions_proba_similarity = (predictions_proba_similarity_ab + predictions_proba_similarity_ba) / 2.
        predictions_proba_similarity = predictions_proba_difference.predict
        dim_reshape = input_data.features.num_classes * input_data.features.num_classes
        predictions_proba_similarity = predictions_proba_similarity.reshape((dim_reshape,dim_reshape,
                                                                             predictions_proba_similarity.shape[1]))
        return predictions_proba_similarity

    def __predict_with_prior(self, input_data: Tensor, sample_weight):
        tests_trains_classes_likelihood = self.predict_proba_samples(input_data)
        tests_classes_likelihood = self._apply_weights(tests_trains_classes_likelihood, sample_weight)
        tests_classes_likelihood = tests_classes_likelihood / tests_classes_likelihood.sum(axis=1)[:, np.newaxis]
        tests_classes_likelihood = tests_classes_likelihood.clip(0, 1)
        return tests_classes_likelihood

    def __predict_without_prior(self, input_data: InputData, sample_weight=None):
        predictions_proba_similarity: Tensor = self.predict_similarity_samples(input_data)
        def f(predictions_proba_similarity: pd.Series) -> pd.Series:
            target = pd.Series(self.pdl_target)
            df = pd.DataFrame(
                {'start': target.reset_index(drop=True), 'similarity': predictions_proba_similarity})
            df = df.fillna(0)
            mean = df.groupby('start', observed=False).mean()['similarity']
            return mean

        predictions_proba_similarity_df = pd.DataFrame(predictions_proba_similarity[:,:,0])
        original_classes_prob_matrix = predictions_proba_similarity_df.apply(f, axis='columns').values
        original_classes_prob_matrix = Tensor(original_classes_prob_matrix)
        # without this normalization it should work for multiclass-multilabel
        if self.proba_aggregate_method == 'norm':
            tests_classes_likelihood_np = original_classes_prob_matrix/original_classes_prob_matrix.sum()
        elif self.proba_aggregate_method == 'softmax':
            tests_classes_likelihood_np = nn.Softmax(-1)(original_classes_prob_matrix)
        return tests_classes_likelihood_np

    def predict_proba_samples(self, X: Union[Tensor, pd.DataFrame]) -> Tensor:
        # todo add unit test with weight ==[1 1 1 ] and weights = None
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        predictions_proba_similarity: pd.DataFrame = self.predict_similarity_samples(X)

        def g(anchor_class: Tensor, predicted_similarity: Tensor) -> Tensor:
            """

            :param anchor_class: array int
            :param predicted_similarity: array float
            :return:
            """
            prior_cls_probs = (1 - self.prior[anchor_class])
            likelyhood_per_anchor = ((1 - predicted_similarity) / prior_cls_probs)
            likelyhood_per_anchor = likelyhood_per_anchor * self.prior
            n_samples = np.arange(len(likelyhood_per_anchor))
            likelyhood_per_anchor[n_samples, anchor_class] = predicted_similarity
            return likelyhood_per_anchor

        anchor_class = self.target.astype(int)

        def f(predictions_proba_similarity: Tensor) -> Tensor:
            """ Here we focus on one test point.
            Given its similarity probabilities.
            Return the probability for each class"""
            test_i_trains_classes = g(anchor_class=anchor_class, predicted_similarity=predictions_proba_similarity)
            np.testing.assert_array_equal(test_i_trains_classes.shape, (len(self.target), self.num_classes))
            return test_i_trains_classes

        tests_trains_classes_likelihood = np.apply_along_axis(f, axis=1, arr=predictions_proba_similarity.values)
        return tests_trains_classes_likelihood

    def _apply_weights(self,
                       tests_trains_classes_likelihood: Tensor,
                       sample_weight: Tensor) -> Tensor:
        tests_classes_likelihood = (tests_trains_classes_likelihood *
                                    sample_weight[np.newaxis, :, np.newaxis]).sum(axis=1)
        # np.testing.assert_array_almost_equal(tests_classes_likelihood.sum(axis=-1), 1.)
        return tests_classes_likelihood

    def _abstract_predict(self,
                          input_data: InputData,
                          output_mode: str = 'default'):
        if self.sample_weight_ is not None:
            self.sample_weight_ = self.sample_weight_.loc[self.target.index].values

        predict_output = Either(value=input_data,
                                monoid=[input_data, self.use_prior]).either(
            left_function=lambda features: self.__predict_without_prior(features, self.sample_weight_),
            right_function=lambda features: self.__predict_with_prior(features, self.sample_weight_))
        output = self.pde.predict(predict_output, output_mode)
        output = output if self.min_label_zero else output + 1
        return output

    def predict(self,
                input_data: InputData,
                output_mode: str = 'labels') -> pd.Series:
        """ For each input sample, output one prediction the most probable class.

        """
        return self._abstract_predict(input_data, output_mode)

    def predict_proba(self,
                      input_data: InputData,
                      output_mode: str = 'default') -> pd.Series:
        """ For each input sample, output one prediction the most probable class.

        """

        return self.predict(input_data, output_mode)

    def predict_for_fit(self,
                        input_data: InputData,
                        output_mode: str = 'default'):
        """ For each input sample, output one prediction the most probable class.
        """
        return self.predict(input_data, output_mode)

    def score_difference(self, input_data: InputData) -> float:
        y_pair_diff = self.pde.pair_output_difference(input_data.target, self.target,
                                                      self.num_classes)  # 0 if similar, 1 if diff
        predictions_proba_similarity: pd.DataFrame = self.predict_similarity_samples(
            input_data.features, reshape=False)  # 0% if different, 100% if similar

        return abs(y_pair_diff - (1 - predictions_proba_similarity)).mean()
