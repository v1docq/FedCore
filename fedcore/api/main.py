"""Public FedCore facade with explicit runtime ownership."""
import logging
from copy import deepcopy
from functools import partial
from pathlib import Path
import torch
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedcore.api.api_configs import ConfigTemplate, validate_config
from fedcore.api.utils.misc import camel_to_snake, extract_fitted_operation
from fedcore.api.utils.evaluation import predict_module, predict_modules, evaluation_loader
from fedcore.data.data import CompressionInputData
from fedcore.repository.initializer_industrial_models import FedcoreModels
from fedcore.tools.registry.checkpoint_manager import CheckpointManager, CheckpointError


class _PipelineInputData(CompressionInputData):
    """FEDOT copies metadata/models while preserving the existing data sources."""
    def __deepcopy__(self, memo):
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        for name, value in vars(self).items():
            setattr(copied, name, value if name.endswith('_dataloader') else deepcopy(value, memo))
        return copied


class FedCore(Fedot):
    def __init__(self, api_config: ConfigTemplate, **kwargs):
        self._external_dask_client = kwargs.pop('dask_client', None)
        self._external_dask_cluster = kwargs.pop('dask_cluster', None)
        if not isinstance(api_config, ConfigTemplate):
            raise TypeError('api_config must be a materialized ConfigTemplate')
        self.manager = deepcopy(api_config)
        self.manager.update(kwargs)
        validate_config(self.manager)
        self.logger = logging.getLogger('Fedcore')
        self.fedcore_model = None
        self.__original_model = None
        self._owns_dask = False
        self._adaptation = FedcoreModels()
        self.metric_dict = None

    @property
    def original_model(self):
        return self.__original_model

    @property
    def compressed_model(self):
        operation = self.fedcore_model
        if isinstance(operation, Pipeline):
            operation = extract_fitted_operation(operation)
        compressed = getattr(operation, 'model_after', None)
        return compressed if compressed is not None else (operation if isinstance(operation, torch.nn.Module) else None)

    @property
    def fedcore_model_for_inference(self):
        return self.compressed_model

    def __init_fedcore_backend(self, input_data=None):
        from fedcore.interfaces.fedcore_optimizer import FedcoreEvoOptimizer
        if not isinstance(self.manager.automl_config.optimizer, partial):
            optimizer = partial(FedcoreEvoOptimizer, optimisation_params={
                'mutation_strategy': self.manager.automl_config.mutation_strategy,
                'mutation_agent': self.manager.automl_config.mutation_agent,
                'trace_path': self.manager.automl_config.search_trace_path})
            self.manager.automl_config.optimizer = optimizer
            self.manager.automl_config.fedot_config.optimizer = optimizer
        return input_data

    def __init_dask(self, input_data):
        if self._external_dask_client is not None:
            self.manager.dask_client = self._external_dask_client
            self.manager.dask_cluster = self._external_dask_cluster
            return input_data
        from distributed import Client, LocalCluster
        params = self.manager.compute_config.distributed
        params = params.to_dict() if hasattr(params, 'to_dict') else dict(params or {})
        params = params.get('cluster_params', params)
        cluster = LocalCluster(**params)
        try:
            client = Client(cluster)
        except BaseException:
            cluster.close()
            raise
        self.manager.dask_client, self.manager.dask_cluster = client, cluster
        self._owns_dask = True
        return input_data

    def __build_assumption(self):
        builder = PipelineBuilder()
        configs = self.manager.learning_config.peft_strategy_params
        configs = configs if isinstance(configs, (tuple, list)) else (configs,)
        learning = self.manager.learning_config.learning_strategy_params
        tokenizer = learning.get('tokenizer') or learning.get('custom_learning_params', {}).get('tokenizer')
        for config in configs:
            params = config.to_dict()
            params['device'] = self.manager.device_config.device
            if config.get_default_name() == 'Lora':
                rank, r = params.pop('rank'), params.pop('r')
                params['lora_r'] = rank or r or params['lora_r']
                aliases = params.pop('target_layers')
                if aliases is not None:
                    params['lora_target_modules'] = aliases
            if tokenizer is not None:
                params.setdefault('tokenizer', tokenizer)
            name = config.get_default_name().removesuffix('Config')
            builder.add_node(operation_type=camel_to_snake(name) + '_model', params=params)
        return builder.build()

    def __init_solver(self, data=None):
        settings = self.manager.automl_config.fedot_config.to_dict()
        self.manager.solver = Fedot(**settings, use_input_preprocessing=False, use_auto_preprocessing=False)
        self.manager.solver.params.data['initial_assumption'] = self.__build_assumption()
        return data

    def __init_solver_no_evo(self, data=None):
        self.manager.solver = self.__build_assumption()
        return data

    def _resolve_model(self, specification):
        if isinstance(specification, torch.nn.Module):
            return specification
        if isinstance(specification, dict):
            path = specification.get('path_to_model') or specification.get('checkpoint_path')
            template = specification.get('model')
            factory = specification.get('model_factory')
            if path is None:
                raise CheckpointError('Checkpoint input requires path_to_model or checkpoint_path')
            if template is None and factory is None and specification.get('model_type'):
                from fedcore.models.backbone.backbone_loader import load_backbone
                template = load_backbone(specification, self.manager.learning_config.learning_strategy_params)
            return CheckpointManager('.').load_from_file(str(path), 'cpu', model=template, model_factory=factory)
        if isinstance(specification, (str, Path)):
            if Path(specification).is_file():
                return CheckpointManager('.').load_from_file(str(specification), 'cpu')
            from fedcore.models.backbone.backbone_loader import load_backbone
            return load_backbone(str(specification), self.manager.learning_config.learning_strategy_params)
        raise TypeError('Expected torch.nn.Module, backbone name or checkpoint specification')

    def _process_input_data(self, data):
        if not isinstance(data, CompressionInputData):
            raise TypeError('input_data must be CompressionInputData')
        specification = data.model
        if specification is None:
            specification = self.manager.automl_config.fedot_config.initial_assumption
        model = self._resolve_model(specification)
        processed = _PipelineInputData.__new__(_PipelineInputData)
        processed.__dict__.update(vars(data))
        processed.model = model
        processed.supplementary_data = deepcopy(data.supplementary_data)
        processed.supplementary_data.is_auto_preprocessed = True
        if self.__original_model is None:
            self.__original_model = deepcopy(model)
        return processed

    def _save_metrics_from_evaluator(self):
        solver = self.manager.solver
        if getattr(solver, 'history', None) is None:
            return
        operation = extract_fitted_operation(self.fedcore_model) if isinstance(self.fedcore_model, Pipeline) else self.fedcore_model
        fedcore_id = getattr(operation, '_fedcore_id', None)
        model_id = getattr(operation, '_model_id_after', None)
        if fedcore_id and model_id:
            from fedcore.tools.registry.model_registry import ModelRegistry
            ModelRegistry().save_metrics_from_evaluator(solver, fedcore_id, model_id)

    def _pretrain_before_optimise(self, data):
        pipeline = PipelineBuilder().add_node('training_model',
            params={**self.manager.learning_config.learning_strategy_params.to_dict(),
                    'device': self.manager.device_config.device}).build()
        pipeline.fit(data)
        operation = extract_fitted_operation(pipeline)
        trained = getattr(operation, 'model', None) or getattr(operation, 'model_after', None)
        if trained is None:
            raise RuntimeError('Pretraining did not produce a model')
        data.model = trained
        return data

    def fit(self, input_data, manually_done=False, **kwargs):
        validate_config(self.manager)
        with self._adaptation:
            try:
                data = self._process_input_data(input_data)
                self.__init_fedcore_backend(data)
                self.__init_dask(data)
                if self.manager.solver is None:
                    self.__init_solver(data)
                if self.manager.learning_config.learning_strategy == 'from_scratch' and not manually_done:
                    data = self._pretrain_before_optimise(data)
                self.fedcore_model = self.manager.solver.fit(data, **kwargs)
                self._save_metrics_from_evaluator()
                return self.fedcore_model
            finally:
                self.shutdown()

    def fit_no_evo(self, input_data, manually_done=False, **kwargs):
        validate_config(self.manager)
        with self._adaptation:
            try:
                data = self._process_input_data(input_data)
                self.__init_fedcore_backend(data)
                self.__init_dask(data)
                if self.manager.solver is None:
                    self.__init_solver_no_evo(data)
                fitted = self.manager.solver.fit(data)
                operation = extract_fitted_operation(self.manager.solver) if isinstance(self.manager.solver, Pipeline) else self.manager.solver
                self.fedcore_model = operation
                self.optimised_model = self.compressed_model
                return fitted
            finally:
                self.shutdown()

    def predict(self, predict_data, output_mode='fedcore', **kwargs):
        if output_mode not in ('fedcore', 'original', 'default', 'model_before', 'model_after', 'raw', 'labels', 'probs'):
            raise ValueError(f'Unsupported output_mode: {output_mode!r}')
        data = self._process_input_data(predict_data)
        if output_mode in ('original', 'default', 'model_before'):
            model = self.original_model
        else:
            model = self.compressed_model or self.original_model
        if isinstance(model, torch.nn.Module):
            result = predict_module(model, data, kwargs.get('split', 'val'))
            if data.task.task_type.name == 'classification' and output_mode != 'raw':
                if output_mode == 'labels':
                    result.predict = result.predict.argmax(dim=-1)
                else:
                    result.predict = torch.softmax(result.predict, dim=-1)
        elif self.fedcore_model is not None:
            with self._adaptation:
                result = self.fedcore_model.predict(data, output_mode)
        else:
            raise ValueError('No model is available for prediction')
        self.manager.predicted_labels = result
        return result

    def get_report(self, test_data, split='val'):
        data = self._process_input_data(test_data)
        models = {'original': self.original_model, 'fedcore': self.compressed_model or self.original_model}
        _, target, predictions = predict_modules(models, data, split)
        from fedcore.metrics import COMPUTATIONAL_METRICS
        from fedcore.metrics.quality import calculate_metrics, MetricFactory
        metrics = self.manager.automl_config.fedot_config.metric or []
        values = {}
        for mode, model in models.items():
            quality = [metric for metric in metrics if metric not in COMPUTATIONAL_METRICS]
            result = calculate_metrics(quality, target, predictions[mode])
            values[mode] = result.iloc[0].to_dict() if not result.empty else {}
            for metric in metrics:
                if metric in COMPUTATIONAL_METRICS:
                    values[mode][metric] = MetricFactory.get_metric(metric).get_value(model, evaluation_loader(data, split))
        frame = pd.DataFrame(values)
        original, compressed = frame['original'], frame['fedcore']
        frame['change'] = ((compressed - original) / original * 100).round(2)
        frame.loc[(original == 0) & (compressed == 0), 'change'] = 0.0
        frame.index.name = 'metric'
        # Preserve the original two-level report columns.
        frame.columns = pd.MultiIndex.from_product([[0], frame.columns], names=[None, 'mode'])
        self.metric_dict = frame
        return frame

    def save(self, mode='all', **kwargs):
        modes = ('model', 'metrics', 'prediction', 'opt_hist')
        if mode not in (*modes, 'all'):
            raise ValueError(f'Unsupported save mode: {mode!r}')
        folder = Path(kwargs.get('path', self.manager.compute_config.output_folder))
        folder.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for kind in modes if mode == 'all' else (mode,):
            if kind == 'model':
                model = self.compressed_model or self.original_model
                if model is None:
                    raise ValueError('No model is available to save')
                target = folder / 'model.pt'
                manager = CheckpointManager(str(folder), auto_cleanup=False)
                manager.save_to_file(manager.serialize_to_bytes(model), str(target))
            elif kind == 'metrics':
                if self.metric_dict is None:
                    if mode == 'all':
                        continue
                    raise ValueError('Call get_report before saving metrics')
                target = folder / 'metrics.csv'
                self.metric_dict.to_csv(target)
            elif kind == 'prediction':
                output = getattr(self.manager, 'predicted_labels', None)
                if output is None:
                    if mode == 'all':
                        continue
                    raise ValueError('Call predict before saving predictions')
                target = folder / 'labels.csv'
                pd.DataFrame(torch.as_tensor(output.predict).cpu().numpy()).to_csv(target, index=False)
            else:
                history = getattr(self.manager.solver, 'history', None)
                if history is None:
                    if mode == 'all':
                        continue
                    raise ValueError('Optimization history is unavailable')
                target = folder / 'optimization_history.json'
                history.save(str(target))
            artifacts[kind] = target
        return artifacts

    def load(self, path, *, model=None, model_factory=None):
        target = Path(path)
        if target.is_dir():
            target = target / 'model.pt'
        restored = CheckpointManager(str(target.parent), auto_cleanup=False).load_from_file(
            str(target), 'cpu', model=model, model_factory=model_factory)
        self.fedcore_model = restored
        self.__original_model = deepcopy(restored)
        return restored

    def shutdown(self):
        if not self._owns_dask:
            return
        self._owns_dask = False
        client = getattr(self.manager, 'dask_client', None)
        cluster = getattr(self.manager, 'dask_cluster', None)
        self.manager.dask_client = self.manager.dask_cluster = None
        try:
            if client is not None:
                client.close()
        finally:
            if cluster is not None:
                cluster.close()

    def export(
            self,
            framework: str = "ONNX",
            framework_config: dict = None,
            supplementary_data: dict = None,
    ):
        """Export a model to torchscript, ONNX, or TensorRT.

        Parameters
        ----------
        framework :
            Target format. Supported: ``torchscript`` / ``pt``, ``onnx``,
            ``tensorrt`` / ``engine``. Unknown names raise ValueError.
            TensorRT raises if the SDK is missing.
        framework_config :
            Export options: ``output_path``, ``example_inputs``,
            ``opset_version``, ``input_names``, ``output_names``,
            ``dynamic_axes``, ``do_constant_folding``, ``workspace_size``.
        supplementary_data :
            Must provide ``model_to_export``. Optional ``example_inputs`` /
            ``dummy_input``; if omitted, a default ``(1, 3, 224, 224)``
            tensor is generated.

        Returns
        -------
        pathlib.Path
            Path to the written artifact (``.pt``, ``.onnx``, or ``.engine``).
        """
        from fedcore.tools.export import export_model, default_output_path
        framework_config = dict(framework_config or {})
        supplementary_data = dict(supplementary_data or {})

        model = supplementary_data.get("model_to_export")
        if model is None:
            raise ValueError(
                "supplementary_data['model_to_export'] is required for export"
            )

        example_input = framework_config.get("example_inputs")
        if example_input is None:
            example_input = supplementary_data.get("example_inputs")
        if example_input is None:
            example_input = supplementary_data.get("dummy_input")

        output_path = framework_config.get("output_path")
        if output_path is None:
            output_path = default_output_path(framework)

        return export_model(
            model,
            framework,
            output_path,
            example_input,
            framework_config,
        )
