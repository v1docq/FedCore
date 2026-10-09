"""Optional, explicit IndustrialTS/FEDOT and Tensor_decompose boundaries."""
from __future__ import annotations
from pathlib import Path
import torch
from .client import compress
from .contracts import ContractError, InputSpec, Resources, CompressionRequest, DataRoles
from .models import load_model_bundle
from .security import safe_load


class CompressedTensorEstimator:
    """Compress a fitted safe model, retaining the caller's tensor preparation.

    ``fit`` data has the training role; an independent, labeled validation bundle
    is mandatory. No scaler, windowing, or horizon transformation is introduced.
    """
    def __init__(self, params, task):
        self.params = dict(params or {})
        self.task = task
        required = {"model_bundle", "validation_bundle", "jobs_root"}
        allowed = required | {"rank", "retained_energy", "max_relative_error", "python_executable", "resources"}
        if required - self.params.keys() or self.params.keys() - allowed:
            raise ContractError("invalid_adapter_params", "Adapter requires model_bundle, validation_bundle and jobs_root")
        if any(not isinstance(self.params[key], (str, Path)) or not str(self.params[key]) for key in required):
            raise ContractError("invalid_adapter_params", "Adapter paths must be explicit nonempty paths")
        if "python_executable" in self.params and (not isinstance(self.params["python_executable"], (str, Path)) or not str(self.params["python_executable"])):
            raise ContractError("invalid_adapter_params", "python_executable must be an explicit path")
        try:
            self.resources = Resources(**self.params.get("resources", {}))
        except TypeError as error:
            raise ContractError("invalid_adapter_params", "Invalid resource parameter schema") from error
        CompressionRequest("model.fcb", "example.fcb", InputSpec((1, 1)), DataRoles("validation.fcb"), task=task,
                           rank=self.params.get("rank"), retained_energy=self.params.get("retained_energy", 1.),
                           max_relative_error=self.params.get("max_relative_error", 1e-4), resources=self.resources)
        self.model = None
        self.result = None
        self.input_spec = None

    def fit(self, features, target):
        x = torch.as_tensor(features).detach().cpu()
        y = torch.as_tensor(target).detach().cpu()
        if self.task == "classification":
            y = y.to(torch.int64)
        base = load_model_bundle(self.params["model_bundle"])
        obj = safe_load(self.params["validation_bundle"])
        if not isinstance(obj, dict) or obj.get("kind") != "fedcore_tensor_dataset" or obj.get("version") != 1:
            raise ContractError("invalid_dataset", "Adapter needs a separate v1 validation tensor bundle")
        validation = obj["features"], obj["targets"]
        if x.ndim == 0 or not len(x):
            raise ContractError("invalid_input", "Training features require a nonempty sample axis")
        self.input_spec = InputSpec(tuple(x[:1].shape), str(x.dtype).removeprefix("torch."))
        self.result = compress(base, x[:1], validation, jobs_root=self.params["jobs_root"], task=self.task,
                               train=(x, y), rank=self.params.get("rank"), retained_energy=self.params.get("retained_energy", 1.0),
                               max_relative_error=self.params.get("max_relative_error", 1e-4),
                               resources=self.resources,
                               python_executable=self.params.get("python_executable"))
        if self.result.get("status") != "succeeded":
            error = self.result.get("error", {})
            raise ContractError(error.get("code", "compression_failed"), error.get("message", "Compression failed"))
        artifact = Path(self.result["job_directory"]) / self.result["artifact"]
        with artifact.open("rb") as stream:
            self.model = torch.jit.load(stream, map_location="cpu").eval()
        return self

    def _scores(self, features):
        if self.model is None:
            raise ContractError("not_fitted", "Fit the compression adapter before prediction")
        x = torch.as_tensor(features)
        self.input_spec.validate_tensor(x, batch_dynamic=True)
        with torch.inference_mode():
            return self.model(x)

    def predict(self, features):
        scores = self._scores(features)
        return (scores.argmax(1) if self.task == "classification" else scores).numpy()

    def predict_proba(self, features):
        if self.task != "classification":
            raise ContractError("unsupported_operation", "Regression adapter has no class probabilities")
        return self._scores(features).softmax(1).numpy()


def build_industrial_extension_manifest():
    """Use the actual modern FEDOT ExtensionManifest used by IndustrialTS.

    Discovery neither publishes to a registry nor imports ``fedot_ind``. The
    caller validates and smoke-tests this manifest in an isolated registry.
    """
    try:
        from fedot.extensions import ArrayBackend, ExtensionManifest, ExternalModelSpec, ModelCapabilities, ModelHyperparamsSchema
        from fedot.core.repository.dataset_types import DataTypesEnum
        from fedot.core.repository.tasks import TaskTypesEnum
    except ImportError as error:
        raise ContractError("extension_contract_unavailable", "Industrial adapter requires FEDOT's modern ExtensionManifest API") from error

    def classification_factory(params=None):
        return CompressedTensorEstimator(params, "classification")

    def regression_factory(params=None):
        return CompressedTensorEstimator(params, "regression")

    schema = ModelHyperparamsSchema(required=("model_bundle", "validation_bundle", "jobs_root"),
                                    optional=("rank", "retained_energy", "max_relative_error", "python_executable", "resources"), defaults={})
    models = tuple(ExternalModelSpec(
        name=f"fedcore_compressed_{task}", factory=factory,
        capabilities=ModelCapabilities(tasks=(TaskTypesEnum[task],),
                                       data_types=(DataTypesEnum.table, DataTypesEnum.image, DataTypesEnum.ts),
                                       tags=("compression", "pretrained_tensor", "cpu"), supports_multimodal=False,
                                       backend=ArrayBackend.numpy, output_data_type=DataTypesEnum.table if task == "classification" else None, requires_target=True),
        hyperparams_schema=schema, description="SVD compression of allowlisted fitted Linear/Conv tensor architectures")
        for task, factory in (("classification", classification_factory), ("regression", regression_factory)))
    return ExtensionManifest(name="fedcore_compression", version="1.0.0", models=models,
                             module=__name__, description="Isolated compression; caller owns preprocessing and independent validation")


def decompose_matrix_with_tdecomp(matrix, *, rank):
    """Tensor_decompose's matrix API, imported only at this optional boundary.

    This adapter lives in FedCore, never in tdecomp, preserving dependency order.
    It returns actual factors and reconstruction with a measured relative error.
    """
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2 or matrix.dtype not in (torch.float32, torch.float64):
        raise ContractError("unsupported_matrix", "Expected a float32/float64 matrix tensor")
    if type(rank) is not int or not 1 <= rank <= min(matrix.shape) or not torch.isfinite(matrix).all():
        raise ContractError("invalid_rank", "Matrix must be finite and rank must fit both axes")
    try:
        from tdecomp.matrix.decomposer import SVDDecomposition
    except ImportError as error:
        raise ContractError("dependency_unavailable", "Install tdecomp in the compression environment") from error
    decomposer = SVDDecomposition()
    factors = decomposer.decompose(matrix, rank=rank)
    # Published tdecomp 0.2.18 returns full SVD even when rank is provided.
    if len(factors) != 3 or factors[1].ndim != 1:
        raise ContractError("invalid_factors", "SVD adapter requires canonical U/S/Vh factors")
    factors = (factors[0][:, :rank], factors[1][:rank], factors[2][:rank, :])
    restored = decomposer.compose(*factors)
    norm = torch.linalg.vector_norm(matrix)
    error = torch.linalg.vector_norm(restored-matrix)
    relative = float(error / norm) if norm > 0 else (0.0 if error == 0 else float("inf"))
    return {"factors": factors, "reconstruction": restored, "relative_error": relative,
            "method": "tdecomp.SVDDecomposition", "rank": rank}
