"""Pure immutable request validation and planning. All paths are job-relative."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath, PureWindowsPath


class ContractError(ValueError):
    def __init__(self, code: str, message: str, path: str = "request"):
        super().__init__(message)
        self.code, self.path = code, path

    def to_dict(self):
        return {"code": self.code, "path": self.path, "message": str(self)}


def relative_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ContractError("invalid_path", "Only nonempty relative POSIX artifact names are accepted", "path")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute() or any(p in ("..", ".") for p in value.split("/")):
        raise ContractError("invalid_path", "Artifact path must remain inside the job directory", "path")
    return value


def _object(payload, allowed, required=()):
    if not isinstance(payload, dict) or set(payload) - set(allowed) or set(required) - set(payload):
        raise ContractError("invalid_schema", f"Expected object with fields {sorted(allowed)}")
    return payload


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ContractError("invalid_value", f"{name} must be an integer in [{low}, {high}]", name)
    return value


@dataclass(frozen=True)
class InputSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"
    name: str = "input"

    def __post_init__(self):
        if not isinstance(self.shape, tuple) or not 1 <= len(self.shape) <= 5:
            raise ContractError("unsupported_input", "v1 supports one tensor with 1–5 axes", "input_spec")
        for size in self.shape:
            _integer(size, 1, 1000000, "input_spec.shape")
        if math.prod(self.shape) > 100000000 or self.dtype not in ("float32", "float64") or self.name != "input":
            raise ContractError("unsupported_input", "InputSpec exceeds the tensor limits or supported dtype/name", "input_spec")

    @classmethod
    def parse(cls, payload):
        p = _object(payload, ("shape", "dtype", "name"), ("shape",))
        if not isinstance(p["shape"], (list, tuple)):
            raise ContractError("unsupported_input", "Container and multi-input schemas are unsupported", "input_spec")
        return cls(tuple(p["shape"]), p.get("dtype", "float32"), p.get("name", "input"))

    def validate_tensor(self, tensor, *, batch_dynamic=False):
        import torch
        if not isinstance(tensor, torch.Tensor):
            raise ContractError("unsupported_input", "Input must be one tensor", "input")
        shape = tuple(tensor.shape)
        matches = shape[1:] == self.shape[1:] if batch_dynamic else shape == self.shape
        if not matches or tensor.ndim != len(self.shape) or str(tensor.dtype) != "torch." + self.dtype:
            raise ContractError("input_mismatch", "Tensor does not match the declared InputSpec", "input")
        if not tensor.numel() or not torch.isfinite(tensor).all():
            raise ContractError("invalid_input", "Input must contain finite values", "input")


@dataclass(frozen=True)
class DeviceProfile:
    name: str = "CPU"
    supported_ops: tuple[str, ...] = ()
    unsupported_ops: tuple[str, ...] = ()
    cpu_framework: str = "torchscript"
    npu_framework: str = "onnx"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or len(self.name) > 128:
            raise ContractError("invalid_profile", "Profile requires a bounded name")
        for ops in (self.supported_ops, self.unsupported_ops):
            if not isinstance(ops, tuple) or len(ops) > 1000 or any(not isinstance(op, str) or not op or len(op) > 64 for op in ops):
                raise ContractError("invalid_profile", "Operations must be immutable tuples of names")
        if self.cpu_framework not in ("torchscript", "onnx", "tensorrt") or self.npu_framework not in ("torchscript", "onnx", "tensorrt"):
            raise ContractError("unsupported_backend", "Profile names an unimplemented export backend")

    @classmethod
    def parse(cls, payload):
        p = _object(payload, ("name", "supported_ops", "unsupported_ops", "cpu_framework", "npu_framework"))
        if any(not isinstance(p.get(key, ()), (list, tuple)) for key in ("supported_ops", "unsupported_ops")):
            raise ContractError("invalid_profile", "Profile operation fields must be lists")
        return cls(p.get("name", "CPU"), tuple(p.get("supported_ops", ())), tuple(p.get("unsupported_ops", ())),
                   p.get("cpu_framework", "torchscript"), p.get("npu_framework", "onnx"))

    def supports(self, operation):
        return operation in self.supported_ops and operation not in self.unsupported_ops


@dataclass(frozen=True)
class DataRoles:
    validation: str
    train: str | None = None
    calibration: str | None = None

    def __post_init__(self):
        if self.validation is None:
            raise ContractError("missing_data_roles", "Validation data is required", "data.validation")
        paths = [relative_name(v) for v in (self.validation, self.train, self.calibration) if v is not None]
        if len(paths) != len(set(paths)):
            raise ContractError("overlapping_data_roles", "Train, validation and calibration must be separate artifacts", "data")


@dataclass(frozen=True)
class Resources:
    timeout_seconds: float = 120.0
    max_bytes: int = 64 * 1024 * 1024
    threads: int = 1
    repetitions: int = 5
    device: str = "cpu"

    def __post_init__(self):
        if type(self.timeout_seconds) not in (float, int) or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 3600:
            raise ContractError("invalid_resources", "timeout_seconds must be finite and in (0, 3600]")
        _integer(self.max_bytes, 1024, 1024 * 1024 * 1024, "max_bytes")
        _integer(self.threads, 1, 16, "threads")
        _integer(self.repetitions, 1, 1000, "repetitions")
        if self.device != "cpu":
            raise ContractError("unsupported_device", "External runtime v1 supports CPU execution only")


@dataclass(frozen=True)
class WeightedOptions:
    """Versioned CPU weighted-SVD profile; never interpreted as ordinary SVD."""
    ridge: float = 0.0
    rcond: float | None = None
    nullspace_policy: str = "support_only"
    batch_size: int = 16
    max_workspace_bytes: int = 256 * 1024 * 1024
    max_peak_bytes: int = 12 * 1024 * 1024 * 1024
    method_version: int = 1

    def __post_init__(self):
        if type(self.method_version) is not int or self.method_version != 1:
            raise ContractError("unsupported_method_version", "Only weighted-SVD method version 1 is supported")
        for name, value in (("ridge", self.ridge), ("rcond", self.rcond)):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ContractError("invalid_metric", f"{name} must be finite and nonnegative", name)
        if self.ridge is None or self.nullspace_policy not in ("support_only", "preserve_nullspace"):
            raise ContractError("invalid_metric", "Weighted profile requires an explicit supported metric policy")
        _integer(self.batch_size, 1, 100000, "weighted.batch_size")
        _integer(self.max_workspace_bytes, 1024, 12 * 1024 ** 3, "weighted.max_workspace_bytes")
        _integer(self.max_peak_bytes, 1024, 12 * 1024 ** 3, "weighted.max_peak_bytes")
        if self.max_workspace_bytes > self.max_peak_bytes:
            raise ContractError("invalid_resources", "Workspace cannot exceed the process planning limit")

    @classmethod
    def parse(cls, payload):
        return cls(**_object(payload, tuple(cls.__dataclass_fields__)))


@dataclass(frozen=True)
class CompressionRequest:
    model: str
    example: str
    input_spec: InputSpec
    data: DataRoles
    task: str = "regression"
    method: str = "svd"
    rank: int | None = None
    retained_energy: float = 1.0
    max_relative_error: float = 1e-4
    artifact_format: str = "torchscript"
    profile: DeviceProfile = DeviceProfile()
    resources: Resources = Resources()
    version: int = 1
    weighted: WeightedOptions | None = None

    def __post_init__(self):
        relative_name(self.model)
        relative_name(self.example)
        if type(self.version) is not int or self.version not in (1, 2):
            raise ContractError("unsupported_version", "Only contract versions 1 and 2 are supported")
        if self.task not in ("classification", "regression") or self.method not in ("svd", "weighted_svd", "export"):
            raise ContractError("unsupported_method", "Only declared tensor classification/regression profiles are supported")
        if self.method == "weighted_svd":
            if self.version != 2 or not isinstance(self.weighted, WeightedOptions):
                raise ContractError("unsupported_profile", "Weighted SVD requires contract v2 and WeightedOptions")
            if not isinstance(self.data, DataRoles) or self.data.calibration is None:
                raise ContractError("missing_calibration", "Weighted SVD requires a separate calibration artifact", "data.calibration")
            if self.rank is None:
                raise ContractError("invalid_rank", "External weighted profile requires an explicit rank", "rank")
        elif self.version != 1 or self.weighted is not None:
            raise ContractError("unsupported_profile", "Ordinary SVD/export keep the existing v1 contract")
        if self.method == "export" and (self.rank is not None or self.retained_energy != 1):
            raise ContractError("conflicting_rank", "Export requests cannot set compression parameters")
        if self.artifact_format not in ("torchscript", "onnx"):
            raise ContractError("unsupported_backend", "External v1 supports TorchScript and ONNX artifacts")
        if self.rank is not None:
            _integer(self.rank, 1, 100000, "rank")
            if self.retained_energy != 1:
                raise ContractError("conflicting_rank", "Set either rank or retained_energy, not both")
        for name, value, low, high in (("retained_energy", self.retained_energy, 0, 1),
                                       ("max_relative_error", self.max_relative_error, -1e-300, 1e6)):
            if type(value) not in (float, int) or not math.isfinite(value) or not low < value <= high:
                raise ContractError("invalid_value", f"Invalid {name}", name)
        if not isinstance(self.input_spec, InputSpec) or not isinstance(self.data, DataRoles) or not isinstance(self.profile, DeviceProfile) or not isinstance(self.resources, Resources):
            raise ContractError("invalid_schema", "Request fields must be validated contract values")

    def to_dict(self):
        result = asdict(self)
        if self.weighted is None:
            result.pop("weighted")
        return result

    @classmethod
    def parse(cls, payload):
        allowed = tuple(cls.__dataclass_fields__)
        p = _object(payload, allowed, ("model", "example", "input_spec", "data", "version"))
        roles = _object(p["data"], ("validation", "train", "calibration"), ("validation",))
        resources = _object(p.get("resources", {}), tuple(Resources.__dataclass_fields__))
        weighted = WeightedOptions.parse(p["weighted"]) if p.get("weighted") is not None else None
        return cls(**{k: v for k, v in p.items() if k not in ("input_spec", "data", "profile", "resources", "weighted")},
                   input_spec=InputSpec.parse(p["input_spec"]), data=DataRoles(**roles),
                   profile=DeviceProfile.parse(p.get("profile", {})), resources=Resources(**resources), weighted=weighted)


@dataclass(frozen=True)
class ExecutionPlan:
    request: CompressionRequest
    steps: tuple[str, ...] = ("load_safe_model", "validate_inputs", "svd", "evaluate", "export", "record_provenance")
    artifact_name: str = "compressed.pt"


def plan_request(request: CompressionRequest) -> ExecutionPlan:
    if not isinstance(request, CompressionRequest):
        raise ContractError("invalid_schema", "Planner consumes a validated CompressionRequest")
    steps = ("load_safe_model", "validate_inputs", "evaluate", "export", "record_provenance")
    if request.method in ("svd", "weighted_svd"):
        steps = steps[:2] + (request.method,) + steps[2:]
    return ExecutionPlan(request, steps=steps, artifact_name="compressed.pt" if request.artifact_format == "torchscript" else "compressed.onnx")
