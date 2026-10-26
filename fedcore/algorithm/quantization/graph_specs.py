"""Metadata-only process, representation and execution contracts."""
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math

from .tensor_formats import QuantizedTensorSpec, Grid, Rounding
from fedcore.algorithm.low_rank.plans import ValidationFailure, Violation
from fedcore.algorithm.low_rank.compression_graph import CalibrationRef,SourceMode


class Method(str, Enum):
    DENSE = "dense"
    PTQ = "weight_ptq"
    DSVD = "dsvd"
    SVD = "real_svd"
    FACTORS = "quantize_ready_factors"
    LPLR = "lplr"
    LPLR_SVD = "lplr_svd"
    CALDERA = "caldera_cpu_scalar"
    TENDER = "tender_linear_reference"


@dataclass(frozen=True)
class ExecutionTargetSpec:
    runtime: str = "cpu_reference"
    allow_float_decode: bool = True


@dataclass(frozen=True)
class TransformSpec:
    method: Method
    rank: int | None = None
    comparison_rank: int | None = None
    left: QuantizedTensorSpec = QuantizedTensorSpec()
    right: QuantizedTensorSpec = QuantizedTensorSpec()
    weight: QuantizedTensorSpec = QuantizedTensorSpec(axis=0)
    seed: int = 0
    rcond: float = 1e-10
    ridge: float = 0.0
    outer_iterations: int = 2
    inner_iterations: int = 2
    activation_bits: int = 8
    accumulator_bits: int = 64
    prequantize: QuantizedTensorSpec | None = None


@dataclass(frozen=True)
class ProcessNode:
    name: str
    operation: str
    inputs: tuple[str, ...]
    semantic: str = "weight_out_in"


@dataclass(frozen=True)
class ProcessGraphSpec:
    nodes: tuple[ProcessNode, ...]
    output: str
    version: int = 1


@dataclass(frozen=True)
class RepresentationSpec:
    kind: str
    shape: tuple[int, int]
    rank: int | None
    bias_count: int


@dataclass(frozen=True)
class OperatorRecipe:
    path: str
    transform: TransformSpec


@dataclass(frozen=True)
class QuantizationGraphRequest:
    recipes: tuple[OperatorRecipe, ...]
    target: ExecutionTargetSpec = ExecutionTargetSpec()
    max_storage_bytes: int | None = None
    max_peak_bytes: int = 1024**3
    source_mode: SourceMode = SourceMode.FROZEN_SNAPSHOT


@dataclass(frozen=True)
class CheckedCompressionPlan:
    request: QuantizationGraphRequest
    source_fingerprint: str
    aliases: tuple[tuple[str, ...], ...]
    processes: tuple[ProcessGraphSpec, ...]
    representations: tuple[RepresentationSpec, ...]
    fingerprint: str
    calibration: CalibrationRef | None = None
    schema: str = "quantization_graph_v1"


def payload(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: payload(item) for key, item in asdict(value).items()}
    if type(value) in (tuple, list):
        return [payload(item) for item in value]
    if type(value) is dict:
        return {key: payload(item) for key, item in value.items()}
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(payload(value), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def recipe_graph(spec):
    """Compile meaningful operations, preserving noncommuting Q(L) -> fit."""
    nodes=[]
    def add(op,*inputs):
        name=f'n{len(nodes)}';nodes.append(ProcessNode(name,op,tuple(inputs)));return name
    if spec.method is Method.FACTORS:
        factors=add('ExtractFactors');left=add('QuantizeLeftFactor',factors);right=add('QuantizeRightFactor',factors)
        output=add('BuildFactorOperator',left,right)
    else:
        weight=add('ExtractMatrix')
        if spec.prequantize is not None:
            weight=add('DecodeTensor',add('PreQuantizeTensor',weight))
        if spec.method is Method.DENSE:output=add('BuildDenseOperator',weight)
        elif spec.method is Method.PTQ:output=add('BuildQuantizedOperator',add('QuantizeWeight',weight))
        elif spec.method in (Method.SVD,Method.DSVD):
            factors=add('SVDTruncate',weight)
            if spec.method is Method.SVD:output=add('BuildRealFactorOperator',factors)
            else:output=add('BuildFactorOperator',add('QuantizeLeftFactor',factors),add('QuantizeRightFactor',factors))
        elif spec.method in (Method.LPLR,Method.LPLR_SVD):
            first=add('RandomProject' if spec.method is Method.LPLR else 'SVDLeft',weight)
            left=add('QuantizeLeft',first)
            decoded=add('DecodeTensor',left)
            fit=add('LeastSquaresFit',decoded,weight)
            output=add('BuildFactorOperator',left,add('QuantizeRight',fit))
            output=add('EvaluateWeightError',output,weight)
        elif spec.method is Method.TENDER:
            rows=add('ObserveInput');output=add('BuildTenderOperator',add('BuildChannelGroups',weight,rows))
        else:
            rows=add('ObserveInput');metric=add('BuildMetric',rows)
            state=add('InitializeSum',weight,metric)
            for _ in range(spec.outer_iterations):
                state=add('SelectBest',add('LDLQ',state,weight,metric),weight,metric)
                state=add('SelectBest',add('WeightedFactorFit',state,weight,metric),weight,metric)
                for _ in range(spec.inner_iterations):
                    state=add('RegressRight',state,weight,metric)
                    state=add('SelectBest',add('RegressLeft',state,weight,metric),weight,metric)
            output=add('BuildSum',state)
    return ProcessGraphSpec(tuple(nodes),output)


def canonical_graph(graph):
    """Topologically canonical names; operation order and ordered ports survive."""
    if type(graph) is not ProcessGraphSpec or type(graph.version) is not int or graph.version != 1:
        raise ValueError("Unknown process graph version")
    names, result = {}, []
    for i, node in enumerate(graph.nodes):
        if (type(node) is not ProcessNode or node.name in names or not node.operation
                or any(ref not in names for ref in node.inputs)):
            raise ValueError("DAG must have unique names and topological, closed inputs")
        new = f"n{i}"
        result.append(ProcessNode(new, node.operation, tuple(names[ref] for ref in node.inputs), node.semantic))
        names[node.name] = new
    if graph.output not in names:
        raise ValueError("Unknown graph output")
    return ProcessGraphSpec(tuple(result), names[graph.output])


def validate_transform(spec, shape):
    problems = []
    def bad(message):
        problems.append(Violation("InvalidTransform", "transform", message))
    if type(spec) is not TransformSpec or type(spec.method) is not Method:
        return ValidationFailure((Violation("InvalidTransform", "transform", "Typed method required"),))
    if spec.method in (Method.SVD, Method.DSVD, Method.LPLR, Method.LPLR_SVD, Method.CALDERA):
        if type(spec.rank) is not int or not 1 <= spec.rank <= min(shape):
            bad("Factor width must fit both matrix axes")
    elif spec.rank is not None:
        bad("This representation has no rank")
    if spec.prequantize is not None and (type(spec.prequantize) is not QuantizedTensorSpec or spec.method in (Method.FACTORS,Method.DENSE,Method.TENDER)):
        bad('Prequantize/decode subgraph is unsupported for this source/target')
    if spec.comparison_rank is not None and (type(spec.comparison_rank) is not int or not 1 <= spec.comparison_rank <= min(shape)):
        bad("Comparison rank must fit matrix axes independently of factor width")
    if any(type(s) is not QuantizedTensorSpec for s in (spec.left, spec.right, spec.weight)):
        bad("Typed tensor formats required")
    else:
        if spec.method in (Method.LPLR, Method.LPLR_SVD) and any(s.grid is not Grid.ENDPOINT or s.rounding is not Rounding.STOCHASTIC for s in (spec.left,spec.right)):
            bad("LPLR requires endpoint stochastic factor grids")
        if spec.method is Method.CALDERA and spec.weight.axis not in (None,0):
            bad("Scalar LDLQ supports only tensor/output-channel weight scales")
        if spec.method is Method.TENDER and (spec.weight.grid is not Grid.SYMMETRIC or spec.weight.rounding is not Rounding.NEAREST or spec.weight.axis != 0 or spec.weight.bound is not None):
            bad("Tender supports symmetric nearest output-channel weight calibration only")
    if type(spec.seed) is not int or not 0 <= spec.seed < 2**63:
        bad("Bounded nonnegative RNG seed required")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (spec.rcond, spec.ridge)):
        bad("Finite nonnegative numerical thresholds required")
    if any(type(v) is not int or not 1 <= v <= 32 for v in (spec.outer_iterations, spec.inner_iterations)):
        bad("Bounded positive iteration counts required")
    if spec.activation_bits not in (2, 4, 8) or type(spec.activation_bits) is not int or spec.accumulator_bits not in (16, 32, 64):
        bad("Supported activation/accumulator precision required")
    return ValidationFailure(tuple(problems)) if problems else None
