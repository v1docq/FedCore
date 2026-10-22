"""Metadata-only contracts for two bounded compression recipes.

This is a workflow graph, not a neural-network graph or a search genotype.
G1 defaults to a whole checked recipe. The explicit G2a staged interpreter
executes these same canonical nodes and owns its statistics store outside the
metadata plan. Neither mode permits an arbitrary user-composed DAG.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json

from .method_specs import ASVD
from .plans import (MetricPolicy, OperatorDescriptor, ResourceLimits, StepSpec,
                    TransformPlan, TransformRequest,
                    ValidationFailure, Violation, _metadata_only, plan_transform)


class Recipe(str, Enum):
    WEIGHTED_SVD = "weighted_svd"
    ASVD = "asvd"


class SourceMode(str, Enum):
    FROZEN_SNAPSHOT = "frozen_snapshot"
    CURRENT_STUDENT = "current_student"


class ArtifactKind(str, Enum):
    MODEL = "model_version"
    CALIBRATION = "calibration_observations"
    INPUT_ABS_MEAN = "input_absolute_mean"
    INPUT_ABS_MAX = "input_absolute_maximum"
    INPUT_SECOND_MOMENT = "input_uncentered_second_moment"
    # These are deliberately different semantic types, not alternate spellings
    # of the statistics accepted by the two G1 recipes.
    INPUT_MEAN = "input_mean"
    OUTPUT_MEAN = "output_mean"
    CENTERED_INPUT_COVARIANCE = "centered_input_covariance"
    CENTERED_OUTPUT_COVARIANCE = "centered_output_covariance"
    DIAGONAL_METRIC = "diagonal_input_metric"
    FULL_METRIC = "full_input_metric"
    OUTPUT_PROJECTOR = "centered_output_projector"
    APPROXIMATION = "low_rank_approximation"
    CHANNEL_SELECTION = "selected_structural_channels"
    REPLACEMENT = "operator_replacement"


class Stage(str, Enum):
    OBSERVE = "observe"
    METRIC = "build_metric"
    SOLVE = "solve"
    BUILD = "build_replacement"
    APPLY = "apply_replacements"


@dataclass(frozen=True)
class ModelVersion:
    graph_version: str
    fingerprint: str


@dataclass(frozen=True)
class CalibrationRef:
    fingerprint: str
    shape: tuple[int, ...]
    dtype: str
    role: str = "calibration"


@dataclass(frozen=True)
class OperatorScope:
    operator: OperatorDescriptor
    aliases: tuple[str, ...]
    coordinates: str = "original_operator_input_output"


@dataclass(frozen=True)
class ArtifactRef:
    key: str
    kind: ArtifactKind
    model_version: str
    scope: tuple[str, ...] = ()
    coordinates: str = "original_operator_input_output"


@dataclass(frozen=True)
class Port:
    name: str
    kind: ArtifactKind
    artifact: str


@dataclass(frozen=True)
class CompressionNode:
    key: str
    stage: Stage
    scope: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    inputs: tuple[Port, ...]
    outputs: tuple[ArtifactRef, ...]


@dataclass(frozen=True)
class RuntimeOptions:
    batch_size: int = 16
    max_workspace_bytes: int = 256 * 1024**2
    max_peak_bytes: int = 12 * 1024**3
    parameter_budget: int | None = None
    tensor_byte_budget: int | None = None


@dataclass(frozen=True)
class FixedCompressionGraph:
    recipe: Recipe
    source_mode: SourceMode
    source: ModelVersion
    calibration: CalibrationRef
    transform: TransformPlan
    options: RuntimeOptions
    method_spec: ASVD | None
    module_aliases: tuple[tuple[str, ...], ...]
    scopes: tuple[OperatorScope, ...]
    external_artifacts: tuple[ArtifactRef, ...]
    nodes: tuple[CompressionNode, ...]
    fingerprint: str
    schema_version: int = 1
    executor: str = "checked_whole_recipe_v1"


def _failure(code, path, message):
    return ValidationFailure((Violation(code, path, message),))


def _canonical(value):
    """Internal digest encoding; this is not an external JSON decoding API."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    return value


def _check_operator_plan(request, operators, limits):
    try:
        return plan_transform(request, operators, limits)
    except OverflowError:
        return _failure("InvalidTransform", "transform", "Numeric metadata exceeds the supported finite domain")


def graph_fingerprint(graph):
    """Content identifier, not a signature or a substitute for validation."""
    if not isinstance(graph, FixedCompressionGraph) or not _metadata_only(graph):
        raise ValueError("A metadata-only FixedCompressionGraph is required")
    payload = _canonical(replace(graph, fingerprint=""))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _check_inputs(recipe, transform, source, calibration, options, method_spec, aliases):
    values = (recipe, transform, source, calibration, options, method_spec, aliases)
    if not _metadata_only(values):
        return _failure("LiveResource", "graph", "Only immutable metadata and references may enter a graph")
    if type(recipe) is not Recipe:
        return _failure("UnsupportedRecipe", "recipe", "Only weighted_svd and asvd fixed recipes are executable")
    if type(source) is not ModelVersion or type(source.graph_version) is not str or not source.graph_version:
        return _failure("InvalidSource", "source", "An explicit original graph version is required")
    if not _digest(source.fingerprint):
        return _failure("InvalidSource", "source.fingerprint", "A SHA-256 source fingerprint is required")
    if (type(calibration) is not CalibrationRef or not _digest(calibration.fingerprint)
            or calibration.role != "calibration" or type(calibration.shape) is not tuple
            or len(calibration.shape) < 2 or any(type(d) is not int or d <= 0 for d in calibration.shape)
            or calibration.dtype not in ("float32", "float64")):
        return _failure("InvalidCalibration", "calibration", "Separate nonempty CPU float calibration metadata is required")
    if type(options) is not RuntimeOptions:
        return _failure("InvalidOptions", "options", "RuntimeOptions required")
    for name in ("batch_size", "max_workspace_bytes", "max_peak_bytes"):
        if type(getattr(options, name)) is not int or getattr(options, name) <= 0:
            return _failure("InvalidOptions", "options." + name, "A positive integer is required")
    if options.max_peak_bytes > 12 * 1024**3 or options.max_workspace_bytes > options.max_peak_bytes:
        return _failure("InvalidOptions", "options", "Workspace must fit a declared peak of at most 12 GiB")
    for name in ("parameter_budget", "tensor_byte_budget"):
        value = getattr(options, name)
        if value is not None and (type(value) is not int or value <= 0):
            return _failure("InvalidOptions", "options." + name, "An optional budget must be a positive integer")
    if recipe is Recipe.WEIGHTED_SVD:
        if method_spec is not None or options.parameter_budget is not None or options.tensor_byte_budget is not None:
            return _failure("UnsupportedOptions", "options", "Weighted G1 has no method spec or method-only budget")
    elif type(method_spec) is not ASVD:
        return _failure("InvalidMethod", "method_spec", "The ASVD recipe requires an explicit ASVD description")
    else:
        try:
            # Frozen dataclasses can still be forged via object.__setattr__.
            ASVD(**{field.name: getattr(method_spec, field.name) for field in fields(method_spec)})
        except (ValueError, TypeError):
            return _failure("InvalidMethod", "method_spec", "Invalid ASVD parameters")
    if (type(aliases) is not tuple or any(type(group) is not tuple or len(group) < 2
            or any(type(path) is not str for path in group) or len(set(group)) != len(group) for group in aliases)
            or len({path for group in aliases for path in group}) != sum(map(len, aliases))):
        return _failure("InvalidScope", "module_aliases", "Module alias groups must be disjoint immutable path tuples")
    if (type(transform) is not TransformPlan or type(transform.schema_version) is not int
            or transform.schema_version != 1 or type(transform.steps) is not tuple or not transform.steps
            or type(transform.request) is not TransformRequest
            or type(transform.resource_limits) is not ResourceLimits):
        return _failure("InvalidTransform", "transform", "A nonempty schema-1 checked operator plan is required")
    if transform.resource_limits.max_peak_bytes != options.max_peak_bytes:
        return _failure("InvalidOptions", "options.max_peak_bytes", "Graph and operator resource limits must agree")
    expected_objective = "input_second_moment" if recipe is Recipe.WEIGHTED_SVD else "asvd_diagonal"
    request = transform.request
    if request.objective != expected_objective or request.statistics_refs:
        return _failure("InvalidObjective", "transform.request", "The fixed recipe requires its own objective and unstarted statistics")
    if recipe is Recipe.ASVD and (request.representation != "two_layers" or request.metric_policy != MetricPolicy()):
        return _failure("UnsupportedOptions", "transform.request", "The ASVD backend installs its checked two-layer factors")
    if any(type(step) is not StepSpec or type(step.operator) is not OperatorDescriptor for step in transform.steps):
        return _failure("InvalidTransform", "transform.steps", "Typed operator steps are required")
    checked_request = _check_operator_plan(replace(request, objective="input_second_moment"),
        tuple(step.operator for step in transform.steps), transform.resource_limits)
    if isinstance(checked_request, ValidationFailure):
        return checked_request
    paths = []
    for index, step in enumerate(transform.steps):
        if (type(step.rank) is not int or type(step.requires_statistics) is not bool
                or type(step.requires_operator_spectrum) is not bool):
            return _failure("InvalidTransform", f"transform.steps.{index}", "Resolved integer ranks and boolean stage requirements are required")
        # Re-run the existing pure rank/capability rules per resolved step. ASVD
        # permits per-path fixed ranks while weighted has one shared rank rule.
        if recipe is Recipe.ASVD and step.operator.kind != "Linear":
            return _failure("UnsupportedCapability", step.operator.path, "The G1 ASVD backend supports Linear only")
        checked = _check_operator_plan(replace(request, objective="input_second_moment", structure=step.structure),
                                 (step.operator,), transform.resource_limits)
        if isinstance(checked, ValidationFailure):
            return checked
        if checked.steps[0] != step:
            return _failure("InvalidRank", step.operator.path, "Resolved rank and required stages must match the operator plan")
        if recipe is Recipe.WEIGHTED_SVD and step.structure != request.structure:
            return _failure("InvalidRank", step.operator.path, "Weighted G1 uses one declared structure policy")
        if step.rank is None:
            return _failure("InvalidRank", step.operator.path, "The fixed recipe requires resolved ranks")
        paths.append(step.operator.path)
    if len(set(paths)) != len(paths):
        return _failure("InvalidScope", "transform.steps", "Each selected module must be planned once")
    for group in aliases:
        if len(set(group).intersection(paths)) > 1:
            return _failure("InvalidScope", "transform.steps", "A shared module must be planned once through a canonical path")
    writes = tuple(alias for path in paths for alias in next((group for group in aliases if path in group), (path,)))
    for index, path in enumerate(writes):
        if any(path == "" or other.startswith(path + ".") or other == "" or path.startswith(other + ".")
               for other in writes[index + 1:]):
            return _failure("TopologyConflict", "transform.steps", "Fixed recipes cannot replace nested operator scopes")
    return None


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _template(recipe, transform, source, calibration, options, aliases, method_spec):
    frozen = recipe is Recipe.WEIGHTED_SVD
    base = source.fingerprint + ":initial"
    # Future versions are symbolic recipe identities, not hashes of weights
    # that have not been computed. Distinct rank/metric/data branches must not
    # acquire the same version merely because both accepted one replacement.
    branch = hashlib.sha256(json.dumps(_canonical((recipe, transform, source, calibration,
        options, aliases, method_spec)), sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()
    model = ArtifactRef("model:initial", ArtifactKind.MODEL, base, coordinates="model_graph")
    data = ArtifactRef("data:calibration", ArtifactKind.CALIBRATION, base, coordinates="calibration_input")
    current = model
    scopes, nodes, replacements = [], [], []
    for index, step in enumerate(transform.steps):
        path = step.operator.path
        scope = next((group for group in aliases if path in group), (path,))
        scopes.append(OperatorScope(step.operator, scope))
        prefix = f"operator:{index}"
        version = model.model_version if frozen else current.model_version
        read = model if frozen else current
        stat_kind = (ArtifactKind.INPUT_SECOND_MOMENT if frozen else ArtifactKind.INPUT_ABS_MEAN
                     if method_spec.abs_mode == "abs_mean" else ArtifactKind.INPUT_ABS_MAX)
        metric_kind = ArtifactKind.FULL_METRIC if frozen else ArtifactKind.DIAGONAL_METRIC
        stat = ArtifactRef(prefix + ":statistics", stat_kind, version, scope)
        metric = ArtifactRef(prefix + ":metric", metric_kind, version, scope)
        answer = ArtifactRef(prefix + ":approximation", ArtifactKind.APPROXIMATION, version, scope)
        replacement = ArtifactRef(prefix + ":replacement", ArtifactKind.REPLACEMENT, version, scope)
        nodes.extend((
            CompressionNode(prefix + ":observe", Stage.OBSERVE, scope, scope, (),
                (Port("model", ArtifactKind.MODEL, read.key), Port("data", ArtifactKind.CALIBRATION, data.key)), (stat,)),
            CompressionNode(prefix + ":metric", Stage.METRIC, scope, (), (),
                (Port("statistics", stat_kind, stat.key),), (metric,)),
            CompressionNode(prefix + ":solve", Stage.SOLVE, scope, scope, (),
                (Port("operator", ArtifactKind.MODEL, read.key), Port("metric", metric_kind, metric.key)), (answer,)),
            CompressionNode(prefix + ":build", Stage.BUILD, scope, scope, (),
                (Port("operator", ArtifactKind.MODEL, read.key), Port("approximation", ArtifactKind.APPROXIMATION, answer.key)), (replacement,)),
        ))
        replacements.append(replacement)
        if not frozen:
            next_model = ArtifactRef(f"model:accepted={index + 1}", ArtifactKind.MODEL,
                branch + f":accepted={index + 1}", coordinates="model_graph")
            nodes.append(CompressionNode(prefix + ":apply", Stage.APPLY, scope, (), scope,
                (Port("model", ArtifactKind.MODEL, current.key), Port("replacement", ArtifactKind.REPLACEMENT, replacement.key)), (next_model,)))
            current = next_model
    if frozen:
        writes = tuple(path for scope in scopes for path in scope.aliases)
        next_model = ArtifactRef("model:accepted=1", ArtifactKind.MODEL,
                                branch + ":accepted=1", coordinates="model_graph")
        nodes.append(CompressionNode("model:apply_batch", Stage.APPLY, writes, (), writes,
            (Port("model", ArtifactKind.MODEL, model.key),) + tuple(
                Port(f"replacement:{index}", ArtifactKind.REPLACEMENT, artifact.key)
                for index, artifact in enumerate(replacements)), (next_model,)))
    return tuple(scopes), (model, data), tuple(nodes)


def compile_fixed_graph(recipe, transform, source, calibration, options=RuntimeOptions(),
                        *, method_spec=None, module_aliases=(), executor="checked_whole_recipe_v1"):
    """Pure compilation from checked metadata; never copy, observe or solve."""
    failure = _check_inputs(recipe, transform, source, calibration, options, method_spec, module_aliases)
    if failure is not None:
        return failure
    if type(executor) is not str or executor not in ("checked_whole_recipe_v1", "staged_fixed_recipe_v1"):
        return _failure("UnsupportedExecutor", "executor", "A declared fixed recipe interpreter is required")
    scopes, external, nodes = _template(recipe, transform, source, calibration, options, module_aliases, method_spec)
    graph = FixedCompressionGraph(recipe,
        SourceMode.FROZEN_SNAPSHOT if recipe is Recipe.WEIGHTED_SVD else SourceMode.CURRENT_STUDENT,
        source, calibration, transform, options, method_spec, module_aliases, scopes, external, nodes, "", executor=executor)
    return replace(graph, fingerprint=graph_fingerprint(graph))


def validate_fixed_graph(graph):
    """Reject malformed/tampered plans before any runtime effects.

    A matching digest alone is insufficient: all nodes, ports, versions, scopes
    and required stages must match the complete supported recipe template.
    """
    if type(graph) is not FixedCompressionGraph:
        return _failure("InvalidGraph", "graph", "FixedCompressionGraph required")
    if not _metadata_only(graph):
        return _failure("LiveResource", "graph", "Graph descriptions cannot own runtime resources")
    failure = _check_inputs(graph.recipe, graph.transform, graph.source, graph.calibration,
                            graph.options, graph.method_spec, graph.module_aliases)
    if failure is not None:
        return failure
    if type(graph.schema_version) is not int or graph.schema_version != 1 or graph.executor not in ("checked_whole_recipe_v1", "staged_fixed_recipe_v1"):
        return _failure("UnsupportedSchema", "graph", "Only schema-1 fixed recipe interpreters are executable")
    available = {}
    if type(graph.external_artifacts) is not tuple or type(graph.nodes) is not tuple:
        return _failure("InvalidGraph", "graph.nodes", "Immutable node and artifact tuples required")
    for artifact in graph.external_artifacts:
        if type(artifact) is not ArtifactRef or type(artifact.key) is not str or type(artifact.kind) is not ArtifactKind:
            return _failure("InvalidArtifact", "graph.external_artifacts", "Typed artifact references required")
        if artifact.key in available:
            return _failure("DuplicateArtifact", artifact.key, "Artifact producers must be unique")
        available[artifact.key] = artifact
    node_keys = set()
    for node in graph.nodes:
        if type(node) is not CompressionNode or type(node.key) is not str or type(node.stage) is not Stage:
            return _failure("InvalidNode", "graph.nodes", "Typed compression nodes required")
        if node.key in node_keys:
            return _failure("DuplicateNode", node.key, "Node identifiers must be unique")
        node_keys.add(node.key)
        if type(node.inputs) is not tuple or type(node.outputs) is not tuple:
            return _failure("InvalidNode", node.key, "Immutable input/output tuples required")
        for port in node.inputs:
            if (type(port) is not Port or type(port.name) is not str or not port.name
                    or type(port.artifact) is not str or type(port.kind) is not ArtifactKind):
                return _failure("InvalidPort", node.key, "Typed ports required")
            if port.artifact not in available:
                return _failure("MissingProducer", node.key + "." + port.name, "Input must refer to an earlier producer")
            if available[port.artifact].kind is not port.kind:
                return _failure("SemanticTypeMismatch", node.key + "." + port.name, "Statistic and metric meanings are not interchangeable")
        for artifact in node.outputs:
            if type(artifact) is not ArtifactRef or type(artifact.key) is not str or type(artifact.kind) is not ArtifactKind:
                return _failure("InvalidArtifact", node.key, "Typed artifact references required")
            if artifact.key in available:
                return _failure("DuplicateArtifact", artifact.key, "Artifact producers must be unique")
            available[artifact.key] = artifact
    expected = compile_fixed_graph(graph.recipe, graph.transform, graph.source, graph.calibration,
        graph.options, method_spec=graph.method_spec, module_aliases=graph.module_aliases, executor=graph.executor)
    if (graph.source_mode is not expected.source_mode or graph.scopes != expected.scopes
            or graph.external_artifacts != expected.external_artifacts or graph.nodes != expected.nodes):
        return _failure("UnsupportedRecipeGraph", "graph.nodes", "Required stages, scopes, versions and ports must match the fixed recipe")
    if graph.fingerprint != expected.fingerprint:
        return _failure("GraphFingerprintMismatch", "graph.fingerprint", "Graph metadata changed after compilation")
    return None
