"""A finite, metadata-only grammar compiled to complete compression workflows.

The genotype contains choices, never statistics, measurements or model objects.
All independent scope recipes observe one frozen source M0 and their complete
replacements are applied together. This is deliberately not an arbitrary DAG.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import random

from .compression_graph import (ArtifactKind, ArtifactRef, CalibrationRef,
    CompressionNode, FixedCompressionGraph, ModelVersion, OperatorScope, Port,
    Recipe, RuntimeOptions, Stage, _canonical, _digest, compile_fixed_graph)
from .method_specs import ASVD
from .plans import (FixedRank, Goal, MetricPolicy, OperatorDescriptor,
    ResourceLimits, TransformPlan, TransformRequest, ValidationFailure, Violation,
    _metadata_only, plan_transform)


@dataclass(frozen=True)
class WeightedChoice:
    rank: int
    representation: str = "two_layers"
    policy: MetricPolicy = MetricPolicy()
    centering: str = "none"
    bias_policy: str = "preserve"


@dataclass(frozen=True)
class ASVDChoice:
    rank: int
    spec: ASVD = ASVD()
    representation: str = "two_layers"
    centering: str = "none"
    bias_policy: str = "preserve"


@dataclass(frozen=True)
class AffinePCAChoice:
    rank: int
    representation: str = "two_layers"
    centering: str = "output_mean"
    bias_policy: str = "affine_mean_correction"


CompressionChoice = WeightedChoice | ASVDChoice | AffinePCAChoice


@dataclass(frozen=True)
class NoRecovery:
    """The bounded G3 space declares no recovery/training capability."""


@dataclass(frozen=True)
class ScopeGene:
    aliases: tuple[str, ...]
    choice: CompressionChoice
    recovery: NoRecovery = NoRecovery()


@dataclass(frozen=True)
class CompressionGenotype:
    genes: tuple[ScopeGene, ...]
    schema_version: int = 1


@dataclass(frozen=True)
class ScopeDomain:
    operator: OperatorDescriptor
    aliases: tuple[str, ...]
    alternatives: tuple[CompressionChoice, ...]


@dataclass(frozen=True)
class GrammarSpace:
    source: ModelVersion
    calibration: CalibrationRef
    domains: tuple[ScopeDomain, ...]
    module_aliases: tuple[tuple[str, ...], ...] = ()
    options: RuntimeOptions = RuntimeOptions()
    max_scopes: int = 32
    max_alternatives: int = 4096
    source_storage_fingerprint: str | None = None


@dataclass(frozen=True)
class AffineWholeProfilePlan:
    """Checked whole AFM profile; no claim of a staged StatisticsStore cache."""
    source: ModelVersion
    calibration: CalibrationRef
    scope: OperatorScope
    transform: TransformPlan
    centering: str = "output_mean"
    bias_policy: str = "affine_mean_correction"
    executor: str = "checked_afm_whole_profile_v1"


@dataclass(frozen=True)
class CompiledScope:
    gene: ScopeGene
    plan: FixedCompressionGraph | AffineWholeProfilePlan


@dataclass(frozen=True)
class CheckedWorkflow:
    space: GrammarSpace
    genotype: CompressionGenotype
    scopes: tuple[CompiledScope, ...]
    external_artifacts: tuple[ArtifactRef, ...]
    nodes: tuple[CompressionNode, ...]
    fingerprint: str
    source_semantics: str = "independent_frozen_M0_then_atomic_batch_apply"


class MutationKind(str, Enum):
    CHOICE = "choice"
    METRIC = "metric"
    RANK = "rank"
    REPRESENTATION = "representation"
    CENTERING = "centering"
    SCOPE = "scope"
    RECOVERY = "recovery"


def _failure(code, path, message):
    return ValidationFailure((Violation(code, path, message),))


def _hash(value):
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _choice_plan(choice, operator, options):
    if type(choice) not in (WeightedChoice, ASVDChoice, AffinePCAChoice):
        return _failure("InvalidChoice", operator.path, "A declared conditional recipe choice is required")
    if not _metadata_only(choice):
        return _failure("LiveResource", operator.path, "Recipe choices contain immutable metadata only")
    centered = type(choice) is AffinePCAChoice
    if ((choice.centering, choice.bias_policy) != (("output_mean", "affine_mean_correction")
            if centered else ("none", "preserve"))):
        return _failure("IncompatibleCenteringBias", operator.path, "Centering and the affine bias policy must agree with the recipe")
    if type(choice) is not WeightedChoice and (operator.kind != "Linear" or choice.representation != "two_layers"):
        return _failure("UnsupportedCapability", operator.path, "ASVD and AFM require Linear and two_layers")
    if type(choice) is ASVDChoice:
        if type(choice.spec) is not ASVD:
            return _failure("InvalidMethod", operator.path, "ASVD parameters belong only to the ASVD branch")
        try:
            ASVD(choice.spec.abs_mode, choice.spec.alpha, choice.spec.epsilon, choice.spec.zero_policy)
        except (TypeError, ValueError, OverflowError):
            return _failure("InvalidMethod", operator.path, "Invalid ASVD parameters")
    request = TransformRequest(Goal.REPLACE_OPERATOR, "input_second_moment", "svd",
        FixedRank(choice.rank), choice.representation, "channel",
        choice.policy if type(choice) is WeightedChoice else MetricPolicy())
    try:
        plan = plan_transform(request, (operator,), ResourceLimits(options.max_peak_bytes))
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return _failure("InvalidOperator", operator.path, "Invalid operator or finite recipe metadata")
    if isinstance(plan, ValidationFailure):
        return plan
    if type(choice) is not WeightedChoice:
        plan = replace(plan, request=replace(plan.request, objective=
            "asvd_diagonal" if type(choice) is ASVDChoice else "centered_affine_output_pca"))
    return plan


def validate_grammar_space(space):
    """Reject unsupported domains before generation; never prune alternatives."""
    if type(space) is not GrammarSpace or not _metadata_only(space):
        return _failure("InvalidGrammarSpace", "space", "A metadata-only GrammarSpace is required")
    if (type(space.source) is not ModelVersion or type(space.source.graph_version) is not str
            or not space.source.graph_version or not _digest(space.source.fingerprint)
            or type(space.calibration) is not CalibrationRef or not _digest(space.calibration.fingerprint)
            or space.calibration.role != "calibration" or type(space.calibration.shape) is not tuple
            or len(space.calibration.shape) < 2 or any(type(d) is not int or d <= 0 for d in space.calibration.shape)
            or space.calibration.dtype not in ("float32", "float64")):
        return _failure("InvalidSourceContext", "space", "Explicit source and calibration identities are required")
    if space.source_storage_fingerprint is not None and not _digest(space.source_storage_fingerprint):
        return _failure("InvalidSourceContext", "space.source_storage_fingerprint", "An optional full-storage SHA-256 identity is required")
    options = space.options
    if (type(options) is not RuntimeOptions or any(type(getattr(options, name)) is not int
            or getattr(options, name) <= 0 for name in ("batch_size", "max_workspace_bytes", "max_peak_bytes"))
            or options.max_peak_bytes > 12 * 1024**3 or options.max_workspace_bytes > options.max_peak_bytes
            or any(value is not None and (type(value) is not int or value <= 0)
                   for value in (options.parameter_budget, options.tensor_byte_budget))):
        return _failure("InvalidResourceLimit", "space.options", "Finite positive CPU limits and optional exact budgets are required")
    if (type(space.max_scopes) is not int or not 1 <= space.max_scopes <= 128
            or type(space.max_alternatives) is not int or not 1 <= space.max_alternatives <= 4096
            or type(space.domains) is not tuple or not 1 <= len(space.domains) <= space.max_scopes):
        return _failure("InvalidGrammarLimit", "space.domains", "A bounded nonempty finite scope domain is required")
    if (type(space.module_aliases) is not tuple or any(type(group) is not tuple or len(group) < 2
            or any(type(path) is not str for path in group) or len(set(group)) != len(group)
            for group in space.module_aliases)
            or len({path for group in space.module_aliases for path in group}) != sum(map(len, space.module_aliases))):
        return _failure("InvalidScope", "space.module_aliases", "Disjoint complete module alias groups are required")
    seen, count = [], 0
    for domain in space.domains:
        if (type(domain) is not ScopeDomain or type(domain.operator) is not OperatorDescriptor
                or type(domain.operator.path) is not str or type(domain.aliases) is not tuple
                or not domain.aliases or any(type(path) is not str for path in domain.aliases)
                or domain.aliases != tuple(sorted(set(domain.aliases))) or domain.operator.path != domain.aliases[0]
                or type(domain.alternatives) is not tuple or not domain.alternatives):
            return _failure("InvalidScope", "space.domains", "Canonical nonempty alias scopes and alternatives are required")
        closure = next((tuple(sorted(group)) for group in space.module_aliases if domain.operator.path in group), (domain.operator.path,))
        if domain.aliases != closure:
            return _failure("IncompleteAliasScope", domain.operator.path, "Each scope must include every module alias")
        for path in domain.aliases:
            if any(path == previous or path == "" or previous == "" or path.startswith(previous + ".")
                   or previous.startswith(path + ".") for previous in seen):
                return _failure("OverlappingScopes", path, "All declared scopes must be disjoint and nonnested")
            seen.append(path)
        count += len(domain.alternatives)
        if count > space.max_alternatives:
            return _failure("InvalidGrammarLimit", "space.domains", "The finite alternative cap is exceeded")
        for index, choice in enumerate(domain.alternatives):
            plan = _choice_plan(choice, domain.operator, options)
            if isinstance(plan, ValidationFailure):
                return plan
            if choice in domain.alternatives[:index]:
                return _failure("DuplicateAlternative", domain.operator.path, "Recipe alternatives must be distinct")
    return None


def normalize_genotype(genotype, space):
    """Only order independent scopes; do not clip ranks or repair semantics."""
    failure = validate_grammar_space(space)
    if failure is not None:
        return failure
    if (type(genotype) is not CompressionGenotype or not _metadata_only(genotype)
            or type(genotype.schema_version) is not int or genotype.schema_version != 1
            or type(genotype.genes) is not tuple or not genotype.genes):
        return _failure("InvalidGenotype", "genotype", "A nonempty immutable schema-1 genotype is required")
    domains = {domain.aliases: domain for domain in space.domains}
    seen = set()
    for gene in genotype.genes:
        if type(gene) is not ScopeGene or type(gene.aliases) is not tuple or any(type(p) is not str for p in gene.aliases):
            return _failure("InvalidScope", "genotype.genes", "A typed complete scope gene is required")
        if gene.aliases not in domains or gene.aliases in seen:
            return _failure("InvalidScope", "genotype.genes", "Each declared complete scope may be selected once")
        seen.add(gene.aliases)
        failure = _choice_plan(gene.choice, domains[gene.aliases].operator, space.options)
        if isinstance(failure, ValidationFailure):
            return failure
        if gene.choice not in domains[gene.aliases].alternatives:
            return _failure("ChoiceOutsideDomain", gene.aliases[0], "The exact recipe choice is outside its finite domain")
        if type(gene.recovery) is not NoRecovery:
            return _failure("UnsupportedRecovery", gene.aliases[0], "G3 declares NoRecovery only; recovery/training must not be inferred")
    return replace(genotype, genes=tuple(sorted(genotype.genes, key=lambda gene: gene.aliases)))


def compile_genotype(genotype, space):
    """Compile every mandatory stage and fresh artifact dependency from choices."""
    genotype = normalize_genotype(genotype, space)
    if isinstance(genotype, ValidationFailure):
        return genotype
    domains = {domain.aliases: domain for domain in space.domains}
    # Namespace every dependent reference by the full source/data/choice context.
    branch = _hash((space, genotype))
    version = space.source.fingerprint + ":frozen_M0:workflow=" + branch
    model = ArtifactRef("model:M0", ArtifactKind.MODEL, version, coordinates="model_graph")
    data = ArtifactRef("data:calibration", ArtifactKind.CALIBRATION, version, coordinates="calibration_input")
    scopes, nodes, replacements = [], [], []
    for index, gene in enumerate(genotype.genes):
        domain = domains[gene.aliases]
        choice, aliases = gene.choice, gene.aliases
        plan = _choice_plan(choice, domain.operator, space.options)
        if type(choice) is AffinePCAChoice:
            compiled = AffineWholeProfilePlan(space.source, space.calibration,
                OperatorScope(domain.operator, aliases), plan)
            kinds = (ArtifactKind.OUTPUT_MEAN, ArtifactKind.CENTERED_OUTPUT_COVARIANCE)
            metric_kind = ArtifactKind.OUTPUT_PROJECTOR
        else:
            fixed_options = replace(space.options, parameter_budget=None, tensor_byte_budget=None)
            compiled = compile_fixed_graph(Recipe.WEIGHTED_SVD if type(choice) is WeightedChoice else Recipe.ASVD,
                plan, space.source, space.calibration, fixed_options,
                method_spec=choice.spec if type(choice) is ASVDChoice else None,
                module_aliases=space.module_aliases, executor="staged_fixed_recipe_v1")
            if isinstance(compiled, ValidationFailure):
                return compiled
            kinds = (ArtifactKind.INPUT_SECOND_MOMENT,) if type(choice) is WeightedChoice else (
                ArtifactKind.INPUT_ABS_MEAN if choice.spec.abs_mode == "abs_mean" else ArtifactKind.INPUT_ABS_MAX,)
            metric_kind = ArtifactKind.FULL_METRIC if type(choice) is WeightedChoice else ArtifactKind.DIAGONAL_METRIC
        scopes.append(CompiledScope(gene, compiled))
        prefix = f"scope:{index}"
        statistics = tuple(ArtifactRef(prefix + f":stat:{number}", kind, version, aliases)
                           for number, kind in enumerate(kinds))
        metric = ArtifactRef(prefix + ":metric", metric_kind, version, aliases)
        answer = ArtifactRef(prefix + ":approximation", ArtifactKind.APPROXIMATION, version, aliases)
        replacement = ArtifactRef(prefix + ":replacement", ArtifactKind.REPLACEMENT, version, aliases)
        stat_ports = tuple(Port(f"stat:{number}", ref.kind, ref.key) for number, ref in enumerate(statistics))
        nodes.extend((
            CompressionNode(prefix + ":observe", Stage.OBSERVE, aliases, aliases, (),
                (Port("model", model.kind, model.key), Port("data", data.kind, data.key)), statistics),
            CompressionNode(prefix + ":metric", Stage.METRIC, aliases, (), (), stat_ports, (metric,)),
            CompressionNode(prefix + ":solve", Stage.SOLVE, aliases, aliases, (),
                (Port("model", model.kind, model.key), Port("metric", metric.kind, metric.key)), (answer,)),
            CompressionNode(prefix + ":build", Stage.BUILD, aliases, aliases, (),
                (Port("model", model.kind, model.key), Port("approximation", answer.kind, answer.key))
                + (stat_ports[:1] if type(choice) is AffinePCAChoice else ()), (replacement,)),
        ))
        replacements.append(replacement)
    writes = tuple(alias for gene in genotype.genes for alias in gene.aliases)
    output = ArtifactRef("model:compressed", ArtifactKind.MODEL, branch + ":compressed", coordinates="model_graph")
    nodes.append(CompressionNode("model:apply_batch", Stage.APPLY, writes, (), writes,
        (Port("model", model.kind, model.key),) + tuple(Port(f"replacement:{i}", ref.kind, ref.key)
            for i, ref in enumerate(replacements)), (output,)))
    result = CheckedWorkflow(space, genotype, tuple(scopes), (model, data), tuple(nodes), "")
    return replace(result, fingerprint=_hash(result))


def validate_workflow(workflow):
    if type(workflow) is not CheckedWorkflow or not _metadata_only(workflow):
        return _failure("InvalidWorkflow", "workflow", "A metadata-only CheckedWorkflow is required")
    expected = compile_genotype(workflow.genotype, workflow.space)
    if isinstance(expected, ValidationFailure):
        return expected
    if workflow != expected:
        return _failure("WorkflowBindingMismatch", "workflow", "Mandatory stages, scope dependencies, source and fingerprint must match recompilation")
    return None


def _seed(seed):
    return type(seed) is int and seed >= 0


def generate_genotype(space, *, seed=0):
    failure = validate_grammar_space(space)
    if failure is not None:
        return failure
    if not _seed(seed):
        return _failure("InvalidSeed", "seed", "A nonnegative integer seed is required")
    rng = random.Random(seed)
    selected = rng.sample(list(space.domains), rng.randint(1, len(space.domains)))
    return normalize_genotype(CompressionGenotype(tuple(ScopeGene(domain.aliases,
        rng.choice(domain.alternatives)) for domain in selected)), space)


def mutate_genotype(genotype, space, *, kind=MutationKind.CHOICE, seed=0):
    genotype = normalize_genotype(genotype, space)
    if isinstance(genotype, ValidationFailure):
        return genotype
    if type(kind) is not MutationKind or not _seed(seed):
        return _failure("InvalidMutation", "mutation", "A declared mutation and nonnegative integer seed are required")
    if kind is MutationKind.RECOVERY:
        return _failure("UnsupportedRecovery", "mutation", "Only NoRecovery belongs to this declared grammar space")
    rng = random.Random(seed)
    genes = list(genotype.genes)
    domains = {domain.aliases: domain for domain in space.domains}
    if kind is MutationKind.SCOPE:
        additions = [domain for domain in space.domains if domain.aliases not in {gene.aliases for gene in genes}]
        moves = [("add", domain) for domain in additions] + ([("drop", gene) for gene in genes] if len(genes) > 1 else [])
        if moves:
            operation, target = rng.choice(moves)
            if operation == "add":
                genes.append(ScopeGene(target.aliases, rng.choice(target.alternatives)))
            else:
                genes.remove(target)
    else:
        moves = []
        for index, gene in enumerate(genes):
            old = gene.choice
            for choice in domains[gene.aliases].alternatives:
                if choice == old:
                    continue
                if kind is MutationKind.RANK and (type(choice) is not type(old) or replace(choice, rank=old.rank) != old):
                    continue
                if kind is MutationKind.REPRESENTATION and (type(choice) is not type(old) or replace(choice, representation=old.representation) != old):
                    continue
                if kind is MutationKind.CENTERING and (choice.centering == old.centering or choice.rank != old.rank):
                    continue
                if kind is MutationKind.METRIC and (choice.rank != old.rank or choice.representation != old.representation or choice.centering != old.centering):
                    continue
                moves.append((index, choice))
        if moves:
            index, choice = rng.choice(moves)
            genes[index] = replace(genes[index], choice=choice)
    return normalize_genotype(replace(genotype, genes=tuple(genes)), space)


def crossover_genotypes(first, second, space, *, seed=0):
    """Exchange complete compatible scopes, binding children to destination M0.

    Parents contain no source/statistic references; every child compilation
    creates all dependencies anew in the supplied destination context.
    """
    left, right = normalize_genotype(first, space), normalize_genotype(second, space)
    if isinstance(left, ValidationFailure):
        return left
    if isinstance(right, ValidationFailure):
        return right
    if not _seed(seed):
        return _failure("InvalidSeed", "seed", "A nonnegative integer seed is required")
    rng = random.Random(seed)
    a, b = {gene.aliases: gene for gene in left.genes}, {gene.aliases: gene for gene in right.genes}
    common = sorted(a.keys() & b.keys())
    if common:
        scope = rng.choice(common)
        a[scope], b[scope] = b[scope], a[scope]
    else:
        # Swap full disjoint scope sets, retaining at least one scope per child.
        a, b = b, a
    return (normalize_genotype(CompressionGenotype(tuple(a.values())), space),
            normalize_genotype(CompressionGenotype(tuple(b.values())), space))


def genotype_payload(genotype):
    """Export the tagged JSON shape consumed by genotype_from_payload."""
    if (type(genotype) is not CompressionGenotype or not _metadata_only(genotype)
            or type(genotype.schema_version) is not int or genotype.schema_version != 1
            or type(genotype.genes) is not tuple or not genotype.genes
            or any(type(gene) is not ScopeGene or type(gene.choice) not in
                (WeightedChoice, ASVDChoice, AffinePCAChoice) or type(gene.recovery) is not NoRecovery
                or type(gene.aliases) is not tuple or any(type(path) is not str for path in gene.aliases)
                for gene in genotype.genes)):
        return _failure("InvalidGenotype", "genotype", "A metadata-only genotype is required")
    payload = {"schema_version": genotype.schema_version, "genes": [
        {"aliases": list(gene.aliases), "recipe": {WeightedChoice: "weighted_svd", ASVDChoice: "asvd",
            AffinePCAChoice: "afm"}[type(gene.choice)], "choice": _canonical(gene.choice), "recovery": "none"}
        for gene in genotype.genes]}
    try:
        json.dumps(payload, allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        return _failure("InvalidGenotype", "genotype", "Genotype export requires finite JSON metadata")
    return payload


def genotype_from_payload(payload, space):
    """Decode the supported tagged transport shape, then validate its domain.

    Unknown keys and conditional parameters are rejected rather than silently
    discarded. Export and import preserve exact recipe parameters and aliases.
    """
    try:
        if (type(payload) is not dict or set(payload) != {"schema_version", "genes"}
                or type(payload["genes"]) is not list or not payload["genes"]
                or len(payload["genes"]) > 128):
            return _failure("InvalidGenotypePayload", "payload", "A bounded schema and gene list are required")
        constructors = {"weighted_svd": WeightedChoice, "asvd": ASVDChoice, "afm": AffinePCAChoice}
        genes = []
        for item in payload["genes"]:
            if (type(item) is not dict or set(item) != {"aliases", "recipe", "choice", "recovery"}
                    or type(item["aliases"]) is not list or not item["aliases"]
                    or any(type(alias) is not str for alias in item["aliases"])
                    or type(item["recipe"]) is not str or item["recipe"] not in constructors
                    or type(item["choice"]) is not dict or item["recovery"] != "none"):
                return _failure("InvalidGenotypePayload", "payload.genes", "A complete tagged conditional recipe and NoRecovery are required")
            values = dict(item["choice"])
            common = {"rank", "representation", "centering", "bias_policy"}
            kind = item["recipe"]
            expected = common | ({"policy"} if kind == "weighted_svd" else {"spec"} if kind == "asvd" else set())
            if set(values) != expected:
                return _failure("InvalidConditionalParameters", "payload.choice", "Only the selected recipe's complete parameters are accepted")
            if kind == "weighted_svd":
                if type(values["policy"]) is not dict or set(values["policy"]) != {"ridge", "rcond", "nullspace_policy"}:
                    return _failure("InvalidConditionalParameters", "payload.policy", "A complete weighted metric policy is required")
                values["policy"] = MetricPolicy(**values["policy"])
            elif kind == "asvd":
                if type(values["spec"]) is not dict or set(values["spec"]) != {"abs_mode", "alpha", "epsilon", "zero_policy"}:
                    return _failure("InvalidConditionalParameters", "payload.spec", "A complete ASVD parameter description is required")
                values["spec"] = ASVD(**values["spec"])
            genes.append(ScopeGene(tuple(item["aliases"]), constructors[kind](**values)))
        return normalize_genotype(CompressionGenotype(tuple(genes), payload["schema_version"]), space)
    except (KeyError, TypeError, ValueError, OverflowError):
        return _failure("InvalidGenotypePayload", "payload", "Invalid finite conditional recipe metadata")
