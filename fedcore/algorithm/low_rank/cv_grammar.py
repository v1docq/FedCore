"""Pure finite CV adapter of the common compression workflow/search graph.

G3 and G5 keep their existing capability boundaries. This profile describes
coupled internal BasicBlock coordinates and a fixed prune-then-SVD order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import random

from .compression_graph import (ArtifactKind, ArtifactRef, CalibrationRef, CompressionNode,
    ModelVersion, Port, RuntimeOptions, Stage, _digest)
from .compression_grammar import NoRecovery, MutationKind, _hash
from .plans import ValidationFailure, Violation, _metadata_only
from fedcore.algorithm.pruning.sparse.cv_channels import BLOCK_PATHS, BLOCK_INPUTS, BLOCK_OUTPUTS, BLOCK_STRIDES


@dataclass(frozen=True)
class DenseCV:
    pass


@dataclass(frozen=True)
class ChannelPrune:
    keep_channels: int


@dataclass(frozen=True)
class ConvSVD:
    rank: int


@dataclass(frozen=True)
class ChannelPruneSVD:
    keep_channels: int
    rank: int


CVChoice = DenseCV | ChannelPrune | ConvSVD | ChannelPruneSVD


@dataclass(frozen=True)
class CVScopeGene:
    path: str
    choice: CVChoice
    recovery: NoRecovery = NoRecovery()


@dataclass(frozen=True)
class CVGenotype:
    genes: tuple[CVScopeGene, ...]
    schema_version: int = 1


@dataclass(frozen=True)
class CVBlockDomain:
    path: str
    alternatives: tuple[CVChoice, ...]


@dataclass(frozen=True)
class CVGrammarSpace:
    source: ModelVersion
    calibration: CalibrationRef
    domains: tuple[CVBlockDomain, ...]
    source_storage_fingerprint: str
    source_storage_bytes: int
    num_classes: int = 10
    dtype: str = "float32"
    options: RuntimeOptions = RuntimeOptions()
    profile: str = "cifar_resnet18_internal_channel_l1_svd_v1"


@dataclass(frozen=True)
class CVBlockPlan:
    gene: CVScopeGene
    internal_width: int
    rank: int | None
    original_tensor_bytes: int
    final_tensor_bytes: int
    solver_workspace_bytes: int


@dataclass(frozen=True)
class CheckedCVWorkflow:
    space: CVGrammarSpace
    genotype: CVGenotype
    blocks: tuple[CVBlockPlan, ...]
    external_artifacts: tuple[ArtifactRef, ...]
    nodes: tuple[CompressionNode, ...]
    expected_storage_bytes: int
    fingerprint: str
    source_semantics: str = "independent_frozen_M0_coupled_blocks_then_atomic_apply"


def _failure(code, path, message):
    return ValidationFailure((Violation(code, path, message),))


def _choice(choice, path, dtype):
    if type(choice) not in (DenseCV, ChannelPrune, ConvSVD, ChannelPruneSVD) or not _metadata_only(choice):
        return _failure("InvalidChoice", path, "Declare one exact finite CV recipe")
    index = BLOCK_PATHS.index(path)
    incoming, outgoing = BLOCK_INPUTS[index], BLOCK_OUTPUTS[index]
    width = choice.keep_channels if type(choice) in (ChannelPrune, ChannelPruneSVD) else outgoing
    rank = choice.rank if type(choice) in (ConvSVD, ChannelPruneSVD) else None
    if type(width) is not int or not 1 <= width <= outgoing:
        return _failure("InvalidChannelCount", path, "Internal channel count must fit the frozen block")
    if type(choice) in (ChannelPrune, ChannelPruneSVD) and width >= outgoing:
        return _failure("InvalidChannelCount", path, "Structural pruning must remove at least one internal channel")
    if type(choice) in (ConvSVD, ChannelPruneSVD) and (type(rank) is not int or not 1 <= rank <= min(outgoing, 9 * width)):
        return _failure("InvalidRank", path, "SVD rank must fit the reduced conv2 matrix")
    element = 4 if dtype == "float32" else 8
    # Conv1, four BN1 vectors, BN1's int64 counter, and conv2. Everything else
    # remains in source_storage_bytes and is never charged twice.
    original = element * (9 * incoming * outgoing + 4 * outgoing + 9 * outgoing * outgoing) + 8
    final = element * (9 * incoming * width + 4 * width +
                       (9 * width * outgoing if rank is None else rank * (9 * width + outgoing))) + 8
    if rank is not None and rank * (9 * width + outgoing) >= 9 * width * outgoing:
        return _failure("NoncompressingRank", path, "The declared factors must use fewer physical bytes than reduced conv2")
    workspace = 0 if rank is None else 64 * (outgoing * 9 * width + min(outgoing, 9 * width) ** 2)
    return width, rank, original, final, workspace


def validate_cv_grammar_space(space):
    if type(space) is not CVGrammarSpace or not _metadata_only(space):
        return _failure("InvalidGrammarSpace", "space", "A metadata-only CVGrammarSpace is required")
    if (type(space.source) is not ModelVersion or type(space.source.graph_version) is not str
            or not space.source.graph_version or not _digest(space.source.fingerprint)
            or not _digest(space.source_storage_fingerprint) or type(space.source_storage_bytes) is not int
            or not 0 < space.source_storage_bytes <= 64 * 1024**2
            or type(space.calibration) is not CalibrationRef or not _digest(space.calibration.fingerprint)
            or type(space.calibration.shape) is not tuple or len(space.calibration.shape) != 4
            or space.calibration.shape[1:] != (3, 32, 32) or type(space.calibration.shape[0]) is not int
            or not 1 <= space.calibration.shape[0] <= 4096 or space.calibration.role != "calibration"
            or space.dtype != "float32" or space.calibration.dtype != space.dtype
            or type(space.num_classes) is not int or not 2 <= space.num_classes <= 1000
            or space.profile != "cifar_resnet18_internal_channel_l1_svd_v1"):
        return _failure("InvalidSourceContext", "space", "Explicit bounded CIFAR model and calibration identities are required")
    options = space.options
    if (type(options) is not RuntimeOptions or any(type(getattr(options, field)) is not int or getattr(options, field) <= 0
            for field in ("batch_size", "max_workspace_bytes", "max_peak_bytes"))
            or options.max_workspace_bytes > options.max_peak_bytes or options.max_peak_bytes > 12 * 1024**3
            or options.parameter_budget is not None
            or options.tensor_byte_budget is not None and (type(options.tensor_byte_budget) is not int or options.tensor_byte_budget <= 0)):
        return _failure("InvalidResourceLimit", "space.options", "Bounded byte limits are required; the CV profile accepts tensor-byte budgets only")
    if (type(space.domains) is not tuple or not 1 <= len(space.domains) <= 8
            or any(type(domain) is not CVBlockDomain or domain.path not in BLOCK_PATHS for domain in space.domains)
            or tuple(domain.path for domain in space.domains) != tuple(sorted(set(domain.path for domain in space.domains)))):
        return _failure("InvalidScope", "space.domains", "Declare canonical distinct complete BasicBlock domains")
    for domain in space.domains:
        if type(domain.alternatives) is not tuple or not 1 <= len(domain.alternatives) <= 64:
            return _failure("InvalidGrammarLimit", domain.path, "Declare 1..64 exact alternatives per block")
        for index, choice in enumerate(domain.alternatives):
            plan = _choice(choice, domain.path, space.dtype)
            if type(plan) is ValidationFailure:
                return plan
            if choice in domain.alternatives[:index]:
                return _failure("DuplicateAlternative", domain.path, "CV alternatives must be distinct")
            if plan[-1] > options.max_workspace_bytes:
                return _failure("ResourceLimitExceeded", domain.path, "Declared double SVD buffers exceed max_workspace_bytes")
            if 6 * space.source_storage_bytes + plan[-1] > options.max_peak_bytes:
                return _failure("ResourceLimitExceeded", domain.path, "Source copies and declared solve buffers exceed max_peak_bytes")
    return None


def normalize_cv_genotype(genotype, space):
    failure = validate_cv_grammar_space(space)
    if failure is not None:
        return failure
    if (type(genotype) is not CVGenotype or not _metadata_only(genotype) or type(genotype.schema_version) is not int
            or genotype.schema_version != 1 or type(genotype.genes) is not tuple
            or any(type(gene) is not CVScopeGene or type(gene.path) is not str for gene in genotype.genes)):
        return _failure("InvalidGenotype", "genotype", "An immutable version-1 CV genotype is required")
    domains = {domain.path: domain for domain in space.domains}
    if len(genotype.genes) != len(domains) or set(gene.path for gene in genotype.genes) != set(domains):
        return _failure("InvalidScope", "genotype", "Select exactly one recipe for every declared complete block")
    for gene in genotype.genes:
        if type(gene.recovery) is not NoRecovery:
            return _failure("UnsupportedRecovery", gene.path, "CV grammar declares NoRecovery; caller-owned recovery is a separate evaluation stage")
        plan = _choice(gene.choice, gene.path, space.dtype)
        if type(plan) is ValidationFailure:
            return plan
        if gene.choice not in domains[gene.path].alternatives or type(gene.choice) not in (DenseCV, ChannelPrune, ConvSVD, ChannelPruneSVD):
            return _failure("ChoiceOutsideDomain", gene.path, "The exact CV choice must belong to its declared domain")
    return replace(genotype, genes=tuple(sorted(genotype.genes, key=lambda gene: gene.path)))


def compile_cv_genotype(genotype, space):
    genotype = normalize_cv_genotype(genotype, space)
    if type(genotype) is ValidationFailure:
        return genotype
    branch = _hash((space, genotype))
    version = space.source.fingerprint + ":frozen_M0:cv=" + branch
    model = ArtifactRef("model:M0", ArtifactKind.MODEL, version, coordinates="cifar_resnet18_original_channels")
    data = ArtifactRef("data:calibration", ArtifactKind.CALIBRATION, version, coordinates="NCHW_CIFAR32")
    blocks, nodes, replacements = [], [], []
    expected = space.source_storage_bytes
    for index, gene in enumerate(genotype.genes):
        width, rank, original, final, workspace = _choice(gene.choice, gene.path, space.dtype)
        blocks.append(CVBlockPlan(gene, width, rank, original, final, workspace))
        expected += final - original
        if type(gene.choice) is DenseCV:
            continue
        scope, prefix = (gene.path,), f"cv:{index}"
        selected = ArtifactRef(prefix + ":channels", ArtifactKind.CHANNEL_SELECTION, version, scope,
                               "original_conv1_output_bn1_conv2_input")
        reduced = ArtifactRef(prefix + ":compact", ArtifactKind.REPLACEMENT, branch + ":compact", scope,
                              "reduced_internal_channels_fixed_residual_output")
        nodes.append(CompressionNode(prefix + ":select", Stage.SOLVE, scope, scope, (),
            (Port("model", model.kind, model.key),), (selected,)))
        nodes.append(CompressionNode(prefix + ":compact", Stage.BUILD, scope, scope, (),
            (Port("model", model.kind, model.key), Port("channels", selected.kind, selected.key)), (reduced,)))
        replacement = reduced
        if rank is not None:
            approximation = ArtifactRef(prefix + ":svd", ArtifactKind.APPROXIMATION, branch + ":compact", scope,
                                        "conv2_out_by_reduced_input_kernel")
            replacement = ArtifactRef(prefix + ":factors", ArtifactKind.REPLACEMENT, branch + ":factors", scope,
                                      "reduced_internal_channels_fixed_residual_output")
            nodes.append(CompressionNode(prefix + ":solve", Stage.SOLVE, scope, scope, (),
                (Port("compact", reduced.kind, reduced.key),), (approximation,)))
            nodes.append(CompressionNode(prefix + ":factors", Stage.BUILD, scope, scope, (),
                (Port("compact", reduced.kind, reduced.key), Port("svd", approximation.kind, approximation.key)), (replacement,)))
        replacements.append(replacement)
    writes = tuple(plan.gene.path for plan in blocks if type(plan.gene.choice) is not DenseCV)
    output = ArtifactRef("model:compressed", ArtifactKind.MODEL, branch + ":compressed", coordinates="cifar_resnet18_graph")
    nodes.append(CompressionNode("cv:apply_batch", Stage.APPLY, writes, (), writes,
        (Port("model", model.kind, model.key),) + tuple(Port(str(index), ref.kind, ref.key) for index, ref in enumerate(replacements)), (output,)))
    if space.options.tensor_byte_budget is not None and expected > space.options.tensor_byte_budget:
        return _failure("BudgetExceeded", "genotype", "Complete final physical tensor bytes exceed the declared budget")
    result = CheckedCVWorkflow(space, genotype, tuple(blocks), (model, data), tuple(nodes), expected, "")
    return replace(result, fingerprint=_hash(result))


def validate_cv_workflow(workflow):
    if type(workflow) is not CheckedCVWorkflow or not _metadata_only(workflow):
        return _failure("InvalidWorkflow", "workflow", "A metadata-only checked CV workflow is required")
    expected = compile_cv_genotype(workflow.genotype, workflow.space)
    if type(expected) is ValidationFailure:
        return expected
    return None if expected == workflow else _failure("WorkflowBindingMismatch", "workflow", "CV nodes and source bindings must match exact recompilation")


def generate_cv_genotype(space, *, seed=0):
    failure = validate_cv_grammar_space(space)
    if failure is not None:
        return failure
    if type(seed) is not int or seed < 0:
        return _failure("InvalidSeed", "seed", "Declare a nonnegative integer seed")
    rng = random.Random(seed)
    return CVGenotype(tuple(CVScopeGene(domain.path, rng.choice(domain.alternatives)) for domain in space.domains))


def mutate_cv_genotype(genotype, space, *, kind=MutationKind.CHOICE, seed=0):
    genotype = normalize_cv_genotype(genotype, space)
    if type(genotype) is ValidationFailure:
        return genotype
    if kind not in (MutationKind.CHOICE, MutationKind.SCOPE) or type(seed) is not int or seed < 0:
        return _failure("InvalidMutation", "mutation", "CV profile permits complete recipe mutation only")
    rng, genes = random.Random(seed), list(genotype.genes)
    domains = {domain.path: domain for domain in space.domains}
    moves = [(index, choice) for index, gene in enumerate(genes) for choice in domains[gene.path].alternatives if choice != gene.choice]
    if moves:
        index, choice = rng.choice(moves)
        genes[index] = replace(genes[index], choice=choice)
    return normalize_cv_genotype(CVGenotype(tuple(genes)), space)


def crossover_cv_genotypes(first, second, space, *, seed=0):
    left, right = normalize_cv_genotype(first, space), normalize_cv_genotype(second, space)
    if type(left) is ValidationFailure or type(right) is ValidationFailure:
        return left if type(left) is ValidationFailure else right
    if type(seed) is not int or seed < 0:
        return _failure("InvalidSeed", "seed", "Declare a nonnegative integer seed")
    index = random.Random(seed).randrange(len(left.genes))
    a, b = list(left.genes), list(right.genes)
    a[index], b[index] = b[index], a[index]
    return CVGenotype(tuple(a)), CVGenotype(tuple(b))


def cv_genotype_payload(genotype):
    if (type(genotype) is not CVGenotype or not _metadata_only(genotype)
            or type(genotype.schema_version) is not int or genotype.schema_version != 1
            or type(genotype.genes) is not tuple or not genotype.genes
            or any(type(gene) is not CVScopeGene or type(gene.choice) not in
                   (DenseCV, ChannelPrune, ConvSVD, ChannelPruneSVD) or type(gene.recovery) is not NoRecovery for gene in genotype.genes)):
        return _failure("InvalidGenotype", "genotype", "A typed metadata-only CV genotype is required")
    return {"schema_version": 1, "profile": "cv_internal_channels_v1", "genes": [
        {"aliases": [gene.path], "recipe": type(gene.choice).__name__, "choice": asdict(gene.choice), "recovery": "none"}
        for gene in sorted(genotype.genes, key=lambda gene: gene.path)]}


def cv_genotype_from_payload(payload, space):
    try:
        if (type(payload) is not dict or set(payload) != {"schema_version", "profile", "genes"}
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or payload["profile"] != "cv_internal_channels_v1" or type(payload["genes"]) is not list
                or not 1 <= len(payload["genes"]) <= 8):
            raise ValueError("Invalid CV payload")
        constructors = {kind.__name__: kind for kind in (DenseCV, ChannelPrune, ConvSVD, ChannelPruneSVD)}
        genes = []
        for item in payload["genes"]:
            if (type(item) is not dict or set(item) != {"aliases", "recipe", "choice", "recovery"}
                    or type(item["aliases"]) is not list or len(item["aliases"]) != 1
                    or type(item["aliases"][0]) is not str or item["recovery"] != "none"
                    or type(item["recipe"]) is not str or item["recipe"] not in constructors or type(item["choice"]) is not dict):
                raise ValueError("Invalid CV gene")
            genes.append(CVScopeGene(item["aliases"][0], constructors[item["recipe"]](**item["choice"])))
        return normalize_cv_genotype(CVGenotype(tuple(genes)), space)
    except (ValueError, TypeError, KeyError):
        return _failure("InvalidGenotypePayload", "payload", "Exact tagged finite CV choices are required")
