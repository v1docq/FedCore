"""Closed G1 CV recipes over the existing coupled-channel/SVD interpreter.

PTQ is a deferred post-recovery stage. This module never trains, quantizes a
stale M0, or turns procedure branches into parallel model execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import random
from fedcore.algorithm.low_rank.compression_graph import RuntimeOptions
from fedcore.algorithm.low_rank.cv_grammar import (
    DenseCV, ChannelPrune, ConvSVD, ChannelPruneSVD, CVScopeGene, CVGenotype,
    compile_cv_genotype, validate_cv_grammar_space)
from fedcore.algorithm.low_rank.cv_execution import prepare_cv_grammar_space, execute_cv_workflow
from fedcore.algorithm.low_rank.plans import ValidationFailure, Violation, _metadata_only
from fedcore.algorithm.pruning.sparse.cv_channels import BLOCK_INPUTS, BLOCK_OUTPUTS, BLOCK_PATHS
from .contracts import (Dependency, JoinApplyReplacements, Operation, ParallelDisjoint,
                        PTQPlacement, Representation, Scope, Sequence, Step, TensorRepresentation, fingerprint, payload)


def _failure(code, message):
    return ValidationFailure((Violation(code, "heterogeneous", message),))


@dataclass(frozen=True)
class BlockRecipe:
    keep_channels: int | None = None
    rank: int | None = None
    ptq: bool = False
    ptq_placement: PTQPlacement | None = field(default=None, metadata={"omit_none": True})

    def __post_init__(self):
        if (type(self.ptq) is not bool or any(v is not None and (type(v) is not int or v < 1)
                                              for v in (self.keep_channels, self.rank))):
            raise ValueError("Explicit positive channel/rank choices and Boolean PTQ required")
        if self.ptq_placement is not None and (not self.ptq or type(self.ptq_placement) is not PTQPlacement):
            raise ValueError("Explicit placement requires an enabled PTQ step and a typed policy")

    @property
    def structural(self):
        if self.keep_channels is not None and self.rank is not None:
            return ChannelPruneSVD(self.keep_channels, self.rank)
        if self.keep_channels is not None:
            return ChannelPrune(self.keep_channels)
        return ConvSVD(self.rank) if self.rank is not None else DenseCV()


@dataclass(frozen=True)
class HeterogeneousDomain:
    path: str
    alternatives: tuple[BlockRecipe, ...]


@dataclass(frozen=True)
class HeterogeneousSpace:
    structural_space: object
    domains: tuple[HeterogeneousDomain, ...]
    schema: str = "heterogeneous_cifar_g1_v1"


@dataclass(frozen=True)
class HeterogeneousGene:
    path: str
    recipe: BlockRecipe


@dataclass(frozen=True)
class HeterogeneousGenotype:
    genes: tuple[HeterogeneousGene, ...]
    schema_version: int = 1


@dataclass(frozen=True)
class HeterogeneousWorkflow:
    space: HeterogeneousSpace
    genotype: HeterogeneousGenotype
    structural_workflow: object
    procedure: JoinApplyReplacements
    ptq_regions: tuple[str, ...]
    deferred_ptq: tuple[Step, ...]
    fingerprint: str


def default_alternatives():
    """Finite G1 catalog; ranks are derived after the chosen channel reduction.

    Each block has at most 18 alternatives (three widths, three rank states,
    with/without PTQ). No unprofitable rank or illegal independent integer is
    generated. These are candidates, not promises of deployment feasibility.
    """
    result = {}
    for path, width in zip(BLOCK_PATHS, BLOCK_OUTPUTS):
        choices = []
        for keep in (None, 3*width//4, width//2):
            internal = keep or width
            largest = min(width,9*internal,(9*internal*width-1)//(9*internal+width))
            ranks = tuple(dict.fromkeys((None,max(1,largest//4),max(1,largest//2))))
            for rank in ranks:
                choices.extend(BlockRecipe(keep,rank,ptq) for ptq in (False,True))
        result[path] = tuple(choices)
    return result


def prepare_space(model, calibration, *, alternatives, graph_version="heterogeneous-cifar-g1-v1",
                  options=RuntimeOptions()):
    if (type(alternatives) is not dict or not alternatives or any(type(path) is not str
            or type(choices) is not tuple or not choices or any(type(c) is not BlockRecipe for c in choices)
            for path, choices in alternatives.items())):
        return _failure("InvalidAlternatives", "Declare finite exact BlockRecipe alternatives")
    structural = {path: tuple(dict.fromkeys(c.structural for c in choices)) for path, choices in alternatives.items()}
    base = prepare_cv_grammar_space(model, calibration, alternatives=structural, graph_version=graph_version, options=options)
    if type(base) is ValidationFailure:
        return base
    space = HeterogeneousSpace(base, tuple(HeterogeneousDomain(path, tuple(sorted(set(choices), key=fingerprint)))
                                         for path, choices in sorted(alternatives.items())))
    failure = validate_space(space)
    return failure if failure is not None else space


def validate_space(space):
    if type(space) is not HeterogeneousSpace or not _metadata_only(space) or space.schema != "heterogeneous_cifar_g1_v1":
        return _failure("InvalidSpace", "Exact G1 space required")
    failure = validate_cv_grammar_space(space.structural_space)
    if failure is not None:
        return failure
    if (type(space.domains) is not tuple or any(type(d) is not HeterogeneousDomain for d in space.domains)
            or tuple(d.path for d in space.domains) !=
            tuple(d.path for d in space.structural_space.domains)):
        return _failure("ScopeMismatch", "G1 and structural domains must agree")
    for domain, base in zip(space.domains, space.structural_space.domains):
        if (type(domain.alternatives) is not tuple or not 1 <= len(domain.alternatives) <= 64
                or any(type(c) is not BlockRecipe for c in domain.alternatives)
                or len(set(domain.alternatives)) != len(domain.alternatives)
                or any(type(c) is not BlockRecipe or c.structural not in base.alternatives for c in domain.alternatives)):
            return _failure("InvalidDomain", "Closed distinct G1 alternatives required")
        allowed_targets = {domain.path + ".conv1", domain.path + ".conv2"}
        for recipe in domain.alternatives:
            if recipe.ptq_placement is not None and not allowed_targets.issuperset(
                    recipe.ptq_placement.omit_output_qdq):
                return _failure("InvalidPTQPlacement", "Every target must belong to its PTQ block")
    return None


def normalize_genotype(genotype, space):
    failure = validate_space(space)
    if failure is not None:
        return failure
    if (type(genotype) is not HeterogeneousGenotype or not _metadata_only(genotype)
            or type(genotype.schema_version) is not int or genotype.schema_version != 1
            or type(genotype.genes) is not tuple or any(type(g) is not HeterogeneousGene for g in genotype.genes)):
        return _failure("InvalidGenotype", "Immutable G1 genotype required")
    domains = {d.path: d.alternatives for d in space.domains}
    if len(genotype.genes) != len(domains) or {g.path for g in genotype.genes} != set(domains):
        return _failure("ScopeMismatch", "Select exactly one recipe for each declared block")
    if any(g.recipe not in domains[g.path] for g in genotype.genes):
        return _failure("ChoiceOutsideDomain", "Recipe is outside its checked domain")
    return replace(genotype, genes=tuple(sorted(genotype.genes, key=lambda g: g.path)))


def genotype_payload(genotype):
    return {"schema_version": genotype.schema_version, "profile": "heterogeneous_cifar_g1_v1", "genes": [
        {"aliases": [g.path], "recipe": payload(g.recipe)} for g in sorted(genotype.genes, key=lambda g: g.path)]}


def genotype_from_payload(value, space):
    try:
        if (type(value) is not dict or set(value) != {"schema_version", "profile", "genes"}
                or type(value["schema_version"]) is not int or value["schema_version"] != 1
                or value["profile"] != "heterogeneous_cifar_g1_v1" or type(value["genes"]) is not list
                or not 1 <= len(value["genes"]) <= 8):
            raise ValueError("Malformed genotype header")
        genes = []
        for item in value["genes"]:
            if (type(item) is not dict or set(item) != {"aliases", "recipe"}
                    or type(item["aliases"]) is not list or len(item["aliases"]) != 1
                    or type(item["aliases"][0]) is not str or type(item["recipe"]) is not dict):
                raise ValueError("Malformed complete scope")
            recipe = item["recipe"].copy()
            placement = None
            if "ptq_placement" in recipe:
                placement = PTQPlacement.from_payload(recipe.pop("ptq_placement"))
            if set(recipe) != {"keep_channels", "rank", "ptq"}:
                raise ValueError("Malformed complete recipe")
            genes.append(HeterogeneousGene(item["aliases"][0], BlockRecipe(**recipe, ptq_placement=placement)))
        return normalize_genotype(HeterogeneousGenotype(tuple(genes)), space)
    except (ValueError, TypeError, KeyError):
        return _failure("InvalidGenotypePayload", "Exact tagged G1 payload required")


def generate_genotype(space, *, seed=0):
    failure = validate_space(space)
    if failure is not None:
        return failure
    if type(seed) is not int or seed < 0:
        return _failure("InvalidSeed", "Nonnegative integer seed required")
    rng = random.Random(seed)
    return HeterogeneousGenotype(tuple(HeterogeneousGene(d.path, rng.choice(d.alternatives)) for d in space.domains))


def crossover_genotypes(first, second, space, *, seed=0):
    left, right = normalize_genotype(first, space), normalize_genotype(second, space)
    if type(left) is ValidationFailure or type(right) is ValidationFailure:
        return left if type(left) is ValidationFailure else right
    if type(seed) is not int or seed < 0:
        return _failure("InvalidSeed", "Nonnegative integer seed required")
    index = random.Random(seed).randrange(len(left.genes))
    a, b = list(left.genes), list(right.genes)
    a[index], b[index] = b[index], a[index]
    return HeterogeneousGenotype(tuple(a)), HeterogeneousGenotype(tuple(b))


def compile_genotype(genotype, space):
    normalized = normalize_genotype(genotype, space)
    if type(normalized) is ValidationFailure:
        return normalized
    base = compile_cv_genotype(CVGenotype(tuple(CVScopeGene(g.path, g.recipe.structural) for g in normalized.genes)),
                               space.structural_space)
    if type(base) is ValidationFailure:
        return base
    identity = fingerprint((space, normalized))
    branches, regions, deferred = [], [], []
    source = space.structural_space.source.fingerprint
    calibration = space.structural_space.calibration.fingerprint
    for gene in normalized.genes:
        index = BLOCK_PATHS.index(gene.path)
        outgoing = BLOCK_OUTPUTS[index]
        previous = TensorRepresentation(Representation.DENSE, (outgoing, 9 * outgoing), outgoing)
        steps = []
        dependency = Dependency(source, calibration, "original_internal_channels")
        if gene.recipe.keep_channels is not None:
            width = gene.recipe.keep_channels
            target = TensorRepresentation(Representation.COMPACT, (outgoing, 9 * width), width,
                                          channel_coordinates="frozen_M0_conv1_l1_selected_original_order")
            steps.append(Step(Operation.PRUNE, previous, target, dependency)); previous = target
        if gene.recipe.rank is not None:
            target = replace(previous, kind=Representation.FACTORS, rank=gene.recipe.rank)
            steps.append(Step(Operation.SVD, previous, target, dependency)); previous = target
        if gene.recipe.ptq:
            target = replace(previous, kind=Representation.QDQ, storage_dtype="int8", accumulator_dtype="int32")
            # This symbolic predecessor is rebound to the actual recovered tensor
            # fingerprint by export_onnx_qdq, never used as a calibration-cache key.
            current = Dependency(source, calibration, previous.channel_coordinates, "current_student",
                                 "deferred_recovered_candidate:" + identity)
            deferred.append(Step(Operation.PTQ, previous, target, current, gene.recipe.ptq_placement))
            regions.append(gene.path)
        branches.append(Scope(gene.path, Sequence(tuple(steps)), source, (gene.path,), (gene.path,),
                              (BLOCK_INPUTS[index], outgoing)))
    procedure = JoinApplyReplacements(ParallelDisjoint(tuple(branches), source), identity + ":float_join")
    return HeterogeneousWorkflow(space, normalized, base, procedure, tuple(regions), tuple(deferred), identity)


def validate_workflow(workflow):
    if type(workflow) is not HeterogeneousWorkflow:
        return _failure("UncheckedWorkflow", "Checked G1 workflow required")
    expected = compile_genotype(workflow.genotype, workflow.space)
    return None if expected == workflow else _failure("WorkflowBindingMismatch", "Recompilation changed workflow dependencies")


def execute_float_workflow(model, calibration, workflow):
    failure = validate_workflow(workflow)
    if failure is not None:
        from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
        return StructuralInvalid(failure.violations)
    result = execute_cv_workflow(model, calibration, workflow.structural_workflow)
    from fedcore.algorithm.low_rank.cv_execution import CVWorkflowSuccess
    if type(result) is CVWorkflowSuccess:
        evidence = dict(result.evidence)
        evidence.update(heterogeneous_workflow_fingerprint=workflow.fingerprint,
                        structural_representation_fingerprint=fingerprint(result.model.representation_config()))
        result.model._fedcore_method_evidence = evidence
        from fedcore.tools.registry.checkpoint_manager import CheckpointManager
        checkpoint = CheckpointManager(None).serialize_to_bytes(result.model)
        return replace(result, checkpoint_bytes=checkpoint, evidence=evidence)
    return result
