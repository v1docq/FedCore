"""Independent structural witnesses for the bounded frozen-M0 grammar.

The oracle below follows the declared artifact meanings and producer graph. It
does not call the production validators or compiler to construct expectations.
"""
from collections import Counter
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
import random
from types import SimpleNamespace

from hypothesis import given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
import numpy as np
import pytest
import torch
from torch import nn

from fedcore.algorithm.low_rank.compression_graph import (
    ArtifactKind, CalibrationRef, ModelVersion, Recipe, RuntimeOptions, Stage,
)
from fedcore.algorithm.low_rank.compression_grammar import (
    ASVDChoice, AffinePCAChoice, AffineWholeProfilePlan, CheckedWorkflow,
    CompressionGenotype, GrammarSpace, MutationKind, NoRecovery, ScopeDomain,
    ScopeGene, WeightedChoice, compile_genotype, crossover_genotypes,
    generate_genotype, genotype_from_payload, genotype_payload, mutate_genotype, normalize_genotype,
    validate_grammar_space, validate_workflow,
)
from fedcore.algorithm.low_rank.method_specs import ASVD
from fedcore.algorithm.low_rank.plans import MetricPolicy, OperatorDescriptor, ValidationFailure


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _metadata(value):
    """A stricter, independent finite-value walk, with no production helper."""
    if value is None or type(value) in (str, bool, int) or isinstance(value, Enum):
        return
    if type(value) is float:
        assert math.isfinite(value)
        return
    if isinstance(value, tuple):
        for item in value:
            _metadata(item)
        return
    assert is_dataclass(value) and type(value).__dataclass_params__.frozen
    for field in fields(value):
        _metadata(getattr(value, field.name))


def _codes(value):
    assert isinstance(value, ValidationFailure), value
    assert value.violations and all(item.code and item.path and item.message for item in value.violations)
    return {item.code for item in value.violations}


def _profiles(rank_limit):
    return (
        WeightedChoice(1), WeightedChoice(rank_limit),
        WeightedChoice(1, "one_layer"), WeightedChoice(1, "three_layers"),
        WeightedChoice(1, policy=MetricPolicy(ridge=.25, rcond=.01, nullspace_policy="preserve_nullspace")),
        ASVDChoice(1), ASVDChoice(rank_limit),
        ASVDChoice(1, spec=ASVD(abs_mode="abs_max", alpha=.75, epsilon=.1, zero_policy="error")),
        AffinePCAChoice(1), AffinePCAChoice(rank_limit),
    )


@st.composite
def _spaces(draw):
    domains, aliases = [], []
    for index in range(draw(st.integers(1, 3))):
        rows, columns = draw(st.integers(2, 5)), draw(st.integers(2, 5))
        scope = ((f"alias_{index}", f"layer_{index}") if draw(st.booleans()) else (f"layer_{index}",))
        if len(scope) > 1:
            aliases.append(scope)
        operator = OperatorDescriptor(scope[0], (rows, columns), "Linear", "float64",
                                      bias_elements=rows if draw(st.booleans()) else 0,
                                      storage_id=f"storage:{index}")
        alternatives = draw(st.lists(st.sampled_from(_profiles(min(rows, columns))),
                                    min_size=2, max_size=8, unique=True))
        domains.append(ScopeDomain(operator, scope, tuple(alternatives)))
    return GrammarSpace(ModelVersion("independent-M0", "1" * 64),
        CalibrationRef("2" * 64, (7, 4), "float64"), tuple(domains), tuple(aliases),
        RuntimeOptions(batch_size=3, max_workspace_bytes=1024**2, max_peak_bytes=8 * 1024**2))


@st.composite
def _genotypes(draw, space):
    selected = draw(st.sets(st.integers(0, len(space.domains) - 1), min_size=1,
                            max_size=len(space.domains)))
    genes = tuple(ScopeGene(space.domains[index].aliases,
                           draw(st.sampled_from(space.domains[index].alternatives)))
                  for index in sorted(selected))
    return CompressionGenotype(tuple(draw(st.permutations(genes))))


@st.composite
def _cases(draw):
    space = draw(_spaces())
    return space, draw(_genotypes(space))


def _identity(ref):
    return ref.key, ref.kind, ref.model_version, ref.scope, ref.coordinates


def _workflow_oracle(workflow):
    """Prove frozen source, type closure, complete scopes, and recipe meaning."""
    assert type(workflow) is CheckedWorkflow
    _metadata(workflow)
    assert workflow.source_semantics == "independent_frozen_M0_then_atomic_batch_apply"
    genes = workflow.genotype.genes
    assert tuple(gene.aliases for gene in genes) == tuple(sorted(gene.aliases for gene in genes))
    assert len({gene.aliases for gene in genes}) == len(genes) == len(workflow.scopes)
    domains = {domain.aliases: domain for domain in workflow.space.domains}
    expected_scopes = {gene.aliases for gene in genes}
    selected_aliases = tuple(alias for gene in genes for alias in gene.aliases)
    seen_aliases = set()
    for compiled, gene in zip(workflow.scopes, genes):
        assert compiled.gene == gene and type(gene.recovery) is NoRecovery
        assert gene.choice in domains[gene.aliases].alternatives
        assert not seen_aliases.intersection(gene.aliases)
        seen_aliases.update(gene.aliases)
        operator = domains[gene.aliases].operator
        assert gene.aliases == tuple(sorted(gene.aliases)) and operator.path == gene.aliases[0]
        rows = operator.shape[0] // operator.groups
        width = math.prod(operator.shape[1:])
        assert type(gene.choice.rank) is int and 1 <= gene.choice.rank <= min(rows, width)
        plan = compiled.plan
        assert plan.source == workflow.space.source and plan.calibration == workflow.space.calibration
        transform = plan.transform
        assert transform.steps[0].operator == operator
        assert transform.steps[0].rank == gene.choice.rank
        assert transform.request.representation == gene.choice.representation
        assert transform.request.statistics_refs == ()
        if type(gene.choice) is WeightedChoice:
            assert plan.recipe is Recipe.WEIGHTED_SVD and plan.method_spec is None
            assert transform.request.objective == "input_second_moment"
            assert transform.request.metric_policy == gene.choice.policy
            assert (gene.choice.centering, gene.choice.bias_policy) == ("none", "preserve")
        elif type(gene.choice) is ASVDChoice:
            assert plan.recipe is Recipe.ASVD and plan.method_spec == gene.choice.spec
            assert transform.request.objective == "asvd_diagonal"
            assert transform.request.metric_policy == MetricPolicy()
            assert (gene.choice.centering, gene.choice.bias_policy) == ("none", "preserve")
        else:
            assert type(gene.choice) is AffinePCAChoice and type(plan) is AffineWholeProfilePlan
            assert transform.request.objective == "centered_affine_output_pca"
            assert plan.scope.operator == operator and plan.scope.aliases == gene.aliases
            assert (plan.centering, plan.bias_policy) == ("output_mean", "affine_mean_correction")
    assert len(workflow.external_artifacts) == 2
    external = {ref.kind: ref for ref in workflow.external_artifacts}
    assert set(external) == {ArtifactKind.MODEL, ArtifactKind.CALIBRATION}
    model, calibration = external[ArtifactKind.MODEL], external[ArtifactKind.CALIBRATION]
    assert model.model_version.startswith(workflow.space.source.fingerprint + ":")
    assert model.model_version == calibration.model_version
    assert model.scope == calibration.scope == ()
    assert (model.coordinates, calibration.coordinates) == ("model_graph", "calibration_input")
    available = {ref.key: ref for ref in workflow.external_artifacts}
    stage_nodes = {scope: {} for scope in expected_scopes}
    apply_nodes = []
    for node in workflow.nodes:
        for port in node.inputs:
            assert port.artifact in available, (node.key, port.artifact)
            assert available[port.artifact].kind is port.kind
        if node.stage is Stage.APPLY:
            apply_nodes.append(node)
            assert node is workflow.nodes[-1]
            assert node.scope == node.writes == selected_aliases and node.reads == ()
        else:
            assert node.scope in expected_scopes
            assert node.stage not in stage_nodes[node.scope]
            stage_nodes[node.scope][node.stage] = node
            assert node.writes == ()
            assert node.reads == (node.scope if node.stage in (Stage.OBSERVE, Stage.SOLVE, Stage.BUILD) else ())
            for output in node.outputs:
                assert output.scope == node.scope
                assert output.model_version == model.model_version
                assert output.coordinates == "original_operator_input_output"
        for output in node.outputs:
            assert output.key not in available
            available[output.key] = output
    assert len(workflow.nodes) == len(genes) * 4 + 1 and len(apply_nodes) == 1
    replacements = []
    for gene in genes:
        stages = stage_nodes[gene.aliases]
        assert set(stages) == {Stage.OBSERVE, Stage.METRIC, Stage.SOLVE, Stage.BUILD}
        observe, metric, solve, build = (stages[stage] for stage in
            (Stage.OBSERVE, Stage.METRIC, Stage.SOLVE, Stage.BUILD))
        assert tuple(port.artifact for port in observe.inputs) == (model.key, calibration.key)
        expected_stats = ((ArtifactKind.INPUT_SECOND_MOMENT,) if type(gene.choice) is WeightedChoice
            else (ArtifactKind.INPUT_ABS_MEAN if gene.choice.spec.abs_mode == "abs_mean" else ArtifactKind.INPUT_ABS_MAX,)
            if type(gene.choice) is ASVDChoice else
            (ArtifactKind.OUTPUT_MEAN, ArtifactKind.CENTERED_OUTPUT_COVARIANCE))
        expected_metric = (ArtifactKind.FULL_METRIC if type(gene.choice) is WeightedChoice
            else ArtifactKind.DIAGONAL_METRIC if type(gene.choice) is ASVDChoice else ArtifactKind.OUTPUT_PROJECTOR)
        assert tuple(ref.kind for ref in observe.outputs) == expected_stats
        assert tuple(port.artifact for port in metric.inputs) == tuple(ref.key for ref in observe.outputs)
        assert len(metric.outputs) == 1 and metric.outputs[0].kind is expected_metric
        assert tuple(port.artifact for port in solve.inputs) == (model.key, metric.outputs[0].key)
        assert len(solve.outputs) == 1 and solve.outputs[0].kind is ArtifactKind.APPROXIMATION
        expected_build = (model.key, solve.outputs[0].key)
        if type(gene.choice) is AffinePCAChoice:
            expected_build += (observe.outputs[0].key,)
        assert tuple(port.artifact for port in build.inputs) == expected_build
        assert len(build.outputs) == 1 and build.outputs[0].kind is ArtifactKind.REPLACEMENT
        replacements.append(build.outputs[0].key)
    apply = apply_nodes[0]
    assert tuple(port.artifact for port in apply.inputs) == (model.key, *replacements)
    assert len(apply.outputs) == 1 and apply.outputs[0].kind is ArtifactKind.MODEL
    assert apply.outputs[0].model_version != model.model_version
    assert apply.outputs[0].scope == () and apply.outputs[0].coordinates == "model_graph"
    return {_identity(ref) for node in workflow.nodes for ref in node.outputs}


@settings(max_examples=35, deadline=None)
@given(case=_cases())
def test_independent_valid_construction_and_normalization_preserve_exact_choices(case):
    space, genotype = case
    expected = CompressionGenotype(tuple(sorted(genotype.genes, key=lambda gene: gene.aliases)))
    normalized = normalize_genotype(genotype, space)
    assert normalized == expected and normalize_genotype(normalized, space) == expected
    workflow = compile_genotype(genotype, space)
    _workflow_oracle(workflow)
    assert compile_genotype(normalized, space) == workflow


@settings(max_examples=30, deadline=None)
@given(case=_cases(), seed=st.integers(0, 10000))
def test_generation_is_bounded_deterministic_and_does_not_touch_global_randomness(case, seed):
    space, _ = case
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    generated = generate_genotype(space, seed=seed)
    assert generated == generate_genotype(space, seed=seed)
    assert 1 <= len(generated.genes) <= len(space.domains)
    _workflow_oracle(compile_genotype(generated, space))
    changed = mutate_genotype(generated, space, kind=MutationKind.CHOICE, seed=seed)
    _workflow_oracle(compile_genotype(changed, space))
    for child in crossover_genotypes(generated, changed, space, seed=seed):
        _workflow_oracle(compile_genotype(child, space))
    assert random.getstate() == python_state
    now = np.random.get_state()
    assert now[0] == numpy_state[0] and np.array_equal(now[1], numpy_state[1]) and now[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def _decode_supported_payload(payload):
    """Simple independent transport model: retain every conditional field."""
    genes = []
    for item in payload["genes"]:
        choice = dict(item["choice"])
        if item["recipe"] == "weighted_svd":
            choice["policy"] = MetricPolicy(**choice["policy"])
            value = WeightedChoice(**choice)
        elif item["recipe"] == "asvd":
            choice["spec"] = ASVD(**choice["spec"])
            value = ASVDChoice(**choice)
        else:
            assert item["recipe"] == "afm"
            value = AffinePCAChoice(**choice)
        assert item["recovery"] == "none"
        genes.append(ScopeGene(tuple(item["aliases"]), value))
    return CompressionGenotype(tuple(genes), payload["schema_version"])


@settings(max_examples=30, deadline=None)
@given(case=_cases())
def test_json_export_preserves_all_conditional_parameters_under_independent_decode(case):
    space, genotype = case
    payload = genotype_payload(genotype)
    expected = {"schema_version": 1, "genes": [
        {"aliases": list(gene.aliases), "recipe": {WeightedChoice: "weighted_svd", ASVDChoice: "asvd",
            AffinePCAChoice: "afm"}[type(gene.choice)], "choice": _plain(gene.choice), "recovery": "none"}
        for gene in genotype.genes]}
    assert payload == expected
    restored = _decode_supported_payload(json.loads(json.dumps(payload, allow_nan=False)))
    assert restored == genotype
    assert genotype_from_payload(payload, space) == CompressionGenotype(tuple(sorted(genotype.genes, key=lambda gene: gene.aliases)))


def _fixed_space():
    aliases = (("a", "a_shared"),)
    domains = tuple(ScopeDomain(OperatorDescriptor(scope[0], (3, 4), "Linear", "float64",
                                                  bias_elements=3, storage_id=f"storage:{index}"),
                                scope, _profiles(3))
                    for index, scope in enumerate((aliases[0], ("b",), ("c",))))
    return GrammarSpace(ModelVersion("state-M0", "3" * 64),
                        CalibrationRef("4" * 64, (6, 4), "float64"), domains, aliases,
                        RuntimeOptions(max_workspace_bytes=1024**2, max_peak_bytes=8 * 1024**2))


_STATE_SPACE = _fixed_space()


def _gene_map(genotype):
    return {gene.aliases: gene for gene in genotype.genes}


def _eligible_choices(choice, alternatives, kind):
    candidates = []
    for alternative in alternatives:
        if alternative == choice:
            continue
        first, second = _plain(choice), _plain(alternative)
        if kind is MutationKind.RANK:
            eligible = type(choice) is type(alternative) and {
                key: value for key, value in first.items() if key != "rank"} == {
                key: value for key, value in second.items() if key != "rank"}
        elif kind is MutationKind.REPRESENTATION:
            eligible = type(choice) is type(alternative) and {
                key: value for key, value in first.items() if key != "representation"} == {
                key: value for key, value in second.items() if key != "representation"}
        elif kind is MutationKind.CENTERING:
            eligible = choice.centering != alternative.centering
        elif kind is MutationKind.METRIC:
            eligible = (choice.rank, choice.representation, choice.centering) == (
                alternative.rank, alternative.representation, alternative.centering)
        else:
            eligible = True
        if eligible:
            candidates.append(alternative)
    return candidates


def _assert_mutation_model(before, after, kind, space):
    old, new = _gene_map(before), _gene_map(after)
    if kind is MutationKind.SCOPE:
        assert len(old.keys() ^ new.keys()) == 1
        assert all(old[key] == new[key] for key in old.keys() & new.keys())
        return
    assert old.keys() == new.keys()
    changed = [key for key in old if old[key] != new[key]]
    domains = {domain.aliases: domain for domain in space.domains}
    moves_exist = any(_eligible_choices(gene.choice, domains[key].alternatives, kind)
                      for key, gene in old.items())
    assert len(changed) == int(moves_exist)
    for key in changed:
        first, second = old[key].choice, new[key].choice
        assert old[key].recovery == new[key].recovery
        if kind is MutationKind.RANK:
            assert type(first) is type(second) and first.rank != second.rank
            assert {k: v for k, v in _plain(first).items() if k != "rank"} == {
                k: v for k, v in _plain(second).items() if k != "rank"}
        elif kind is MutationKind.REPRESENTATION:
            assert type(first) is type(second) and first.representation != second.representation
            assert {k: v for k, v in _plain(first).items() if k != "representation"} == {
                k: v for k, v in _plain(second).items() if k != "representation"}
        elif kind is MutationKind.METRIC:
            assert (first.rank, first.representation, first.centering) == (
                second.rank, second.representation, second.centering)
        elif kind is MutationKind.CENTERING:
            assert first.centering != second.centering


class GrammarTransitionModel(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.space = _STATE_SPACE
        self.current = CompressionGenotype((ScopeGene(("a", "a_shared"), WeightedChoice(1)),
                                            ScopeGene(("b",), ASVDChoice(1))))
        self.epoch = 0

    @rule(kind=st.sampled_from(tuple(kind for kind in MutationKind if kind is not MutationKind.RECOVERY)),
          seed=st.integers(0, 10000))
    def mutate_one_closed_area(self, kind, seed):
        old = self.current
        result = mutate_genotype(old, self.space, kind=kind, seed=seed)
        assert type(result) is CompressionGenotype
        _assert_mutation_model(old, result, kind, self.space)
        assert result == mutate_genotype(old, self.space, kind=kind, seed=seed)
        self.current = result

    @rule(donor=_genotypes(_STATE_SPACE), seed=st.integers(0, 10000))
    def exchange_complete_scope_with_donor(self, donor, seed):
        old, other = _gene_map(self.current), _gene_map(donor)
        children = crossover_genotypes(self.current, donor, self.space, seed=seed)
        assert isinstance(children, tuple) and len(children) == 2
        first, second = map(_gene_map, children)
        common = old.keys() & other.keys()
        if common:
            assert first.keys() == old.keys() and second.keys() == other.keys()
            changed = {key for key in common if first[key] != old[key] or second[key] != other[key]}
            assert len(changed) <= 1
            for key in common:
                assert Counter((first[key], second[key])) == Counter((old[key], other[key]))
            assert all(first[key] == old[key] for key in old.keys() - common)
            assert all(second[key] == other[key] for key in other.keys() - common)
        else:
            assert first == other and second == old
        _workflow_oracle(compile_genotype(children[1], self.space))
        self.current = children[0]

    @rule(change_data=st.booleans())
    def rebind_to_fresh_source_or_calibration(self, change_data):
        old_workflow = compile_genotype(self.current, self.space)
        old_refs = _workflow_oracle(old_workflow)
        self.epoch += 1
        if change_data:
            self.space = replace(self.space, calibration=replace(self.space.calibration,
                                 fingerprint=f"{self.epoch + 256:064x}"))
        else:
            self.space = replace(self.space, source=ModelVersion(f"state-M0-{self.epoch}",
                                                                 f"{self.epoch:064x}"))
        fresh = compile_genotype(self.current, self.space)
        fresh_refs = _workflow_oracle(fresh)
        assert fresh.genotype == old_workflow.genotype
        assert not old_refs.intersection(fresh_refs)
        assert fresh.fingerprint != old_workflow.fingerprint

    @invariant()
    def all_intermediate_states_are_closed_metadata_workflows(self):
        _workflow_oracle(compile_genotype(self.current, self.space))


TestGrammarTransitionModel = GrammarTransitionModel.TestCase
TestGrammarTransitionModel.settings = settings(max_examples=15, stateful_step_count=8, deadline=None)


@pytest.mark.parametrize("choice", [WeightedChoice(0), WeightedChoice(True), WeightedChoice(4),
    ASVDChoice(-1), ASVDChoice(1.5), AffinePCAChoice(4)])
def test_invalid_rank_is_rejected_without_clipping_or_domain_pruning(choice):
    space = _fixed_space()
    bad = replace(space.domains[0], alternatives=(choice, WeightedChoice(1)))
    assert "InvalidRank" in _codes(validate_grammar_space(replace(space, domains=(bad,))))


@pytest.mark.parametrize("choice", [WeightedChoice(1, centering="output_mean"),
    WeightedChoice(1, bias_policy="affine_mean_correction"), ASVDChoice(1, centering="output_mean"),
    ASVDChoice(1, bias_policy="affine_mean_correction"), AffinePCAChoice(1, centering="none"),
    AffinePCAChoice(1, bias_policy="preserve")])
def test_incompatible_centering_and_bias_are_rejected_before_normalization(choice):
    space = _fixed_space()
    bad_space = replace(space, domains=(replace(space.domains[0], alternatives=(choice,)),))
    genotype = CompressionGenotype((ScopeGene(space.domains[0].aliases, choice),))
    assert "IncompatibleCenteringBias" in _codes(normalize_genotype(genotype, bad_space))


@pytest.mark.parametrize("change", ["partial_alias", "duplicate", "unknown_scope", "recovery", "outside_domain"])
def test_invalid_genotype_has_explicit_structural_failure_and_is_never_repaired(change):
    space = _fixed_space()
    gene = ScopeGene(("a", "a_shared"), WeightedChoice(1))
    genes = (gene,)
    if change == "partial_alias":
        genes = (replace(gene, aliases=("a",)),)
    elif change == "duplicate":
        genes = (gene, gene)
    elif change == "unknown_scope":
        genes = (replace(gene, aliases=("unknown",)),)
    elif change == "recovery":
        genes = (replace(gene, recovery="train"),)
    else:
        genes = (replace(gene, choice=WeightedChoice(2)),)
    invalid = CompressionGenotype(genes)
    failure = normalize_genotype(invalid, space)
    assert _codes(failure) & {"InvalidScope", "UnsupportedRecovery", "ChoiceOutsideDomain"}
    assert compile_genotype(invalid, space) == failure


@pytest.mark.parametrize("limit,value", [("max_workspace_bytes", 0), ("max_peak_bytes", 0),
    ("max_peak_bytes", 12 * 1024**3 + 1), ("max_workspace_bytes", 16 * 1024**2),
    ("batch_size", True), ("parameter_budget", -1), ("tensor_byte_budget", True)])
def test_resource_metadata_is_admitted_before_any_generation(limit, value):
    space = _fixed_space()
    bad = replace(space, options=replace(space.options, **{limit: value}))
    assert "InvalidResourceLimit" in _codes(generate_genotype(bad))


def _reseal(workflow):
    encoded = json.dumps(_plain(replace(workflow, fingerprint="")), sort_keys=True,
                         allow_nan=False, separators=(",", ":")).encode()
    return replace(workflow, fingerprint=hashlib.sha256(encoded).hexdigest())


@pytest.mark.parametrize("damage", ["missing_stage", "wrong_semantic_type", "foreign_scope", "stale_source", "missing_mean"])
def test_matching_content_hash_cannot_authorize_missing_or_stale_dependencies(damage):
    space = _fixed_space()
    genotype = CompressionGenotype((ScopeGene(("a", "a_shared"), AffinePCAChoice(1)),
                                    ScopeGene(("b",), ASVDChoice(1))))
    workflow = compile_genotype(genotype, space)
    nodes = list(workflow.nodes)
    if damage == "missing_stage":
        del nodes[1]
    elif damage == "wrong_semantic_type":
        node = nodes[0]
        nodes[0] = replace(node, outputs=(replace(node.outputs[0], kind=ArtifactKind.INPUT_MEAN), *node.outputs[1:]))
    elif damage == "foreign_scope":
        nodes[2] = replace(nodes[2], scope=("b",))
    elif damage == "stale_source":
        node = nodes[0]
        nodes[0] = replace(node, outputs=tuple(replace(ref, model_version="stale-M0") for ref in node.outputs))
    else:
        nodes[3] = replace(nodes[3], inputs=nodes[3].inputs[:2])
    forged = _reseal(replace(workflow, nodes=tuple(nodes)))
    assert "WorkflowBindingMismatch" in _codes(validate_workflow(forged))


@pytest.mark.parametrize("bad", [CompressionGenotype(("untyped-gene",)),
    CompressionGenotype((ScopeGene(("a",), "unknown-choice"),)),
    CompressionGenotype((ScopeGene(("a",), WeightedChoice(1), "train"),)),
    CompressionGenotype((), schema_version=True)])
def test_export_of_malformed_metadata_returns_typed_failure(bad):
    assert _codes(genotype_payload(bad)) & {"InvalidGenotype", "InvalidScope", "InvalidChoice", "UnsupportedRecovery"}


@pytest.mark.parametrize("damage", ["unknown_top_level", "extra_conditional_parameter", "missing_conditional_parameter",
    "wrong_recipe_tag", "partial_alias", "recovery", "bool_schema", "nonfinite_metric"])
def test_payload_decode_rejects_unknown_or_incompatible_fields_without_repair(damage):
    space = _fixed_space()
    genotype = CompressionGenotype((ScopeGene(("a", "a_shared"), WeightedChoice(1)),))
    payload = json.loads(json.dumps(genotype_payload(genotype)))
    item = payload["genes"][0]
    if damage == "unknown_top_level":
        payload["cached_score"] = 0.
    elif damage == "extra_conditional_parameter":
        item["choice"]["spec"] = {"alpha": .5}
    elif damage == "missing_conditional_parameter":
        del item["choice"]["policy"]
    elif damage == "wrong_recipe_tag":
        item["recipe"] = "afm"
    elif damage == "partial_alias":
        item["aliases"] = ["a"]
    elif damage == "recovery":
        item["recovery"] = "train"
    elif damage == "bool_schema":
        payload["schema_version"] = True
    else:
        item["choice"]["policy"]["ridge"] = float("nan")
    assert _codes(genotype_from_payload(payload, space))


def test_only_declared_no_recovery_capability_can_be_mutated():
    space = _fixed_space()
    genotype = CompressionGenotype((ScopeGene(("b",), ASVDChoice(1)),))
    assert "UnsupportedRecovery" in _codes(mutate_genotype(genotype, space, kind=MutationKind.RECOVERY))


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 10000))
def test_crossover_exchanges_one_whole_scope_when_every_common_choice_differs(seed):
    space = _fixed_space()
    first = CompressionGenotype(tuple(ScopeGene(domain.aliases, WeightedChoice(1)) for domain in space.domains))
    second = CompressionGenotype(tuple(ScopeGene(domain.aliases, AffinePCAChoice(1)) for domain in space.domains))
    children = crossover_genotypes(first, second, space, seed=seed)
    old, donor = _gene_map(first), _gene_map(second)
    left, right = map(_gene_map, children)
    assert sum(left[key] != old[key] for key in old) == 1
    assert sum(right[key] != donor[key] for key in donor) == 1
    for key in old:
        assert Counter((left[key], right[key])) == Counter((old[key], donor[key]))
    for child in children:
        _workflow_oracle(compile_genotype(child, space))


@pytest.mark.parametrize("groups,rows,columns,kernel", [(1, 2, 3, 1), (2, 3, 2, 3), (3, 2, 2, 1)])
def test_grouped_convolution_rank_is_per_group_and_whole_alias_scope_is_preserved(groups, rows, columns, kernel):
    operator = OperatorDescriptor("conv", (groups * rows, columns, kernel), "Conv1d", "float64",
        groups=groups, kernel_size=(kernel,), stride=(1,), padding=(0,), dilation=(1,), bias_elements=groups * rows)
    cap = min(rows, columns * kernel)
    domain = ScopeDomain(operator, ("conv", "conv_alias"), (WeightedChoice(1), WeightedChoice(cap)))
    space = GrammarSpace(ModelVersion("conv-M0", "5" * 64),
        CalibrationRef("6" * 64, (4, groups * columns, 7), "float64"), (domain,), (("conv", "conv_alias"),))
    genotype = CompressionGenotype((ScopeGene(domain.aliases, WeightedChoice(cap)),))
    _workflow_oracle(compile_genotype(genotype, space))
    bad = replace(space, domains=(replace(domain, alternatives=(WeightedChoice(cap + 1),)),))
    assert "InvalidRank" in _codes(validate_grammar_space(bad))


def _runtime_workflow(model, data, choices, **options):
    from fedcore.algorithm.low_rank.grammar_execution import prepare_grammar_space
    space = prepare_grammar_space(model, data, alternatives={path: (choice,) for path, choice in choices.items()},
                                  options=RuntimeOptions(**options))
    assert type(space) is GrammarSpace, space
    genotype = CompressionGenotype(tuple(ScopeGene(domain.aliases, choices[domain.aliases[0]])
                                        for domain in space.domains))
    workflow = compile_genotype(genotype, space)
    _workflow_oracle(workflow)
    return workflow


@pytest.mark.parametrize("case", ["weighted_overflow", "afm_overflow", "asvd_zero", "asvd_scale_overflow"])
def test_real_numerical_failures_are_typed_and_never_escape_as_structural_or_raw_exceptions(case):
    from fedcore.algorithm.low_rank.grammar_execution import NumericalFailure, execute_workflow
    model = nn.Sequential(nn.Linear(2, 2, bias=False, dtype=torch.float64)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.diag(torch.tensor([1., 2.], dtype=torch.float64)))
    if case == "asvd_zero":
        data = torch.tensor([[0., 1.], [0., -1.], [0., 2.]], dtype=torch.float64)
        choice = ASVDChoice(1, ASVD(zero_policy="error"))
    else:
        size = 1e100 if case == "asvd_scale_overflow" else 1e200
        data = torch.tensor([[size, 0.], [-size, 0.], [0., size], [0., -size]], dtype=torch.float64)
        choice = (ASVDChoice(1, ASVD(alpha=4.)) if case == "asvd_scale_overflow"
                  else AffinePCAChoice(1) if case == "afm_overflow" else WeightedChoice(1))
    assert bool(torch.isfinite(data).all())
    workflow = _runtime_workflow(model, data, {"0": choice})
    original_data, original_weight, original_rng = data.clone(), model[0].weight.clone(), torch.get_rng_state().clone()
    outcome = execute_workflow(model, data, workflow)
    assert type(outcome) is NumericalFailure, outcome
    assert outcome.code and outcome.message
    assert torch.equal(data, original_data) and torch.equal(model[0].weight, original_weight)
    assert torch.equal(torch.get_rng_state(), original_rng)


def _physical_cost(model):
    """Count real registered backing storages without production inventories."""
    storages = {}
    for tensor in (*model.parameters(), *model.buffers()):
        storage = tensor.untyped_storage()
        storages.setdefault(storage._cdata, (storage.nbytes(), tensor.element_size()))
    return sum(size for size, _ in storages.values()), sum(size // width for size, width in storages.values())


@pytest.mark.parametrize("choice", [WeightedChoice(1), ASVDChoice(1), AffinePCAChoice(1)])
def test_physical_budget_boundary_includes_added_bias_and_nonpersistent_integer_buffer(choice):
    from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure, WorkflowSuccess, execute_workflow
    model = nn.Sequential(nn.Linear(3, 4, bias=False, dtype=torch.float64)).eval()
    model.register_buffer("fixed_map", torch.arange(5, dtype=torch.int16), persistent=False)
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[3., 0., 0.], [0., 2., 0.], [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64))
    data = torch.tensor([[1., 2., 3.], [-1., 2., 0.], [2., 3., 1.], [0., 1., 2.]], dtype=torch.float64)
    workflow = _runtime_workflow(model, data, {"0": choice})
    outcome = execute_workflow(model, data, workflow)
    assert type(outcome) is WorkflowSuccess, outcome
    expected_elements = 7 + 5 + (4 if type(choice) is AffinePCAChoice else 0)
    expected_bytes = (7 + (4 if type(choice) is AffinePCAChoice else 0)) * 8 + 5 * 2
    assert _physical_cost(outcome.model) == (expected_bytes, expected_elements)
    assert outcome.evidence["actual_backing_storage_bytes"] == expected_bytes
    assert outcome.evidence["actual_stored_elements"] == expected_elements
    assert "fixed_map" in outcome.model._non_persistent_buffers_set
    assert torch.equal(outcome.model.fixed_map, model.fixed_map)
    for field, exact in (("tensor_byte_budget", expected_bytes), ("parameter_budget", expected_elements)):
        for delta in (-1, 0):
            bounded = replace(workflow.space, options=replace(workflow.space.options, **{field: exact + delta}))
            result = execute_workflow(model, data, compile_genotype(workflow.genotype, bounded))
            if delta < 0:
                assert type(result) is ResourceFailure and result.code == "BudgetInfeasible", result
            else:
                assert type(result) is WorkflowSuccess, result


@pytest.mark.parametrize("second_choice", [ASVDChoice(1), AffinePCAChoice(1)])
def test_mixed_scope_replacements_are_computed_from_the_original_M0(second_choice):
    from fedcore.algorithm.low_rank.grammar_execution import WorkflowSuccess, execute_workflow
    model = nn.Sequential(nn.Linear(3, 3, dtype=torch.float64), nn.Tanh(), nn.Linear(3, 2, dtype=torch.float64)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[3., 1., 2.], [1., -2., 1.], [2., 1., 1.]], dtype=torch.float64))
        model[0].bias.copy_(torch.tensor([.5, -.25, 1.], dtype=torch.float64))
        model[2].weight.copy_(torch.tensor([[1., 2., 3.], [-2., 1., 4.]], dtype=torch.float64))
        model[2].bias.copy_(torch.tensor([.1, -.4], dtype=torch.float64))
    data = torch.tensor([[1., 2., 3.], [-1., 2., 0.], [2., 3., 1.], [0., 1., 2.]], dtype=torch.float64)
    choices = {"0": WeightedChoice(1), "2": second_choice}
    before = {key: value.clone() for key, value in model.state_dict().items()}
    combined = execute_workflow(model, data, _runtime_workflow(model, data, choices))
    assert type(combined) is WorkflowSuccess, combined
    probe = torch.eye(3, dtype=torch.float64)
    for path, choice in choices.items():
        separate = execute_workflow(model, data, _runtime_workflow(model, data, {path: choice}))
        assert type(separate) is WorkflowSuccess, separate
        torch.testing.assert_close(combined.model[int(path)](probe), separate.model[int(path)](probe), rtol=0, atol=0)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert combined.evidence["apply_count"] == 1
    assert all(record["source"] == combined.workflow.space.source.fingerprint
               for record in combined.evidence["scope_profiles"])


@pytest.mark.parametrize("choice", [WeightedChoice(1), ASVDChoice(1), AffinePCAChoice(1)])
def test_nonpersistent_forward_effect_is_refused_without_changing_source_data_modes_or_rng(choice):
    from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid, execute_workflow
    class CounterModel(nn.Sequential):
        def __init__(self):
            super().__init__(nn.Linear(2, 2, dtype=torch.float64))
            self.register_buffer("observations", torch.zeros(1), persistent=False)

        def forward(self, value):
            self.observations.add_(1)
            return super().forward(value)

    model = CounterModel().train()
    model[0].eval()
    data = torch.tensor([[1., 2.], [-1., 1.], [0., 2.]], dtype=torch.float64)
    workflow = _runtime_workflow(model, data, {"0": choice})
    before_data = data.clone()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    modes = tuple(layer.training for layer in model.modules())
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    outcome = execute_workflow(model, data, workflow)
    assert type(outcome) is StructuralInvalid, outcome
    assert "UnsupportedObservationEffect" in {item.code for item in outcome.violations}
    assert model.observations.item() == 0 and torch.equal(data, before_data)
    assert tuple(layer.training for layer in model.modules()) == modes
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    now = np.random.get_state()
    assert now[0] == numpy_rng[0] and np.array_equal(now[1], numpy_rng[1]) and now[2:] == numpy_rng[2:]


@pytest.mark.parametrize("change", ["nonpersistent_value", "hidden_storage_tail", "tensor_object_alias"])
def test_source_storage_provenance_refuses_stale_workflow_and_rebinds_every_dependency(monkeypatch, change):
    from fedcore.algorithm.low_rank import grammar_execution
    from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid, execute_workflow, prepare_grammar_space
    model = nn.Sequential(nn.Linear(2, 2, bias=False, dtype=torch.float64)).eval()
    backing = torch.arange(8, dtype=torch.float64)
    model.register_buffer("window", backing[1:3], persistent=False)
    model.register_buffer("mirror", model.window, persistent=False)
    data = torch.tensor([[1., 2.], [-1., 1.], [0., 2.]], dtype=torch.float64)
    workflow = _runtime_workflow(model, data, {"0": WeightedChoice(1)})
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        if change == "nonpersistent_value":
            model.window[0] += 1
        elif change == "hidden_storage_tail":
            raw = torch.empty(0, dtype=torch.float64).set_(model.window.untyped_storage(), 0, (8,), (1,))
            raw[-1] += 1
        else:
            model.mirror = model.window.view_as(model.window)
    assert all(torch.equal(value, state_before[key]) for key, value in model.state_dict().items())
    def no_clone(*args, **kwargs):
        pytest.fail("A stale source reached model cloning")
    monkeypatch.setattr(grammar_execution, "clone_preserving_storages", no_clone)
    outcome = execute_workflow(model, data, workflow)
    assert type(outcome) is StructuralInvalid, outcome
    assert outcome.violations and all(item.code and item.path for item in outcome.violations)
    rebound = prepare_grammar_space(model, data, alternatives={"0": (WeightedChoice(1),)})
    assert type(rebound) is GrammarSpace
    # state_dict-based provenance alone cannot detect any of the three changes.
    assert rebound.source == workflow.space.source
    assert rebound != workflow.space
    fresh = compile_genotype(workflow.genotype, rebound)
    assert fresh.genotype == workflow.genotype
    assert not _workflow_oracle(workflow).intersection(_workflow_oracle(fresh))


@pytest.mark.parametrize("kind", ["structural", "numerical", "resource", "callback", "poor_quality", "success"])
def test_facade_publishes_only_success_and_marks_its_trainer_for_optimizer_rebuild(monkeypatch, kind):
    from fedcore.algorithm.low_rank import grammar_execution
    from fedcore.algorithm.low_rank.grammar_execution import (
        CallbackFailure, NumericalFailure, PoorQuality, ResourceFailure, StructuralInvalid, WorkflowSuccess,
    )
    from fedcore.algorithm.low_rank.low_rank_opt import LowRankModel
    from fedcore.algorithm.low_rank.plans import Violation
    source, replacement = nn.Linear(2, 2), nn.Linear(2, 2)
    sentinel_before, sentinel_after, sentinel_result = object(), object(), object()
    facade = LowRankModel.__new__(LowRankModel)
    facade._model_before_cached, facade._model_after_cached = sentinel_before, sentinel_after
    facade._model_id_before = facade._model_id_after = None
    facade._fedcore_id = "independent-facade-fixture"
    registrations = []
    def register(**options):
        registrations.append(options)
        return options["stage"]
    facade._registry = SimpleNamespace(register_model=register)
    facade.transform_result = sentinel_result
    facade.trainer = SimpleNamespace(model=source)
    workflow = compile_genotype(CompressionGenotype((ScopeGene(("b",), ASVDChoice(1)),)), _fixed_space())
    success = WorkflowSuccess(replacement, workflow, {"scope": "facade fixture"})
    outcomes = {
        "structural": StructuralInvalid((Violation("Fixture", "workflow", "Structural rejection"),)),
        "numerical": NumericalFailure("Fixture", "Numerical rejection"),
        "resource": ResourceFailure("Fixture", "Resource rejection"),
        "callback": CallbackFailure("Fixture", "Callback rejection"),
        "poor_quality": PoorQuality(success, 2., 1.), "success": success,
    }
    monkeypatch.setattr(grammar_execution, "execute_workflow", lambda *args: outcomes[kind])
    assert facade.compress_workflow(source, torch.zeros(2, 2), workflow) is outcomes[kind]
    if kind == "success":
        assert facade.model_before is source and facade.model_after is replacement and facade.transform_result is success
        assert facade.trainer.model is replacement and facade.trainer._fedcore_requires_optimizer_rebuild is True
        assert [(item["stage"], item["model"]) for item in registrations] == [("before", source), ("after", replacement)]
    else:
        assert facade.model_before is sentinel_before and facade.model_after is sentinel_after
        assert facade.transform_result is sentinel_result and facade.trainer.model is source
        assert not hasattr(facade.trainer, "_fedcore_requires_optimizer_rebuild")
        assert registrations == []


@pytest.mark.parametrize("damage", ["empty_graph", "extra_parameter", "unknown_name"])
def test_native_golem_adapter_returns_explained_structural_failure(damage):
    from golem.core.optimisers.graph import OptGraph
    from fedcore.algorithm.low_rank.grammar_search import grammar_graph, graph_genotype
    space = _fixed_space()
    genotype = CompressionGenotype((ScopeGene(("b",), ASVDChoice(1)),))
    graph = grammar_graph(genotype)
    if damage == "empty_graph":
        graph = OptGraph([])
    elif damage == "extra_parameter":
        graph.nodes[0].parameters["score"] = 0.
    else:
        graph.nodes[0].content["name"] = "unrelated_pipeline"
    assert _codes(graph_genotype(graph, space))


@pytest.mark.parametrize("callback_raises", [False, True])
def test_budget_stop_after_only_failed_evaluations_is_no_valid_fitness(callback_raises):
    from fedcore.algorithm.low_rank.grammar_execution import NumericalFailure
    from fedcore.algorithm.low_rank.grammar_search import GrammarSearchConfig, run_grammar_search
    space = _fixed_space()
    initial = (CompressionGenotype((ScopeGene(("b",), WeightedChoice(1)),)),
               CompressionGenotype((ScopeGene(("b",), ASVDChoice(1)),)))
    called = []
    def fail(workflow):
        called.append(workflow.fingerprint)
        if callback_raises:
            raise ArithmeticError("Real callback rejection")
        return NumericalFailure("FixtureNumericalFailure", "No finite fitness exists")
    result = run_grammar_search(space, GrammarSearchConfig(population_size=2, generations=1,
        max_evaluations=1, max_seconds=10), fail, fitness_context="independent all-failed budget witness",
        initial_genotypes=initial)
    assert result["status"] == "no_valid_fitness", result
    assert result["charged_evaluations"] == result["failure_count"] == len(called) == 1
    assert result["best_genotypes"] == []
    assert all(event["fitness"] is None for event in result["trace"] if event["event"] == "evaluated")


def test_real_optimizer_reproduction_calls_both_grammar_operators_and_keeps_its_constraints(monkeypatch):
    from fedcore.algorithm.low_rank import grammar_search
    from fedcore.algorithm.low_rank.grammar_search import GrammarFitness, GrammarSearchConfig
    from fedcore.interfaces.fedcore_optimizer import FedcoreEvoOptimizer
    originals = {name: getattr(grammar_search, name) for name in ("mutate_genotype", "crossover_genotypes")}
    calls = {"mutation": 0, "crossover": 0, "optimise": 0, "reproduction": 0, "extension": 0}
    instances = []
    def mutation(*args, **kwargs):
        calls["mutation"] += 1
        return originals["mutate_genotype"](*args, **kwargs)
    def crossover(*args, **kwargs):
        calls["crossover"] += 1
        return originals["crossover_genotypes"](*args, **kwargs)
    original_optimise = FedcoreEvoOptimizer.optimise
    original_evolve = FedcoreEvoOptimizer._evolve_population
    original_extend = FedcoreEvoOptimizer._extend_population
    def optimise(self, *args, **kwargs):
        calls["optimise"] += 1
        instances.append(self)
        return original_optimise(self, *args, **kwargs)
    def evolve(self, *args, **kwargs):
        calls["reproduction"] += 1
        return original_evolve(self, *args, **kwargs)
    def extend(self, *args, **kwargs):
        calls["extension"] += 1
        rules = tuple(self.graph_generation_params.verifier._rules)
        result = original_extend(self, *args, **kwargs)
        assert tuple(self.graph_generation_params.verifier._rules) == rules
        assert len(self.mutation.agent._probs) == len(self.graph_optimizer_params.mutation_types) == 1
        return result
    monkeypatch.setattr(grammar_search, "mutate_genotype", mutation)
    monkeypatch.setattr(grammar_search, "crossover_genotypes", crossover)
    monkeypatch.setattr(FedcoreEvoOptimizer, "optimise", optimise)
    monkeypatch.setattr(FedcoreEvoOptimizer, "_evolve_population", evolve)
    monkeypatch.setattr(FedcoreEvoOptimizer, "_extend_population", extend)
    space = _fixed_space()
    initial = tuple(CompressionGenotype((ScopeGene(("b",), choice),)) for choice in space.domains[1].alternatives[:4])
    def evaluate(workflow):
        _workflow_oracle(workflow)
        rank = sum(gene.choice.rank for gene in workflow.genotype.genes)
        return GrammarFitness(float(rank), 10. / rank)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    result = grammar_search.run_grammar_search(space, GrammarSearchConfig(seed=13, population_size=4,
        generations=2, max_evaluations=12, max_seconds=30), evaluate,
        fitness_context="independent metadata-only reproduction witness", initial_genotypes=initial)
    assert result["status"] == "succeeded", result
    assert calls["optimise"] == 1 and len(instances) == 1
    assert instances[0].compression_grammar_profile is True
    assert all(calls[name] > 0 for name in ("mutation", "crossover", "reproduction", "extension")), calls
    assert random.getstate() == python_state
    now = np.random.get_state()
    assert now[0] == numpy_state[0] and np.array_equal(now[1], numpy_state[1]) and now[2:] == numpy_state[2:]
