"""Finite heterogeneous catalogue using the existing exact storage allocator."""
from dataclasses import dataclass
import math
import time
import torch

from fedcore.algorithm.low_rank.allocation import (CandidateOption,CandidateTable,StorageCost,
    ScoreObjective,allocate,BudgetInfeasible)
from fedcore.algorithm.low_rank.candidate_storage import capture_inventory,inventory_cost,inventory_fingerprint
from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid,NumericalFailure,ResourceFailure,CallbackFailure
from fedcore.algorithm.low_rank.plans import Violation,ValidationFailure
from fedcore.algorithm.low_rank.topology import module_paths
from .graph_specs import OperatorRecipe,QuantizationGraphRequest,Method,fingerprint
from .graph_execution import (check_quantization_graph,compress_quantization_graph,
    ExportVerifiedArtifact,calibration_identity,source_identity)


@dataclass(frozen=True)
class CatalogueEntry:
    path: str
    aliases: tuple[str,...]
    option_id: str
    recipe: OperatorRecipe
    physical_bytes: int
    local_error: float


@dataclass(frozen=True)
class QuantizationCatalogue:
    tables: tuple[CandidateTable,...]
    entries: tuple[CatalogueEntry,...]
    fixed_storages: tuple[StorageCost,...]
    source_fingerprint: str
    calibration_fingerprint: str | None
    max_peak_bytes: int
    evidence: dict


@dataclass(frozen=True)
class AssessedArtifact:
    artifact: ExportVerifiedArtifact
    local_surrogate: float
    whole_model_quality: float
    selection: object
    evaluation_context: str


def prepare_quantization_catalogue(model, alternatives, *, calibration=None,
        max_peak_bytes=1024**3,max_options=256,max_catalogue_bytes=128*1024**2,ledger=None):
    """Build each finite option; keep only metadata, not all candidate models."""
    started = time.perf_counter()
    if type(alternatives) is not dict or not alternatives or sum(len(v) for v in alternatives.values()) > max_options:
        return StructuralInvalid((Violation("InvalidCatalogue","alternatives","Finite bounded path-to-tuple alternatives required"),))
    entries,tables,target_names,retained_bytes = [],[],set(),0
    try:
        inventory = capture_inventory(model)
        source = source_identity(model)
        paths = module_paths(model)
        if calibration is not None:
            from .observation import observe_operator_rows
            observed_rows = calibration if type(calibration) is dict else observe_operator_rows(model,calibration,tuple(alternatives),max_peak_bytes)
        else:
            observed_rows = {}
        scopes = set()
        for path,choices in sorted(alternatives.items()):
            if type(choices) is not tuple or not choices:
                return StructuralInvalid((Violation("InvalidCatalogue",path,"Nonempty immutable choices required"),))
            options = []
            aliases = None
            for spec in choices:
                if ledger is not None:ledger.checkpoint("catalogue_option")
                recipe = OperatorRecipe(path,spec)
                request = QuantizationGraphRequest((recipe,),max_peak_bytes=max_peak_bytes)
                plan = check_quantization_graph(model,request,calibration=calibration)
                if isinstance(plan,ValidationFailure):return StructuralInvalid(plan.violations)
                aliases = plan.aliases[0]
                if aliases[0] in scopes:
                    return StructuralInvalid((Violation("AliasOverlap",path,"Each closed alias region occurs once"),))
                result = compress_quantization_graph(model,request,calibration=calibration,ledger=ledger)
                if type(result) is not ExportVerifiedArtifact:return result
                layer = module_paths(result.model)[path]
                option_id = fingerprint(spec)
                local_inventory = capture_inventory(layer,prefix=aliases[0]+":"+option_id)
                cost = inventory_cost(local_inventory,"bytes")
                retained_bytes += cost
                if retained_bytes > max_catalogue_bytes:
                    return ResourceFailure("CatalogueBudget","Cumulative built option bytes exceed the explicit catalogue limit")
                source_layer = paths[path]
                if calibration is not None:
                    x = observed_rows[path]
                    with torch.no_grad():
                        score = float((source_layer(x)-layer(x)).double().square().sum())
                else:
                    if spec.method is Method.DENSE:w = layer.weight.detach().double()
                    elif hasattr(layer,"base"):w = layer.base.weight.decode()+layer.correction.left.decode()@layer.correction.right.decode()
                    elif hasattr(layer,"left"):
                        w = (layer.left.weight.detach().double()@layer.right.weight.detach().double() if spec.method is Method.SVD
                             else layer.left.decode()@layer.right.decode())
                    else:w = layer.weight.decode()
                    score = float((source_layer.weight.detach().double()-w).square().sum())
                storages = tuple(StorageCost(s.storage_id,s.nbytes,1,"physical_bytes") for s in local_inventory)
                ranks = () if spec.rank is None else (spec.rank,)
                options.append(CandidateOption(option_id,ranks,score,storages,"unranked" if not ranks else "ranked"))
                entries.append(CatalogueEntry(path,aliases,option_id,recipe,cost,score))
            scopes.add(aliases[0])
            for alias in aliases:
                target_names.update(view.path for storage in inventory for view in storage.views if view.path.startswith(alias+".") or alias == "")
            tables.append(CandidateTable(aliases[0] or "model",tuple(options),aliases))
        fixed = tuple(StorageCost(storage.storage_id,storage.nbytes,1,"physical_bytes") for storage in inventory
            if not any(view.path in target_names for view in storage.views))
        calibration_ref = calibration_identity(calibration)
        return QuantizationCatalogue(tuple(tables),tuple(entries),fixed,source,
            calibration_ref.fingerprint if calibration_ref else None,max_peak_bytes,
            {"built_options":len(entries),"cumulative_option_bytes":retained_bytes,
             "build_seconds":time.perf_counter()-started,"objective":"additive_frozen_snapshot_surrogate",
             "local_units":"output_squared_error" if calibration is not None else "weight_frobenius_squared"})
    except (ValueError,TypeError,KeyError,RuntimeError) as error:
        from .resources import StageBudgetExceeded
        if isinstance(error,StageBudgetExceeded):return error.failure
        return NumericalFailure(type(error).__name__,str(error))


def allocate_quantization_catalogue(catalogue,budget,*,max_shared_combinations=100000):
    try:
        return allocate(catalogue.tables,budget,objective=ScoreObjective("local_quantization_error"),
            fixed_storages=catalogue.fixed_storages,unit="bytes",max_shared_combinations=max_shared_combinations)
    except ValueError as error:
        return ResourceFailure("AllocationLimit",str(error))


def assess_quantization_selection(model,catalogue,selection,*,calibration=None,evaluate,evaluation_context,ledger=None):
    if isinstance(selection,BudgetInfeasible):
        return ResourceFailure("StorageBudget",selection.reason)
    if type(evaluation_context) is not str or not evaluation_context:
        return StructuralInvalid((Violation("EvaluationContext","evaluation_context","Explicit heldout split/protocol identity required"),))
    if source_identity(model) != catalogue.source_fingerprint:
        return StructuralInvalid((Violation("StaleCatalogue","model","Source inventory changed"),))
    ref = calibration_identity(calibration)
    if (ref.fingerprint if ref else None) != catalogue.calibration_fingerprint:
        return StructuralInvalid((Violation("StaleCalibration","calibration","Calibration rows/order changed"),))
    recipes = []
    for item in selection.selected:
        matches = [entry for entry in catalogue.entries if (entry.aliases[0] or "model") == item.table_id and entry.option_id == item.option.option_id]
        if len(matches) != 1:
            return StructuralInvalid((Violation("UnknownSelection","selection","Selection does not belong to this catalogue"),))
        recipes.append(matches[0].recipe)
    artifact = compress_quantization_graph(model,QuantizationGraphRequest(tuple(recipes),
        max_storage_bytes=selection.total_cost,max_peak_bytes=catalogue.max_peak_bytes),calibration=calibration,ledger=ledger)
    if type(artifact) is not ExportVerifiedArtifact:return artifact
    if artifact.storage_bytes != selection.total_cost:
        return NumericalFailure("AllocationAccounting","Loaded full model differs from allocator physical bytes")
    try:
        if ledger is not None:ledger.checkpoint("whole_model_assessment")
        quality = evaluate(artifact.model)
        if type(quality) not in (int,float) or not math.isfinite(quality):
            return NumericalFailure("InvalidQuality","Finite whole-model heldout quality required")
        return AssessedArtifact(artifact,selection.objective_score,float(quality),selection,evaluation_context)
    except Exception as error:
        from .resources import StageBudgetExceeded
        if isinstance(error,StageBudgetExceeded):return error.failure
        return CallbackFailure(type(error).__name__,str(error))
