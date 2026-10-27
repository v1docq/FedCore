"""One public workflow joining compression, recovery and regional conversion."""
from dataclasses import dataclass
import math
import torch
from .graph_specs import QuantizationGraphRequest,Method,fingerprint
from .graph_execution import check_quantization_graph,compress_quantization_graph,ExportVerifiedArtifact,calibration_identity,source_identity
from .training import BackendStageSpec,execute_backend_stage,recover_quantized_factors,ConvertedInference,validate_backend_spec
from .resources import ResourceLedger,StageBudgetExceeded
from fedcore.algorithm.low_rank.plans import ValidationFailure,Violation
from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid


@dataclass(frozen=True)
class RecoveryStageSpec:
    paths: tuple[str,...]
    steps: int = 2
    learning_rate: float = .01
    adapter_rank: int = 1
    training_id: str = 'recovery_training'
    seed: int = 0


def validate_recovery_spec(spec):
    if (type(spec) is not RecoveryStageSpec or type(spec.paths) is not tuple or not spec.paths
            or any(type(path) is not str for path in spec.paths) or len(set(spec.paths)) != len(spec.paths)
            or type(spec.steps) is not int or spec.steps < 1
            or type(spec.adapter_rank) is not int or spec.adapter_rank < 1
            or type(spec.learning_rate) not in (int, float) or not math.isfinite(spec.learning_rate) or spec.learning_rate <= 0
            or type(spec.training_id) is not str or not spec.training_id
            or type(spec.seed) is not int or not 0 <= spec.seed < 2**63):
        return (Violation('InvalidRecoveryPlan', 'stage', 'Unique paths, positive finite training parameters and an explicit data identity required'),)
    return ()


def training_stage_violations(stage, methods):
    """Validate the same global transition for programs and grammar operators."""
    if stage is None:
        return ()
    if type(stage) is RecoveryStageSpec:
        errors = list(validate_recovery_spec(stage))
        if errors:
            return tuple(errors)
        if any(methods.get(path) not in (Method.DSVD, Method.LPLR, Method.LPLR_SVD, Method.FACTORS) for path in stage.paths):
            errors.append(Violation('UnsupportedRecovery', 'program', 'Recovery paths must refer to constructed packed factor operators'))
        if Method.TENDER in methods.values():
            errors.append(Violation('UnsupportedGradientBoundary', 'program', 'Factor recovery across Tender requires a differentiable training adapter'))
        return tuple(errors)
    errors = list(validate_backend_spec(stage))
    if errors:
        return tuple(errors)
    if any(method not in (Method.DENSE, Method.SVD) for method in methods.values()):
        errors.append(Violation('UnsupportedBackendBoundary', 'program', 'Regional backend conversion requires float dense/real-factor output'))
    for region in stage.regions:
        parent = region.rsplit('.', 1)[0] if '.' in region else '' if region in ('left','right') else region
        if methods.get(region) is Method.SVD:
            errors.append(Violation('UnsupportedBackendRegion',region,'Select the explicit left/right Linear boundaries of real factors'))
            continue
        factor_boundary=region.endswith(('.left','.right')) or region in ('left','right')
        if region not in methods and not (methods.get(parent) is Method.SVD and factor_boundary):
            errors.append(Violation('UnsupportedBackendRegion', region, 'Region must identify a declared float output boundary'))
    return tuple(errors)


@dataclass(frozen=True)
class QuantizationProgram:
    compression: QuantizationGraphRequest
    backend: BackendStageSpec | None = None
    recovery: RecoveryStageSpec | None = None
    training_fingerprint: str | None = None


@dataclass(frozen=True)
class CheckedQuantizationProgram:
    program: QuantizationProgram
    compression_plan: object
    stages: tuple[str,...]
    fingerprint: str


@dataclass(frozen=True)
class ProgramArtifact:
    model: object
    checkpoint_bytes: bytes
    storage_bytes: int
    plan: CheckedQuantizationProgram
    evidence: dict


def training_identity(batches):
    if not batches:return None
    import hashlib
    digest=hashlib.sha256()
    for i,(x,y) in enumerate(batches):
        if (type(x) is not torch.Tensor or type(y) is not torch.Tensor or x.device.type!='cpu' or y.device.type!='cpu'
                or x.dtype not in (torch.float32,torch.float64) or y.dtype not in (torch.float32,torch.float64,torch.int64)
                or x.ndim<2 or y.ndim<1 or not x.numel() or not y.numel() or len(x)!=len(y)
                or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all())):raise ValueError('Finite ordered CPU training feature/target batches required')
        for role,value in (('features',x),('targets',y)):
            digest.update(str((i,role,tuple(value.shape),value.dtype)).encode());digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_quantization_program(model,program,*,calibration=None,training_batches=()):
    if type(program) is not QuantizationProgram:return ValidationFailure((Violation('InvalidProgram','program','Typed program required'),))
    plan=check_quantization_graph(model,program.compression,calibration=calibration)
    if isinstance(plan,ValidationFailure):return plan
    errors=[];stages=['checked_process_dag','apply_replacements','export_verified']
    if program.backend is not None and program.recovery is not None:
        errors.append(Violation('UnsupportedTrainingOrder','program','Backend QAT after packed-factor recovery requires a new explicit float source'))
    methods={alias:recipe.transform.method for recipe,aliases in zip(program.compression.recipes,plan.aliases) for alias in aliases}
    if program.recovery is not None:
        r=program.recovery
        errors.extend(training_stage_violations(r,methods))
        stages+=['decode_ready_factors','prepare_factor_lora','train_left','merge_left','train_right','merge_right','encode_ready_factors','export_verified']
    if program.backend is not None:
        b=program.backend
        errors.extend(training_stage_violations(b,methods))
        if type(calibration) is not torch.Tensor:
            errors.append(Violation('BackendCalibrationCoordinates','data','Regional FX conversion requires an explicit root-model input tensor'))
        stages+=['regional_prepare','train' if type(b) is BackendStageSpec and b.mode=='qat' else 'calibrate','convert','register_backend_storage','export_verified']
    if program.backend is not None or program.recovery is not None:
        try:
            actual=training_identity(training_batches)
            needs_training=program.recovery is not None or type(program.backend) is BackendStageSpec and program.backend.mode=='qat'
            if needs_training and (actual is None or actual!=program.training_fingerprint):
                errors.append(Violation('TrainingIdentity','data','Exact ordered training batches must match the checked program fingerprint'))
        except (ValueError,TypeError):errors.append(Violation('TrainingIdentity','data','Finite CPU training batches required'))
    if errors:return ValidationFailure(tuple(errors))
    return CheckedQuantizationProgram(program,plan,tuple(stages),fingerprint({'compression':plan.fingerprint,'program':program,'stages':stages}))


def execute_quantization_program(model,program,*,calibration=None,training_batches=(),ledger=None,cache=None):
    with torch.random.fork_rng(devices=[]):
        return _execute_quantization_program(model,program,calibration=calibration,training_batches=training_batches,ledger=ledger,cache=cache)


def _execute_quantization_program(model,program,*,calibration=None,training_batches=(),ledger=None,cache=None):
    checked=check_quantization_program(model,program,calibration=calibration,training_batches=training_batches)
    if isinstance(checked,ValidationFailure):return StructuralInvalid(checked.violations)
    ledger=ledger or ResourceLedger()
    with ledger.limit_rss(program.compression.max_peak_bytes):
        return _execute_checked_quantization_program(model,program,checked,calibration=calibration,
            training_batches=training_batches,ledger=ledger,cache=cache)


def _execute_checked_quantization_program(model,program,checked,*,calibration,training_batches,ledger,cache):
    try:ledger.checkpoint('program_started')
    except StageBudgetExceeded as error:return error.failure
    artifact=compress_quantization_graph(model,program.compression,calibration=calibration,ledger=ledger,cache=cache)
    if type(artifact) is not ExportVerifiedArtifact:return artifact
    current=artifact;stage_evidence=[]
    if program.recovery is not None:
        r=program.recovery
        current=recover_quantized_factors(current.model,r.paths,training_batches=training_batches,steps=r.steps,
            learning_rate=r.learning_rate,adapter_rank=r.adapter_rank,ledger=ledger,seed=r.seed)
        if type(current) is not ConvertedInference:return current
        stage_evidence.append(current.evidence)
    if program.backend is not None:
        current=execute_backend_stage(current.model,program.backend,calibration_batches=((calibration,None),),
            training_batches=training_batches,ledger=ledger)
        if type(current) is not ConvertedInference:return current
        stage_evidence.append(current.evidence)
    try:ledger.checkpoint('program_completed')
    except StageBudgetExceeded as error:return error.failure
    if program.compression.max_storage_bytes is not None and current.storage_bytes>program.compression.max_storage_bytes:
        from fedcore.algorithm.low_rank.grammar_execution import ResourceFailure
        return ResourceFailure('FinalStorageBudget','Recovery/conversion output exceeds the whole-program physical budget')
    from .graph_specs import payload
    from fedcore.tools.registry.checkpoint_manager import CheckpointManager
    current.model._fedcore_method_evidence=payload({'schema':'quantization_graph_v1','program':checked})
    manager=CheckpointManager(None);checkpoint=manager.serialize_to_bytes(current.model);loaded=manager.deserialize_from_bytes(checkpoint)
    if type(calibration) is torch.Tensor:
        from fedcore.algorithm.low_rank.grammar_execution import NumericalFailure
        with torch.no_grad():
            expected,actual=current.model(calibration),loaded(calibration)
            if not bool(torch.isfinite(expected).all()) or not bool(torch.isfinite(actual).all()) or not torch.equal(expected,actual):
                return NumericalFailure('ProgramRoundTrip','Final whole-program outputs must be finite and equal after loading')
    try:ledger.checkpoint('program_export_verified')
    except StageBudgetExceeded as error:return error.failure
    return ProgramArtifact(loaded,checkpoint,current.storage_bytes,checked,
        {'compression':artifact.evidence,'training_stages':stage_evidence,'resources':ledger.evidence(),'stages':checked.stages})


def quantize(model,request,*,calibration=None,training_batches=(),ledger=None,cache=None):
    """Unified public facade: finite compression or a checked training program."""
    if type(request) is QuantizationGraphRequest:
        return compress_quantization_graph(model,request,calibration=calibration,ledger=ledger,cache=cache)
    return execute_quantization_program(model,request,calibration=calibration,training_batches=training_batches,ledger=ledger,cache=cache)
