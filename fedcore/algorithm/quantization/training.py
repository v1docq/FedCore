"""Stage adapter over BaseQuantizer and existing factor recovery ownership."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import math
import torch
from torch import nn
from torch.ao.quantization import QConfigMapping,get_default_qat_qconfig,get_default_qconfig

from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid,NumericalFailure,ResourceFailure
from fedcore.algorithm.low_rank.plans import Violation
from fedcore.algorithm.low_rank.candidate_storage import capture_inventory,inventory_cost
from fedcore.algorithm.low_rank.topology import module_paths,replace_modules_atomically
from .backend import RegisteredBackendGraph,RegisteredBackendLinear
from .resources import ResourceLedger,StageBudgetExceeded
from .quantizers import BaseQuantizer,QuantizationError
from .portable_layers import PortableQuantizedFactorLinear


@dataclass(frozen=True)
class BackendStageSpec:
    mode: str
    regions: tuple[str,...]
    backend: str = 'fbgemm'
    training_steps: int = 2
    learning_rate: float = .01
    calibration_id: str = 'calibration'
    training_id: str = 'training'
    selection_id: str = 'selection'
    final_test_id: str = 'test'
    criterion: str = 'mse'
    seed: int = 0


def validate_backend_spec(spec):
    if (type(spec) is not BackendStageSpec or spec.mode not in ('static','qat') or spec.criterion not in ('mse','cross_entropy')
            or type(spec.regions) is not tuple or not spec.regions or any(type(path) is not str for path in spec.regions)
            or len(set(spec.regions))!=len(spec.regions) or type(spec.training_steps) is not int or spec.training_steps<1
            or type(spec.learning_rate) not in (int,float) or not math.isfinite(spec.learning_rate) or spec.learning_rate<=0
            or any(type(value) is not str or not value for value in (spec.calibration_id,spec.training_id,spec.selection_id,spec.final_test_id))
            or len({spec.calibration_id,spec.training_id,spec.selection_id,spec.final_test_id})!=4
            or spec.backend not in torch.backends.quantized.supported_engines or spec.backend=='none'
            or type(spec.seed) is not int or not 0<=spec.seed<2**63):
        return (Violation('InvalidTrainingPlan','stage','Explicit criterion, disjoint data roles, supported backend and bounded regional protocol required'),)
    return ()


@dataclass(frozen=True)
class ConvertedInference:
    model: nn.Module
    checkpoint_bytes: bytes
    storage_bytes: int
    evidence: dict


def execute_backend_stage(model,spec,*,calibration_batches,training_batches=(),ledger=None):
    violations=validate_backend_spec(spec)
    if violations:return StructuralInvalid(violations)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.seed)
        return _execute_backend_stage(model,spec,calibration_batches=calibration_batches,training_batches=training_batches,ledger=ledger)


def _execute_backend_stage(model,spec,*,calibration_batches,training_batches=(),ledger=None):
    """One regional prepare/train/convert, followed by inspectable storage export."""
    ledger = ledger or ResourceLedger()
    violations=validate_backend_spec(spec)
    if violations:return StructuralInvalid(violations)
    paths = module_paths(model)
    if any(type(layer) in (RegisteredBackendGraph,RegisteredBackendLinear,PortableQuantizedFactorLinear) for layer in paths.values()):
        return StructuralInvalid((Violation('ConvertedInference','model','QAT requires an explicit float trainable source'),))
    if any(path not in paths or type(paths[path]) is not nn.Linear for path in spec.regions):
        return StructuralInvalid((Violation('UnsupportedRegion','regions','Regions must identify float Linear boundaries'),))
    try:
        ledger.checkpoint('backend_prepare')
        qconfig = get_default_qat_qconfig(spec.backend) if spec.mode=='qat' else get_default_qconfig(spec.backend)
        mapping = QConfigMapping().set_global(None)
        for path in spec.regions:mapping.set_module_name(path,qconfig)
        quantizer = BaseQuantizer({'quant_type':spec.mode,'backend':spec.backend,'qconfig_mapping':mapping,'stage_ledger':ledger,
            'qat_params':{'epochs':spec.training_steps,'max_steps':spec.training_steps,'lr':spec.learning_rate,'optimizer':'sgd','criterion':spec.criterion}})
        data = SimpleNamespace(model=model,calibration_dataloader=tuple(calibration_batches),train_dataloader=tuple(training_batches))
        if not data.calibration_dataloader:return StructuralInvalid((Violation('MissingCalibration','data','Nonempty calibration batches required'),))
        converted = quantizer.fit(data)
        ledger.checkpoint('backend_convert')
        registered = RegisteredBackendGraph.from_converted(converted,spec.backend).eval()
        x = data.calibration_dataloader[0][0] if isinstance(data.calibration_dataloader[0],(tuple,list)) else data.calibration_dataloader[0]
        with torch.no_grad():
            expected,actual=converted(x),registered(x)
            if not bool(torch.isfinite(expected).all()) or not bool(torch.isfinite(actual).all()):
                return NumericalFailure('BackendOutputOverflow','Converted whole-model output must be finite')
            if not torch.equal(expected,actual):
                return NumericalFailure('BackendStorageAdapter','Registered integer backend differs from converted FX graph')
        from fedcore.tools.registry.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(None)
        checkpoint = manager.serialize_to_bytes(registered)
        loaded = manager.deserialize_from_bytes(checkpoint)
        with torch.no_grad():
            if not torch.equal(loaded(x),converted(x)):
                return NumericalFailure('BackendRoundTrip','Converted output changed after checkpoint')
        cost = inventory_cost(capture_inventory(loaded),'bytes')
        ledger.checkpoint('backend_export_verified')
        return ConvertedInference(loaded,checkpoint,cost,{'mode':spec.mode,'regions':spec.regions,
            'training':quantizer.training_evidence,'data_roles':{'calibration':spec.calibration_id,'training':spec.training_id,
            'selection':spec.selection_id,'final_test':spec.final_test_id},'resources':ledger.evidence(),
            'storage_adapter':'registered_codes_qparams_bias_v1','storage_bytes':cost,'serialized_bytes':len(checkpoint),
            'inference_workspace':'explicit backend prepack temporary; no opaque persistent packed object',
            'backend_reference_max_error':0.})
    except StageBudgetExceeded as error:return error.failure
    except QuantizationError as error:
        cause = error.__cause__
        return cause.failure if isinstance(cause,StageBudgetExceeded) else NumericalFailure('BaseQuantizer',str(error))
    except (ValueError,TypeError,RuntimeError,KeyError) as error:return NumericalFailure(type(error).__name__,str(error))


def recover_quantized_factors(model,paths,*,training_batches,steps=2,learning_rate=.01,adapter_rank=1,ledger=None,seed=0):
    if type(seed) is not int or not 0<=seed<2**63:return StructuralInvalid((Violation('InvalidSeed','recovery','Bounded integer seed required'),))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _recover_quantized_factors(model,paths,training_batches=training_batches,steps=steps,learning_rate=learning_rate,
            adapter_rank=adapter_rank,ledger=ledger,seed=seed)


def _recover_quantized_factors(model,paths,*,training_batches,steps=2,learning_rate=.01,adapter_rank=1,ledger=None,seed=0):
    """Explicit decode -> existing LoRA recovery -> encode ready factors, no SVD."""
    from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
    from fedcore.algorithm.low_rank.structured_profiles import FactorPair
    from fedcore.algorithm.low_rank.factor_recovery import prepare_factor_lora_stage,finish_factor_lora_stage
    ledger = ledger or ResourceLedger()
    if type(steps) is not int or steps<1:
        return StructuralInvalid((Violation('InvalidRecovery','steps','Positive bounded update count required'),))
    batches=tuple(training_batches)
    if not batches:return StructuralInvalid((Violation('MissingTraining','data','Recovery requires training batches'),))
    train_dtype=batches[0][0].dtype
    if train_dtype not in (torch.float32,torch.float64):return StructuralInvalid((Violation('TrainingDtype','data','CPU float training dtype required'),))
    source = module_paths(model);formats={};replacements={};closed={};canonical=[]
    for path in paths:
        layer = source.get(path)
        if type(layer) is not PortableQuantizedFactorLinear:
            return StructuralInvalid((Violation('UnsupportedRecovery',path,'Explicit packed factor parents required'),))
        aliases=tuple(sorted(p for p,item in source.items() if item is layer))
        key=aliases[0]
        if key in closed:continue
        canonical.append(key);closed[key]=aliases
        formats[key]=(layer.left.spec,layer.right.spec)
        decoded=FactorizedLinear.from_factors(FactorPair(layer.left.decode().to(train_dtype),layer.right.decode().to(train_dtype),{'method':'explicit_decode'}),layer.bias)
        for alias in aliases:replacements[alias]=decoded
    work=replace_modules_atomically(deepcopy(model),replacements)
    paths=tuple(canonical)
    evidence=[]
    try:
        for factor in ('left','right'):
            ledger.checkpoint('factor_recovery_prepare')
            stage=prepare_factor_lora_stage(work,tuple(paths),factor=factor,rank=adapter_rank)
            optimizer=torch.optim.SGD(stage.trainable_parameters(),lr=learning_rate)
            stage.validate_optimizer(optimizer)
            before={name:p.detach().clone() for name,p in stage.model.named_parameters() if p.requires_grad}
            for index in range(steps):
                ledger.checkpoint('factor_recovery_train',training_steps=1)
                x,y=batches[index%len(batches)]
                optimizer.zero_grad();loss=(stage.model(x)-y).square().mean()
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite recovery loss')
                loss.backward();optimizer.step()
                ledger.complete_training_step('factor_recovery_train')
            changed=tuple(name for name,p in stage.model.named_parameters() if name in before and not torch.equal(before[name],p.detach()))
            work,details=finish_factor_lora_stage(stage)
            details.update(training_steps=steps,changed_parameters=changed)
            evidence.append(details)
            del optimizer
        replacements={}
        for path in paths:
            layer=module_paths(work)[path];left_spec,right_spec=formats[path]
            packed=PortableQuantizedFactorLinear.from_factors(layer.left.weight,layer.right.weight,layer.left.bias,
                left_spec=left_spec,right_spec=right_spec,generator=torch.Generator().manual_seed(seed))
            for alias in closed[path]:replacements[alias]=packed
        work=replace_modules_atomically(work,replacements).eval()
        from fedcore.tools.registry.checkpoint_manager import CheckpointManager
        manager=CheckpointManager(None);checkpoint=manager.serialize_to_bytes(work);loaded=manager.deserialize_from_bytes(checkpoint)
        with torch.no_grad():
            expected,actual=work(batches[0][0]),loaded(batches[0][0])
            if not bool(torch.isfinite(expected).all()) or not bool(torch.isfinite(actual).all()) or not torch.equal(expected,actual):
                return NumericalFailure('RecoveryRoundTrip','Recovered whole-model outputs must be finite and equal after loading')
        ledger.checkpoint('factor_recovery_export')
        return ConvertedInference(loaded,checkpoint,inventory_cost(capture_inventory(loaded),'bytes'),
            {'stages':evidence,'resources':ledger.evidence(),'transition':'explicit_decode_train_encode_ready_factors'})
    except StageBudgetExceeded as error:return error.failure
    except (ValueError,TypeError,RuntimeError,KeyError) as error:return NumericalFailure(type(error).__name__,str(error))
