"""Executable bounded DAG, bridged to the existing compression ports/artifacts."""
from dataclasses import dataclass,replace
import math
import torch
from torch import nn
from fedcore.algorithm.low_rank.compression_graph import ArtifactRef,ArtifactKind,CompressionNode,Port,Stage
from fedcore.algorithm.low_rank.structured_profiles import FactorPair,balanced_factors,linear_least_squares_fit
from fedcore.algorithm.low_rank.structured_layers import FactorizedLinear
from fedcore.algorithm.low_rank.approximation import prepare_weighted_metric,solve_weighted_prepared
from fedcore.algorithm.low_rank.plans import MetricPolicy
from .graph_specs import canonical_graph,recipe_graph,DEFAULT_MAX_PEAK_BYTES
from .graph_specs import fingerprint
from .tensor_formats import PackedTensor
from .portable_layers import PortableQuantizedLinear,QuantizedSumLinear
from .matrix_transforms import _factor_layer,input_second_moment,scalar_ldlq


_KINDS={
    'ExtractMatrix':ArtifactKind.WEIGHT_MATRIX,'PreQuantizeTensor':ArtifactKind.QUANTIZED_TENSOR,
    'DecodeTensor':ArtifactKind.WEIGHT_MATRIX,'ExtractFactors':ArtifactKind.REAL_FACTORS,
    'SVDTruncate':ArtifactKind.REAL_FACTORS,'RandomProject':ArtifactKind.WEIGHT_MATRIX,'SVDLeft':ArtifactKind.WEIGHT_MATRIX,
    'QuantizeLeft':ArtifactKind.QUANTIZED_TENSOR,'QuantizeRight':ArtifactKind.QUANTIZED_TENSOR,
    'QuantizeLeftFactor':ArtifactKind.QUANTIZED_TENSOR,'QuantizeRightFactor':ArtifactKind.QUANTIZED_TENSOR,
    'QuantizeWeight':ArtifactKind.QUANTIZED_TENSOR,'LeastSquaresFit':ArtifactKind.WEIGHT_MATRIX,
    'ObserveInput':ArtifactKind.CALIBRATION,'BuildMetric':ArtifactKind.FULL_METRIC,
    'InitializeSum':ArtifactKind.QUANTIZED_SUM,'LDLQ':ArtifactKind.QUANTIZED_SUM,
    'WeightedFactorFit':ArtifactKind.QUANTIZED_SUM,'RegressRight':ArtifactKind.QUANTIZED_SUM,
    'RegressLeft':ArtifactKind.QUANTIZED_SUM,'SelectBest':ArtifactKind.QUANTIZED_SUM,
    'BuildChannelGroups':ArtifactKind.TENDER_GROUPS,
    **{name:ArtifactKind.REPLACEMENT for name in ('BuildDenseOperator','BuildQuantizedOperator','BuildRealFactorOperator','BuildFactorOperator','BuildSum','BuildTenderOperator','EvaluateWeightError')}}


def compression_nodes(graph,*,scope,model_version):
    """One common typed port contract; no alternative optimizer/runtime graph."""
    graph=canonical_graph(graph)
    refs={node.name:ArtifactRef(node.name,_KINDS[node.operation],model_version,scope) for node in graph.nodes}
    return tuple(CompressionNode(node.name,Stage.OBSERVE if node.operation=='ObserveInput' else
        Stage.METRIC if node.operation=='BuildMetric' else Stage.BUILD if _KINDS[node.operation] is ArtifactKind.REPLACEMENT else Stage.SOLVE,
        scope,node.inputs,(node.name,),tuple(Port(f'input{i}',refs[ref].kind,ref) for i,ref in enumerate(node.inputs)),(refs[node.name],)) for node in graph.nodes)


@dataclass(frozen=True)
class SumState:
    q: PackedTensor
    left: PackedTensor
    right: PackedTensor
    best: tuple
    best_score: float
    initial_score: float
    history: tuple[float,...]
    feedback: tuple[dict,...] = ()


def execute_process(layer,spec,graph,rows,*,model_version,scope,ledger=None,cache=None,calibration_fingerprint='none',max_peak_bytes=DEFAULT_MAX_PEAK_BYTES):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.seed)
        return _execute_process(layer,spec,graph,rows,model_version=model_version,scope=scope,ledger=ledger,cache=cache,
            calibration_fingerprint=calibration_fingerprint,max_peak_bytes=max_peak_bytes)


def _execute_process(layer,spec,graph,rows,*,model_version,scope,ledger=None,cache=None,calibration_fingerprint='none',max_peak_bytes=DEFAULT_MAX_PEAK_BYTES):
    """Read each node's actual dependencies; a changed prefix changes its tail."""
    graph=canonical_graph(graph)
    if graph!=recipe_graph(spec):raise ValueError('Process graph differs from its checked structural specification')
    nodes=compression_nodes(graph,scope=scope,model_version=model_version)
    generator=torch.Generator().manual_seed(spec.seed)
    bias=layer.left.bias if type(layer) is FactorizedLinear else layer.bias
    bias=None if bias is None else bias.detach()
    values={};trace=[];width=layer.in_features;last_state=None;last_metric=None;weight_errors={}
    users={node.name:sum(ref==node.name for child in graph.nodes for ref in child.inputs) for node in graph.nodes}
    def pack(x,format):
        from .cache import CacheIdentity
        import hashlib
        identity=hashlib.sha256(x.detach().contiguous().numpy().tobytes()+generator.get_state().numpy().tobytes()).hexdigest()
        key=CacheIdentity('quantized_solution',model_version,model_version,calibration_fingerprint,
            ':'.join(scope)+':original_operator_input_output','QuantizeTensor',fingerprint({'input':identity,'shape':tuple(x.shape),'format':format}))
        hit=cache.get(key) if cache is not None else None
        if hit is not None:
            result=PackedTensor(tuple(x.shape),format);result.packed,result.scale=hit[:2];generator.set_state(hit[2]);return result
        result=PackedTensor.from_tensor(x.contiguous(),format,generator=generator)
        if cache is not None:cache.put(key,(result.packed,result.scale,generator.get_state()),dependencies=(model_version,calibration_fingerprint))
        return result
    def fit_factors(w,metric):
        pair=balanced_factors(solve_weighted_prepared(w,metric,spec.rank))
        return pack(pair.left,spec.left),pack(pair.right,spec.right)
    def score(w,metric,q,left,right):return float(((w-q.decode()-left.decode()@right.decode())@metric.root).square().sum())
    for node,common in zip(graph.nodes,nodes):
        if ledger is not None:ledger.checkpoint('process:'+node.operation)
        args=[values[ref] for ref in node.inputs];op=node.operation
        from fedcore.algorithm.low_rank.candidate_storage import storage_guard
        storage_guard(0,max_peak_bytes)
        if op=='ExtractMatrix':value=layer.weight.detach().double().clone()
        elif op=='ExtractFactors':value=FactorPair(layer.left.weight.detach().double().clone(),layer.right.weight.detach().double().clone(),{'source':'ready_factors_no_svd'})
        elif op=='PreQuantizeTensor':value=pack(args[0],spec.prequantize)
        elif op=='DecodeTensor':value=args[0].decode()
        elif op=='BuildDenseOperator':
            value=nn.Linear(layer.in_features,layer.out_features,bias is not None,dtype=layer.weight.dtype)
            with torch.no_grad():
                value.weight.copy_(args[0]);
                if bias is not None:value.bias.copy_(bias)
        elif op in ('SVDTruncate','SVDLeft'):
            u,s,vh=torch.linalg.svd(args[0],full_matrices=False)
            value=(FactorPair(u[:,:spec.rank]*s[:spec.rank].sqrt(),s[:spec.rank].sqrt()[:,None]*vh[:spec.rank],{'method':'dsvd'})
                   if op=='SVDTruncate' else u[:,:spec.rank]*s[:spec.rank])
        elif op=='RandomProject':value=args[0]@(torch.randn(width,spec.rank,dtype=torch.float64,generator=generator)/math.sqrt(spec.rank))
        elif op=='QuantizeLeftFactor':value=pack(args[0].left,spec.left)
        elif op=='QuantizeRightFactor':value=pack(args[0].right,spec.right)
        elif op=='QuantizeLeft':value=pack(args[0],spec.left)
        elif op=='QuantizeRight':value=pack(args[0],spec.right)
        elif op=='QuantizeWeight':value=pack(args[0],spec.weight)
        elif op=='LeastSquaresFit':value=linear_least_squares_fit(args[0],args[1],rcond=spec.rcond,ridge=spec.ridge).coefficients.T.contiguous()
        elif op=='BuildFactorOperator':value=_factor_layer(args[0],args[1],bias)
        elif op=='BuildRealFactorOperator':value=FactorizedLinear.from_factors(FactorPair(args[0].left.to(layer.weight),args[0].right.to(layer.weight),{}),bias)
        elif op=='BuildQuantizedOperator':
            value=PortableQuantizedLinear(layer.in_features,layer.out_features,bias is not None,spec=spec.weight);value.weight=args[0]
            if bias is not None:value.bias=bias.double().clone()
        elif op=='ObserveInput':
            if rows is None:raise ValueError('Explicit checked calibration rows required')
            value=rows.detach().double().clone()
        elif op=='BuildMetric':
            from .cache import CacheIdentity
            from fedcore.algorithm.low_rank.approximation import PreparedWeightedMetric
            policy=MetricPolicy(spec.ridge,spec.rcond)
            key=CacheIdentity('weighted_metric',model_version,model_version,calibration_fingerprint,
                ':'.join(scope)+':original_operator_input_output','BuildMetric',fingerprint(policy))
            hit=cache.get(key) if cache is not None else None
            if hit is not None:value=PreparedWeightedMetric(*hit[:5],float(hit[5]),policy)
            else:
                value=prepare_weighted_metric(input_second_moment(args[0],width),policy)
                if cache is not None:cache.put(key,(value.root,value.inverse_root,value.eigenvalues,value.vectors,value.support,torch.tensor(value.cutoff,dtype=torch.float64)),dependencies=(model_version,calibration_fingerprint))
        elif op=='InitializeSum':
            w,metric=args;q=pack(w,spec.weight);left,right=fit_factors(w-q.decode(),metric);error=score(w,metric,q,left,right)
            value=SumState(q,left,right,(q,left,right),error,error,(error,))
        elif op=='LDLQ':
            state,w,metric=args
            q,details=scalar_ldlq(w-state.left.decode()@state.right.decode(),metric.root@metric.root.T,spec.weight,generator=generator,rcond=spec.rcond)
            value=replace(state,q=q,feedback=state.feedback+(details,))
        elif op=='WeightedFactorFit':
            state,w,metric=args;left,right=fit_factors(w-state.q.decode(),metric);value=replace(state,left=left,right=right)
        elif op=='RegressRight':
            state,w,metric=args;fit=linear_least_squares_fit(state.left.decode(),(w-state.q.decode())@metric.root,rcond=spec.rcond)
            right=pack(fit.coefficients.T@metric.inverse_root,spec.right);value=replace(state,right=right)
        elif op=='RegressLeft':
            state,w,metric=args;fit=linear_least_squares_fit((state.right.decode()@metric.root).T,((w-state.q.decode())@metric.root).T,rcond=spec.rcond)
            value=replace(state,left=pack(fit.coefficients,spec.left))
        elif op=='SelectBest':
            state,w,metric=args;error=score(w,metric,state.q,state.left,state.right)
            value=replace(state,history=state.history+(error,),best=(state.q,state.left,state.right) if error<state.best_score else state.best,
                          best_score=min(error,state.best_score))
        elif op=='BuildSum':
            state=args[0];q,left,right=state.best
            base=PortableQuantizedLinear(layer.in_features,layer.out_features,bias is not None,spec=spec.weight);base.weight=q
            if bias is not None:base.bias=bias.double().clone()
            value=QuantizedSumLinear(base,_factor_layer(left,right,None))
        elif op=='BuildChannelGroups':
            from .tender import TenderLinear
            value=TenderLinear.calibrate(args[0],bias,args[1],weight_bits=spec.weight.bits,activation_bits=spec.activation_bits,accumulator_bits=spec.accumulator_bits)
        elif op=='BuildTenderOperator':value=args[0]
        elif op=='EvaluateWeightError':
            value,w=args;k=spec.comparison_rank or spec.rank
            u,s,vh=torch.linalg.svd(w,full_matrices=False)
            weight_errors={'weight_frobenius_squared':float((w-value.left.decode()@value.right.decode()).square().sum()),
                           'dsvd_reference_squared':float((w-(u[:,:k]*s[:k])@vh[:k]).square().sum())}
        else:raise ValueError('No interpreter capability for process operation')
        values[node.name]=value
        if type(value) is SumState:last_state=value
        if op=='BuildMetric':last_metric=value
        trace.append({'node':node.name,'operation':op,'inputs':node.inputs,'output_kind':common.outputs[0].kind.value,
                      'artifact':common.outputs[0].key,'coordinates':common.outputs[0].coordinates})
        for ref in node.inputs:
            users[ref]-=1
            if users[ref]==0 and ref!=graph.output:del values[ref]
    details={'method':spec.method.value,'process_trace':trace,'process_fingerprint':__import__('fedcore.algorithm.quantization.graph_specs',fromlist=['fingerprint']).fingerprint(graph),
             'source_version':model_version,'calibration_coordinates':'original_operator_input_output'}
    if spec.method.value.startswith('lplr'):
        details.update(factor_width_m=spec.rank,comparison_rank_k=spec.comparison_rank or spec.rank,regression_design='decoded_quantized_left',**weight_errors)
    if last_state is not None:
        state=last_state;metric=last_metric
        details.update(initial_error=state.initial_score,best_error=state.best_score,candidate_errors=state.history,
                       ldlq=state.feedback,metric_support=int(metric.support.sum()),bias_count=int(bias is not None))
    return values[graph.output],details
