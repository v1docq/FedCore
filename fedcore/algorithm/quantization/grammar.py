"""Structural subgraph grammar over the existing GrammarAdapter/GOLEM engine."""
from dataclasses import dataclass,replace
import random
from fedcore.algorithm.low_rank.plans import ValidationFailure,Violation
from fedcore.algorithm.low_rank.grammar_search import GrammarAdapter,run_grammar_search,GrammarFitness,GrammarSearchConfig
from .graph_specs import (TransformSpec,OperatorRecipe,QuantizationGraphRequest,ProcessGraphSpec,
    Method,recipe_graph,canonical_graph,fingerprint,payload,DEFAULT_MAX_PEAK_BYTES)
from .tensor_formats import QuantizedTensorSpec,Rounding
from .graph_execution import check_quantization_graph,compress_quantization_graph,ExportVerifiedArtifact,source_identity,calibration_identity
from .resources import ResourceLedger,StageBudgetExceeded
from fedcore.algorithm.low_rank.compression_graph import SourceMode
from .program import (RecoveryStageSpec,QuantizationProgram,ProgramArtifact,execute_quantization_program,training_identity)
from .training import BackendStageSpec


def _failure(message):return ValidationFailure((Violation('InvalidQuantizationGrammar','genotype',message),))


def canonical_transform(spec):
    """Inactive recipe fields cannot create distinct solutions/cache identities."""
    m=spec.method
    keys={Method.DENSE:(),Method.PTQ:('weight','prequantize'),Method.SVD:('rank','prequantize'),
          Method.DSVD:('rank','left','right','prequantize'),Method.FACTORS:('left','right'),
          Method.LPLR:('rank','comparison_rank','left','right','rcond','ridge','prequantize'),
          Method.LPLR_SVD:('rank','comparison_rank','left','right','rcond','ridge','prequantize'),
          Method.CALDERA:('rank','left','right','weight','rcond','ridge','outer_iterations','inner_iterations','prequantize'),
          Method.TENDER:('activation_bits','accumulator_bits')}
    values={key:getattr(spec,key) for key in keys[m]}
    if m is Method.TENDER:values['weight']=QuantizedTensorSpec(spec.weight.bits,axis=0)
    if m is Method.LPLR or any(getattr(spec,key).rounding is Rounding.STOCHASTIC for key in ('left','right','weight') if key in keys[m]) or spec.prequantize is not None and spec.prequantize.rounding is Rounding.STOCHASTIC:
        values['seed']=spec.seed
    return TransformSpec(m,**values)


@dataclass(frozen=True)
class QuantizationDomain:
    aliases: tuple[str,...]
    choices: tuple[TransformSpec,...]


@dataclass(frozen=True)
class QuantizationSpace:
    domains: tuple[QuantizationDomain,...]
    source: str
    calibration: str | None
    evaluation_context: str
    max_storage_bytes: int | None
    max_peak_bytes: int
    prequantize_bits: tuple[int,...] = (2,4,8)
    training_choices: tuple = (None,)
    training_fingerprint: str | None = None


@dataclass(frozen=True)
class QuantizationGene:
    aliases: tuple[str,...]
    transform: TransformSpec
    graph: ProcessGraphSpec


@dataclass(frozen=True)
class QuantizationGenotype:
    genes: tuple[QuantizationGene,...]
    version: int = 1
    training: BackendStageSpec | RecoveryStageSpec | None = None


@dataclass(frozen=True)
class QuantizationWorkflow:
    request: QuantizationGraphRequest
    source: str
    calibration: str | None
    evaluation_context: str
    fingerprint: str
    genotype: QuantizationGenotype
    program: QuantizationProgram | None = None


def prepare_quantization_space(model,alternatives,*,calibration=None,evaluation_context,max_storage_bytes=None,max_peak_bytes=DEFAULT_MAX_PEAK_BYTES,prequantize_bits=(2,4,8),training_choices=(None,),training_batches=()):
    if type(alternatives) is not dict or not alternatives:return _failure('Nonempty finite alternatives required')
    domains=[]
    for path,choices in sorted(alternatives.items()):
        if type(choices) is not tuple or not choices:return _failure('Finite choice tuples required')
        checked=[];aliases=None
        for spec in choices:
            plan=check_quantization_graph(model,QuantizationGraphRequest((OperatorRecipe(path,spec),),max_storage_bytes=max_storage_bytes,max_peak_bytes=max_peak_bytes),calibration=calibration)
            if isinstance(plan,ValidationFailure):return plan
            aliases=plan.aliases[0];checked.append(canonical_transform(spec))
        domains.append(QuantizationDomain(aliases,tuple(sorted(set(checked),key=fingerprint))))
    ref=calibration_identity(calibration)
    space=QuantizationSpace(tuple(sorted(domains,key=lambda d:d.aliases)),source_identity(model),ref.fingerprint if ref else None,
        evaluation_context,max_storage_bytes,max_peak_bytes,prequantize_bits,training_choices,training_identity(training_batches))
    failure=validate_space(space)
    return space if failure is None else failure


def validate_space(space):
    if (type(space) is not QuantizationSpace or type(space.domains) is not tuple or not space.domains
            or type(space.evaluation_context) is not str or not space.evaluation_context
            or type(space.prequantize_bits) is not tuple or any(type(b) is not int or b not in (2,4,8) for b in space.prequantize_bits)
            or type(space.training_choices) is not tuple or not space.training_choices
            or any(item is not None and type(item) not in (BackendStageSpec,RecoveryStageSpec) for item in space.training_choices)):
        return _failure('Typed domains, evaluation context and structural rewrite bounds required')
    scopes=[]
    for domain in space.domains:
        if type(domain) is not QuantizationDomain or not domain.aliases or not domain.choices:return _failure('Closed nonempty domains required')
        scopes.extend(domain.aliases)
    if len(set(scopes))!=len(scopes):return _failure('Overlapping alias regions')
    return None


def _compatible_training(training,genes):
    if training is None:return True
    methods={g.aliases[0]:g.transform.method for g in genes}
    if type(training) is RecoveryStageSpec:
        return all(methods.get(path) in (Method.DSVD,Method.LPLR,Method.LPLR_SVD,Method.FACTORS) for path in training.paths)
    return all(method in (Method.DENSE,Method.SVD) for method in methods.values())


def normalize_genotype(genotype,space):
    failure=validate_space(space)
    if failure is not None:return failure
    if (type(genotype) is not QuantizationGenotype or type(genotype.version) is not int or genotype.version!=1
            or type(genotype.genes) is not tuple or len(genotype.genes)!=len(space.domains)):
        return _failure('One immutable closed gene per region required')
    domains={d.aliases:d for d in space.domains};genes=[];used=set()
    try:
        for gene in genotype.genes:
            if type(gene) is not QuantizationGene or gene.aliases not in domains or gene.aliases in used:return _failure('Unknown or repeated scope')
            used.add(gene.aliases);spec=canonical_transform(gene.transform)
            if replace(spec,prequantize=None) not in tuple(replace(choice,prequantize=None) for choice in domains[gene.aliases].choices):
                return _failure('Unapproved numerical/representation parameters')
            if spec.prequantize is not None:
                if (spec.method in (Method.DENSE,Method.FACTORS,Method.TENDER) or spec.prequantize!=QuantizedTensorSpec(spec.prequantize.bits)
                        or spec.prequantize.bits not in space.prequantize_bits):return _failure('Unsupported compatible-prefix rewrite')
            graph=canonical_graph(gene.graph)
            if graph!=recipe_graph(spec):return _failure('Process ports/order differ from the approved subgraph')
            genes.append(QuantizationGene(gene.aliases,spec,graph))
        if genotype.training not in space.training_choices or not _compatible_training(genotype.training,genes):return _failure('Incompatible closed training stage')
        return QuantizationGenotype(tuple(sorted(genes,key=lambda g:g.aliases)),training=genotype.training)
    except (ValueError,TypeError,AttributeError,KeyError):return _failure('Malformed graph metadata')


def generate_genotype(space,*,seed=0):
    rng=random.Random(seed)
    genes=tuple(QuantizationGene(d.aliases,spec,recipe_graph(spec)) for d in space.domains for spec in (rng.choice(d.choices),))
    compatible=[stage for stage in space.training_choices if _compatible_training(stage,genes)]
    return QuantizationGenotype(genes,training=rng.choice(compatible)) if compatible else _failure('No compatible global stage')


def mutate_genotype(genotype,space,*,kind,seed=0):
    parent=normalize_genotype(genotype,space)
    if isinstance(parent,ValidationFailure):return parent
    rng=random.Random(seed);genes=list(parent.genes);index=rng.randrange(len(genes));gene=genes[index]
    domain=next(d for d in space.domains if d.aliases==gene.aliases)
    # SCOPE mutates a proper prefix subgraph; CHOICE changes compatible local
    # numerical parameters or the method. Both remain inside the common engine.
    if getattr(kind,'value',kind)=='scope' and gene.transform.method not in (Method.DENSE,Method.FACTORS,Method.TENDER) and space.prequantize_bits:
        pre=None if gene.transform.prequantize is not None else QuantizedTensorSpec(rng.choice(space.prequantize_bits))
        spec=replace(gene.transform,prequantize=pre)
    else:
        alternatives=[s for s in domain.choices if s!=replace(gene.transform,prequantize=None)]
        spec=rng.choice(alternatives or list(domain.choices))
    genes[index]=QuantizationGene(gene.aliases,spec,recipe_graph(spec))
    compatible=[stage for stage in space.training_choices if _compatible_training(stage,genes)]
    if not compatible:return _failure('No compatible global stage')
    training=rng.choice(compatible)
    return normalize_genotype(QuantizationGenotype(tuple(genes),training=training),space)


def crossover_genotypes(first,second,space,*,seed=0):
    left,right=normalize_genotype(first,space),normalize_genotype(second,space)
    if isinstance(left,ValidationFailure):return left
    if isinstance(right,ValidationFailure):return right
    selected=random.Random(seed).randrange(len(left.genes));a,b=list(left.genes),list(right.genes)
    a[selected],b[selected]=b[selected],a[selected]
    def child(genes,training):
        compatible=[stage for stage in space.training_choices if _compatible_training(stage,genes)]
        return QuantizationGenotype(tuple(genes),training=training if training in compatible else compatible[0])
    return child(a,left.training),child(b,right.training)


def genotype_payload(genotype):
    return {'version':genotype.version,'genes':[{'aliases':list(g.aliases),'transform':payload(g.transform),'graph':payload(g.graph)} for g in genotype.genes],
            'training':payload(genotype.training)}


def compile_genotype(genotype,space):
    normalized=normalize_genotype(genotype,space)
    if isinstance(normalized,ValidationFailure):return normalized
    request=QuantizationGraphRequest(tuple(OperatorRecipe(g.aliases[0],g.transform) for g in normalized.genes),
        max_storage_bytes=space.max_storage_bytes,max_peak_bytes=space.max_peak_bytes,source_mode=SourceMode.CURRENT_STUDENT)
    program=(QuantizationProgram(request,backend=normalized.training if type(normalized.training) is BackendStageSpec else None,
        recovery=normalized.training if type(normalized.training) is RecoveryStageSpec else None,training_fingerprint=space.training_fingerprint)
        if normalized.training is not None else None)
    identity=fingerprint({'source':space.source,'calibration':space.calibration,'evaluation':space.evaluation_context,
        'genotype':genotype_payload(normalized),'target':request.target,'storage_budget':space.max_storage_bytes,
        'max_peak_bytes':space.max_peak_bytes,'source_mode':request.source_mode})
    identity=fingerprint({'workflow':identity,'training_data':space.training_fingerprint,'program':program})
    return QuantizationWorkflow(request,space.source,space.calibration,space.evaluation_context,identity,normalized,program)


QUANTIZATION_GRAMMAR_ADAPTER=GrammarAdapter(validate_space,normalize_genotype,generate_genotype,mutate_genotype,
    crossover_genotypes,compile_genotype,genotype_payload,domain='CPU Linear typed quantization process DAG with prefix rewrites')


def run_quantization_search(model,space,config,*,calibration,evaluate,ledger=None,trace_callback=None,initial_genotypes=None,training_batches=()):
    """Whole loaded-model quality and physical bytes, through FedcoreEvoOptimizer."""
    ledger=ledger or ResourceLedger()
    from .cache import DependencyCache
    cache=DependencyCache()
    def assessment(workflow):
        from fedcore.algorithm.low_rank.grammar_execution import StructuralInvalid
        if source_identity(model)!=workflow.source:
            return StructuralInvalid((Violation('StaleSource','model','Source architecture or tensor state changed'),))
        ref=calibration_identity(calibration)
        if (ref.fingerprint if ref else None)!=workflow.calibration:
            return StructuralInvalid((Violation('StaleCalibration','data','Calibration changed'),))
        if training_identity(training_batches)!=space.training_fingerprint:
            return StructuralInvalid((Violation('StaleTraining','data','Ordered training data changed'),))
        try:ledger.checkpoint('search_assessment')
        except StageBudgetExceeded as error:return error.failure
        artifact=(execute_quantization_program(model,workflow.program,calibration=calibration,training_batches=training_batches,ledger=ledger,cache=cache)
                  if workflow.program is not None else compress_quantization_graph(model,workflow.request,calibration=calibration,ledger=ledger,cache=cache))
        if type(artifact) not in (ExportVerifiedArtifact,ProgramArtifact):return artifact
        quality=evaluate(artifact.model)
        try:ledger.checkpoint('search_assessment_complete')
        except StageBudgetExceeded as error:return error.failure
        return GrammarFitness(float(quality),float(artifact.storage_bytes))
    result=run_grammar_search(space,config,assessment,fitness_context=space.evaluation_context,
        grammar_adapter=QUANTIZATION_GRAMMAR_ADAPTER,strict_deadline=True,trace_callback=trace_callback,initial_genotypes=initial_genotypes)
    result['resource_ledger']=ledger.evidence()
    result['artifact_cache']={'bytes':cache.bytes,'events':cache.events}
    result['process_topology']='One GOLEM genotype node; structural variation occurs inside each checked executable process DAG'
    result['baseline_mode']='Declared recipes with prequantize_bits=(); same engine and evaluation protocol'
    return result
