"""Unified CPU quantization API; imports stay lazy for existing BaseQuantizer."""
from importlib import import_module

_EXPORTS={
    'quantize':'program','QuantizationProgram':'program','RecoveryStageSpec':'program','ProgramArtifact':'program','training_identity':'program',
    'QuantizationGraphRequest':'graph_specs','OperatorRecipe':'graph_specs','TransformSpec':'graph_specs','Method':'graph_specs',
    'ExecutionTargetSpec':'graph_specs','QuantizedTensorSpec':'tensor_formats','Grid':'tensor_formats','Rounding':'tensor_formats',
    'check_quantization_graph':'graph_execution','ExportVerifiedArtifact':'graph_execution',
    'BackendStageSpec':'training','BaseQuantizer':'quantizers','ResourceLedger':'resources','StageBudgets':'resources',
    'prepare_quantization_catalogue':'catalogue','allocate_quantization_catalogue':'catalogue','assess_quantization_selection':'catalogue',
    'prepare_quantization_space':'grammar','run_quantization_search':'grammar','GrammarSearchConfig':'grammar',
}
__all__=list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    value=getattr(import_module('.'+_EXPORTS[name],__name__),name)
    globals()[name]=value
    return value
