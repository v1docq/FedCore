"""Batch objectives plugged into the existing training boundary; no runner."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from .approximation import NumericalDomainViolation
from .projection_profiles import pela_regularizer, PELA_SOURCE
from .structured_layers import FactorizedLinear


def observed_loss(prediction, labels, loss, reduction="mean"):
    if loss == "cross_entropy":
        if labels.dtype not in (torch.int32,torch.int64):
            raise NumericalDomainViolation("Cross-entropy requires integral observed labels")
        raw = F.cross_entropy(prediction.reshape(-1,prediction.shape[-1]),labels.reshape(-1).long(),reduction="none")
    elif loss == "mse":
        if prediction.shape != labels.shape:
            raise NumericalDomainViolation("MSE labels and prediction shapes must match")
        raw = (prediction-labels).square()
    else:
        raise NumericalDomainViolation("Unknown observed-label loss")
    return raw.mean() if reduction == "mean" else raw.sum()


class FixedProjectorRecoveryObjective:
    def __init__(self,spec):
        self.spec = spec
        self.evidence = {"objective":"espace_full_weight_observed_loss","loss":spec.loss,
                         "reduction":spec.reduction,"projection_trainable":False,
                         "recovery_scope":"selected_full_weights_and_biases"}

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        return False

    def __call__(self,model,inputs,labels):
        return observed_loss(model(inputs),labels,self.spec.loss,self.spec.reduction)


class PELARecoveryObjective:
    """Private frozen teacher, explicit identity block map and literal losses.

    Only standard tensor block outputs are accepted. Context exit removes all
    hooks on success and failure. The training boundary owns optimizer/batches.
    ``validate_features`` runs before training and rejects missing/reused blocks
    and shape differences rather than silently broadcasting token features.
    """
    def __init__(self,student,teacher,spec,target_paths):
        from .method_execution import _tensor_state_bytes
        self.student,self.teacher,self.spec = student,deepcopy(teacher),spec
        self.teacher.eval().requires_grad_(False)
        if any(t.device.type != "cpu" for t in (*self.teacher.parameters(),*self.teacher.buffers())):
            raise NumericalDomainViolation("PELA's first recovery profile requires a CPU teacher")
        self.teacher_state = {name:value.detach().clone() for name,value in self.teacher.state_dict().items()}
        self.target_paths = tuple(target_paths)
        self.references = {}
        for path in self.target_paths:
            factor = student.get_submodule(path)
            original = teacher.get_submodule(path)
            if type(factor) is not FactorizedLinear or type(original) is not nn.Linear:
                raise NumericalDomainViolation("PELA recovery requires corresponding standard/factorized Linear targets")
            self.references[path] = original.weight.detach().clone()
        for student_path,teacher_path in spec.block_map:
            try:
                student.get_submodule(student_path)
                self.teacher.get_submodule(teacher_path)
            except AttributeError as error:
                raise NumericalDomainViolation("Unknown PELA teacher/student block map") from error
        if spec.freeze == "factors_only":
            student.requires_grad_(False)
            for path in self.target_paths:
                student.get_submodule(path).requires_grad_(True)
        else:
            student.requires_grad_(True)
        if not any(p.requires_grad for p in student.parameters()):
            raise NumericalDomainViolation("PELA recovery requires trainable student parameters")
        self.handles,self.student_features,self.teacher_features = [],{},{}
        self.calls = 0
        self.evidence = {"objective":"pela_base_feature_weight","source":PELA_SOURCE,
            "teacher":"private_frozen_eval","block_map":[list(p) for p in spec.block_map],
            "feature_normalization":spec.feature_normalization,"feature_map":"identity",
            "infinity_norm":spec.infinity_norm,"regularizer":spec.regularizer,
            "epsilon":spec.epsilon,"epsilon_is_hard_constraint":False,
            "freeze":spec.freeze,"trainability":"selected_factor_parameters" if spec.freeze == "factors_only" else "all_student_parameters",
            "requested_updates":spec.updates,
            "teacher_bytes":_tensor_state_bytes(self.teacher),
            "teacher_snapshot_bytes":sum(v.numel()*v.element_size() for v in self.teacher_state.values()),
            "original_weight_reference_bytes":sum(v.numel()*v.element_size() for v in self.references.values())}

    def _capture(self,values,path,detach):
        def hook(_module,_inputs,output):
            if path in values or not isinstance(output,torch.Tensor) or not output.numel() or not bool(torch.isfinite(output).all()):
                raise NumericalDomainViolation("PELA needs one finite tensor feature per mapped block per forward")
            values[path] = output.detach() if detach else output
        return hook

    def __enter__(self):
        if self.handles:
            raise NumericalDomainViolation("PELA objective context cannot be nested")
        try:
            for student_path,teacher_path in self.spec.block_map:
                self.handles.append(self.student.get_submodule(student_path).register_forward_hook(self._capture(self.student_features,student_path,False)))
                self.handles.append(self.teacher.get_submodule(teacher_path).register_forward_hook(self._capture(self.teacher_features,teacher_path,True)))
        except Exception:
            self.__exit__(None,None,None)
            raise
        return self

    def __exit__(self,*exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.student_features.clear()
        self.teacher_features.clear()
        self.evidence["objective_calls"] = self.calls
        if any(not torch.equal(value,self.teacher.state_dict()[name]) for name,value in self.teacher_state.items()):
            raise NumericalDomainViolation("PELA teacher state changed during recovery")
        return False

    def _forward_features(self,inputs):
        if not self.handles:
            raise NumericalDomainViolation("PELA objective must execute inside its hook context")
        self.student_features.clear()
        self.teacher_features.clear()
        with torch.no_grad():
            self.teacher(inputs)
        prediction = self.student(inputs)
        losses = []
        for student_path,teacher_path in self.spec.block_map:
            if student_path not in self.student_features or teacher_path not in self.teacher_features:
                raise NumericalDomainViolation("A mapped PELA feature block was not executed")
            student_feature,teacher_feature = self.student_features[student_path],self.teacher_features[teacher_path]
            if student_feature.shape != teacher_feature.shape:
                raise NumericalDomainViolation("PELA identity feature map requires equal tensor shapes")
            difference = (student_feature-teacher_feature).square()
            losses.append(difference.sum() if self.spec.feature_normalization == "paper_sum_squared" else difference.mean())
        return prediction,torch.stack(losses).sum()/(2*len(losses))

    def validate_features(self,inputs):
        modes = {module:module.training for module in self.student.modules()}
        try:
            self.student.eval()
            with self,torch.no_grad():
                self._forward_features(inputs)
        finally:
            for module,training in modes.items():
                module.training = training

    def __call__(self,model,inputs,labels):
        if model is not self.student:
            raise NumericalDomainViolation("PELA objective belongs to its prepared student")
        prediction,feature = self._forward_features(inputs)
        penalties = []
        for path,reference in self.references.items():
            layer = model.get_submodule(path)
            penalties.append(pela_regularizer(layer.left.weight,layer.right.weight,reference,
                infinity_norm=self.spec.infinity_norm,variant=self.spec.regularizer,epsilon=self.spec.epsilon))
        regularization = torch.stack(penalties).sum()
        self.calls += 1
        return observed_loss(prediction,labels,self.spec.loss,self.spec.reduction)+self.spec.feature_weight*feature+self.spec.regularizer_weight*regularization
