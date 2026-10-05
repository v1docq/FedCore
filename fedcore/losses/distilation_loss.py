import torch

from fedcore.losses.losses_impl import f_divergence, alpha_divergence


class DistilationLoss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        if reduction not in ("mean", "batchmean", "sum", "none"):
            raise ValueError(f"Unknown reduction: {reduction}")

    def reduce(self, loss):
        from fedcore.losses.losses_impl import reduce_loss
        return reduce_loss(loss, self.reduction)


class CrossEntropyLossSoft(DistilationLoss):
    """inplace distillation for image classification"""

    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.detach().unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return self.reduce(cross_entropy_loss.reshape(-1))


class FdivTopKLossSoft(DistilationLoss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    """

    def forward(self, output, target, T=1.0):
        if T <= 0:
            raise ValueError("temperature must be positive")
        output, target = output / T, target / T
        output_prob = torch.nn.functional.softmax(output, dim=1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)

        target_prob = torch.nn.functional.softmax(target.detach(), dim=1)
        # density ratios
        density_ratio = target_prob.clamp_min(torch.finfo(target_prob.dtype).tiny).log() - output_log_prob
        # _, indices = torch.topk(density_ratio, k, dim=1, largest=True)
        # one_hot_w = torch.zeros_like(target).scatter(1, indices, 1)
        one_hot_w = torch.ge(density_ratio, 0.0).float()

        ##probablity
        # _, indices = torch.topk(target_prob, k, dim=1, largest=True)
        # one_hot_p = torch.zeros_like(target).scatter(1, indices, 1)

        # one_hot = one_hot_w * one_hot_p
        one_hot = one_hot_w.detach()
        # loss = -torch.sum( target_prob * one_hot * output_log_prob, dim=1)
        loss = -torch.sum(target_prob.detach() * one_hot * output_log_prob, dim=1)
        return self.reduce(loss)


class HardThresLossSoft(DistilationLoss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    """

    def forward(self, output, target, eps=0.1):
        output_prob = torch.nn.functional.softmax(output, dim=1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)

        target_prob = torch.nn.functional.softmax(target.detach(), dim=1)
        one_hot = torch.ge(target_prob, eps).float()
        n_class = output.size(1)
        noise_labels = torch.sum(target_prob * (1.0 - one_hot), 1, keepdim=True) / (
            (n_class - torch.sum(one_hot, 1, keepdim=True)).clamp_min(1)
        )
        target_prob = one_hot * target_prob + (1.0 - one_hot) * noise_labels

        loss = -torch.sum(target_prob * output_log_prob, dim=1)
        return self.reduce(loss)


class TopkLossSoft(DistilationLoss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    """

    def forward(self, output, target, k=5):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)

        target_prob = torch.nn.functional.softmax(target.detach(), dim=1)
        k = min(k, output.size(1))
        topk_vals, topk_idxs = torch.topk(target_prob, k, dim=1, largest=True)
        one_hot = torch.zeros_like(target).scatter(1, topk_idxs, 1)  # topk, one hot
        n_class = output.size(1)
        noise_labels = torch.sum(target_prob * (1.0 - one_hot), 1, keepdim=True) / (
            max(n_class - k, 1)
        )
        target_prob = one_hot * target_prob + (1.0 - one_hot) * noise_labels

        loss = -torch.sum(target_prob * output_log_prob, dim=1)
        return self.reduce(loss)


class KLLossSoft(DistilationLoss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    T: temperature
    KL(p||q) = Ep \log p - \Ep log q
    """

    def forward(self, output, soft_logits, target=None, temperature=1.0, alpha=0.9):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        kd_loss = alpha_divergence(soft_logits / temperature, output / temperature,
                                   0.0, reduction="none") * temperature ** 2
        if target is not None:
            ce = torch.nn.functional.cross_entropy(output, target, reduction="none")
            kd_loss = alpha * kd_loss + (1 - alpha) * ce
        return self.reduce(kd_loss)


class ReverseKLLossSoft(DistilationLoss):
    """inplace distillation for image classification
    output: output logits of the student network
    target: output logits of the teacher network
    T: temperature
    KL(q||p) = Eq(\log q - \log p)
    """

    def forward(self, output, target, T=1.0):
        if T <= 0:
            raise ValueError("temperature must be positive")
        return self.reduce(alpha_divergence(target / T, output / T, 1.0) * T ** 2)


class AdaptiveLossSoft(DistilationLoss):
    def __init__(self, alpha_min, alpha_max, iw_clip=1):
        super(AdaptiveLossSoft, self).__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip

    def forward(
        self,
        output,
        target,
        alpha_min=None,
        alpha_max=None,
        mode_seeking_weight=None,
        p_normalize=False,
        reduction=True,
    ):
        alpha_min = self.alpha_min if alpha_min is None else alpha_min
        alpha_max = self.alpha_max if alpha_max is None else alpha_max

        # loss_left = alpha_divergence(target, output, alpha_min, iw_clip=self.iw_clip)
        # loss_right = alpha_divergence(target, output, alpha_max, iw_clip=self.iw_clip)
        # loss = torch.max(loss_left, loss_right)
        if mode_seeking_weight is None:
            loss_left, grad_loss_left = f_divergence(
                target, output, alpha_min, iw_clip=self.iw_clip, p_normalize=p_normalize
            )
            loss_right, grad_loss_right = f_divergence(
                target, output, alpha_max, iw_clip=self.iw_clip, p_normalize=p_normalize
            )

            # change max -> min
            ind = torch.gt(loss_left, loss_right).float()
            loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

        else:
            alpha = alpha_min * mode_seeking_weight + alpha_max * (
                1.0 - mode_seeking_weight
            )
            _, loss = f_divergence(target, output, alpha, iw_clip=self.iw_clip)
            # loss = mode_seeking_weight * grad_loss_left + (1.0 - mode_seeking_weight) * grad_loss_right

        if not reduction:
            return loss
        return self.reduce(loss)


class AlphaDivergenceLossSoft(DistilationLoss):
    """alpha divergence
    output: output logits of the student network
    target: output logits of the teacher network
    T: temperature
    D_a(q||p) = 1/(a(a-1)) * E_q[p^a q^-a -1] = 1/(a(a-1)) (\sum p^a q^1-a - 1)
    """

    def forward(self, output, target, alpha):
        loss = alpha_divergence(target, output, alpha, reduction="none")
        return self.reduce(loss)


class EntropyAlphaDivergence(DistilationLoss):

    def forward(self, output, target):
        prob = torch.nn.functional.softmax(target.detach(), dim=1)
        ent = -torch.sum(prob * prob.clamp_min(torch.finfo(prob.dtype).tiny).log(), dim=1)
        # alpha = (ent - torch.min(ent)) / (torch.max(ent) - torch.min(ent))
        alpha = 1.0 - torch.max(prob, 1).values
        # alpha = torch.max(prob, 1).values
        loss = alpha_divergence(target, output, alpha, reduction="none")

        thr = torch.gt(torch.max(prob, 1).values, 0.8).float()
        forward_kl = alpha_divergence(target, output, 0.0, reduction="none")
        reverse_kl = alpha_divergence(target, output, 1.0, reduction="none")
        loss = thr * forward_kl + (1.0 - thr) * reverse_kl
        return self.reduce(loss)


"""
    idea: cross entropy loss doesn't satisfy triangle inequality
          we might want to use a symmetric divergence
"""


class JSDLossSoft(DistilationLoss):
    def __init__(self, reduction="mean"):
        super(JSDLossSoft, self).__init__(reduction=reduction)

    # {{\rm {JSD}}}(P\parallel Q)={\frac  {1}{2}}D(P\parallel M)+{\frac  {1}{2}}D(Q\parallel M)
    def forward(self, output, target):
        output_prob = torch.nn.functional.softmax(output, dim=1)
        target_prob = torch.nn.functional.softmax(target.detach(), dim=1)

        M = (output_prob + target_prob) / 2.0
        # student network
        kl_qm = output_prob * (torch.nn.functional.log_softmax(output, dim=1) - M.clamp_min(torch.finfo(M.dtype).tiny).log())
        kl_pm = target_prob * (target_prob.clamp_min(torch.finfo(target_prob.dtype).tiny).log() - M.clamp_min(torch.finfo(M.dtype).tiny).log())
        loss = torch.sum(0.5 * (kl_qm + kl_pm), dim=1)
        return self.reduce(loss)


class JSDLossSmooth(DistilationLoss):
    def __init__(self, label_smoothing=0.1, reduction="mean"):
        super(JSDLossSmooth, self).__init__(reduction=reduction)
        self.eps = label_smoothing

    # {{\rm {JSD}}}(P\parallel Q)={\frac  {1}{2}}D(P\parallel M)+{\frac  {1}{2}}D(Q\parallel M)
    def forward(self, output, target):
        output_prob = torch.nn.functional.softmax(output, dim=1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target_prob = one_hot * (1 - self.eps) + self.eps / n_class

        M = (output_prob + target_prob) / 2.0
        # student network
        kl_qm = output_prob * (torch.nn.functional.log_softmax(output, dim=1) - M.clamp_min(torch.finfo(M.dtype).tiny).log())
        kl_pm = target_prob * (target_prob.clamp_min(torch.finfo(target_prob.dtype).tiny).log() - M.clamp_min(torch.finfo(M.dtype).tiny).log())
        loss = torch.sum(0.5 * (kl_qm + kl_pm), dim=1)
        return self.reduce(loss)


class CrossEntropyLossSmooth(DistilationLoss):
    def __init__(self, label_smoothing=0.1, reduction="mean"):
        super(CrossEntropyLossSmooth, self).__init__(reduction=reduction)
        self.eps = label_smoothing

    """ label smooth """

    def forward(self, output, target, reduction=True):
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - self.eps) + self.eps / n_class
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.detach().unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        loss = -torch.bmm(target, output_log_prob)
        return loss if not reduction else self.reduce(loss)


class CrossEntropyEma(DistilationLoss):
    def __init__(self, label_smoothing=0.1, reduction="mean"):
        super(CrossEntropyEma, self).__init__(reduction=reduction)
        self.eps = label_smoothing

    def _forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.detach().unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        loss = -torch.bmm(target, output_log_prob)
        return loss.squeeze(-2).squeeze(-1)

    def forward(self, output, target, ema_output=None, beta=0.1):
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - self.eps) + self.eps / n_class

        loss_model = self._forward(output, target)
        if ema_output is not None:
            loss_ema = self._forward(ema_output, target)
            loss_model_ema = self._forward(
                output, torch.nn.functional.softmax(ema_output.detach(), dim=1)
            )
            indicators = torch.ge(loss_model, loss_ema).float() * beta
            loss = (1.0 - indicators) * loss_model + indicators * loss_model_ema
        else:
            loss = loss_model
        return self.reduce(loss)
