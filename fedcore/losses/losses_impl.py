import torch


def alpha_divergence(teacher_logits, student_logits, alpha, reduction="none", clip=1e3):
    """Compute alpha-divergence distillation loss.

    Args:
        teacher_logits: Logits from the teacher network.
        student_logits: Logits from the student network.
        alpha: Divergence coefficient in ``[0, 1]``. Values near 0 approximate
            ``KL(teacher || student)``; values near 1 approximate
            ``KL(student || teacher)``.
        reduction: Reduction mode applied across the batch.
        clip: Clamp value for numerical stability.
    """
    assert isinstance(alpha, float)
    q_prob = torch.nn.functional.softmax(teacher_logits, dim=1)
    p_prob = torch.nn.functional.softmax(student_logits, dim=1)
    alpha_is_small = abs(alpha) < 1e-3
    alpha_is_big = abs(alpha - 1.0) < 1e-3
    if alpha_is_small:
        lndiff = q_prob.log() - p_prob.log()
        lndiff.clamp_(-clip, clip)
        loss = torch.sum(q_prob * lndiff, dim=1)  # KL(q||p)
    elif alpha_is_big:
        loss = torch.sum(p_prob * (p_prob.log() - q_prob.log()), dim=1)  # KL(p||q)
    else:
        iw_ratio = torch.pow(p_prob / q_prob, alpha)
        iw_ratio = iw_ratio.clamp(0, clip)
        loss = (
            1.0 / (alpha * (alpha - 1.0)) * ((iw_ratio * q_prob).sum(1) - 1.0)
        )  # D_a(p||q)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def f_divergence(teacher_logits, student_logits, alpha, iw_clip=1e3, p_normalize=False):
    """Compute f-divergence distillation loss.

    Args:
        teacher_logits: Logits from the teacher network.
        student_logits: Logits from the student network.
        alpha: Divergence coefficient in ``[0, 1]``.
        iw_clip: Clamp value for the importance-ratio term.
        p_normalize: Whether to normalize student probabilities before
            computing the divergence.
    """
    assert isinstance(alpha, float)
    teacher_prob = torch.nn.functional.softmax(teacher_logits, dim=1).detach()
    student_prob = (
        student_logits.detach()
        if p_normalize
        else torch.nn.functional.softmax(student_logits, dim=1).detach()
    )
    teacher_log_prob = torch.nn.functional.log_softmax(
        teacher_logits, dim=1
    )  # gradient is only backpropagated here
    importance_ratio = student_prob / teacher_prob

    alpha_is_small = abs(alpha) < 1e-3
    alpha_is_big = abs(alpha - 1.0) < 1e-3
    if alpha_is_small:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif alpha_is_big:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha)
        iw_alpha = iw_alpha.clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(teacher_prob * (f - f_base), dim=1)
    grad_loss = -torch.sum(teacher_prob * rho_f * teacher_log_prob, dim=1)
    return loss, grad_loss
