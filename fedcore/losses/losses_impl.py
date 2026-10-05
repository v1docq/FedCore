"""Stable distillation divergences: teacher first, student second.

Teacher logits are detached. Alpha=0 is KL(teacher||student), alpha=1 is
KL(student||teacher); interior alpha is the normalized alpha divergence.
"""
import torch
from torch.nn import functional as F


def reduce_loss(loss,reduction):
    if reduction in ('mean','batchmean'):
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    raise ValueError(f'Unknown reduction: {reduction}')


def alpha_divergence(teacher_logits,student_logits,alpha,reduction='none',clip=1e3):
    if teacher_logits.shape != student_logits.shape or student_logits.ndim < 2:
        raise ValueError('Matching logits with a class dimension are required')
    q = F.log_softmax(teacher_logits.detach(),dim=-1)
    p = F.log_softmax(student_logits,dim=-1)
    a = torch.as_tensor(alpha,device=p.device,dtype=p.dtype)
    if not torch.isfinite(a).all() or ((a<0)|(a>1)).any():
        raise ValueError('alpha must be finite and in [0,1]')
    a = torch.broadcast_to(a,p.shape[:-1])
    safe_a = a.clamp(1e-4,1-1e-4)
    log_mass = torch.logsumexp((1-safe_a[...,None])*q + safe_a[...,None]*p,dim=-1).clamp(max=0)
    middle = -torch.expm1(log_mass)/(safe_a*(1-safe_a))
    forward = (q.exp()*(q-p)).sum(-1)
    reverse = (p.exp()*(p-q)).sum(-1)
    loss = torch.where(a == 0,forward,torch.where(a == 1,reverse,middle))
    return reduce_loss(loss,reduction)


def f_divergence(teacher_logits,student_logits,alpha,iw_clip=1e3,p_normalize=False):
    """Return detached diagnostic and exact differentiable student loss.

    This is the alpha-divergence itself, not an importance-ratio surrogate.
    ``p_normalize=True`` accepts nonnegative normalized student probabilities.
    """
    if p_normalize:
        if (student_logits < 0).any() or not torch.allclose(student_logits.sum(-1),torch.ones_like(student_logits.sum(-1))):
            raise ValueError('Normalized student probabilities are required')
        student_logits = student_logits.clamp_min(torch.finfo(student_logits.dtype).tiny).log()
    loss = alpha_divergence(teacher_logits,student_logits,alpha)
    return loss.detach(),loss
