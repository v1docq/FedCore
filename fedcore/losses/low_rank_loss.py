"""Regularizers on the canonical factors, including grouped/spatial matrices."""
import torch
from torch import nn
from fedcore.models.network_impl.decomposed_layers import IDecomposed


def _zero(model):
    p = next(model.parameters(),None)
    return p.sum()*0 if p is not None else torch.tensor(0.)


class SVDLoss(nn.Module):
    def __init__(self,factor=1.):
        super().__init__()
        self.factor = factor


class OrthogonalLoss(SVDLoss):
    def forward(self,model):
        penalties = []
        for module in model.modules():
            if isinstance(module,IDecomposed) and module.U is not None:
                u,_,vh = module.get_U_S_Vh()
                rank = u.shape[-1]
                eye = torch.eye(rank,device=u.device,dtype=u.dtype)
                penalties.append(((u.transpose(-2,-1)@u-eye).square().sum(dim=(-2,-1)) +
                                  (vh@vh.transpose(-2,-1)-eye).square().sum(dim=(-2,-1))).mean()/rank)
        return self.factor*torch.stack(penalties).mean() if penalties else _zero(model)


class HoyerLoss(SVDLoss):
    def forward(self,model):
        values = []
        for name,p in model.named_parameters():
            if name.split('.')[-1] == 'S':
                norm = p.norm()
                values.append(p.abs().sum()/norm.clamp_min(torch.finfo(p.dtype).tiny))
        return self.factor*torch.stack(values).mean() if values else _zero(model)
