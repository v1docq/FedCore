"""Pareto dominance: no worse in every objective and better in at least one."""
import torch


class ParetoMetrics:
    def pareto_metric_list(self,costs,maximise=True):
        values=torch.as_tensor(costs)
        if values.numel()==0:
            return torch.zeros(values.shape[0] if values.ndim else 0,dtype=torch.bool,device=values.device)
        if values.ndim!=2:raise ValueError('Pareto costs must have shape [objects, objectives]')
        if not torch.isfinite(values).all():raise ValueError('Pareto costs must be finite (NaN/Inf unsupported)')
        directions=torch.as_tensor(maximise,dtype=torch.bool,device=values.device)
        if directions.ndim and directions.shape!=(values.shape[1],):raise ValueError('One direction per objective required')
        mask=torch.ones(len(values),dtype=torch.bool,device=values.device)
        for i in range(len(values)):
            no_worse=torch.where(directions,values>=values[i],values<=values[i]).all(-1)
            better=torch.where(directions,values>values[i],values<values[i]).any(-1)
            dominates=no_worse & better
            mask[i]=not bool(dominates.any())
        return mask
