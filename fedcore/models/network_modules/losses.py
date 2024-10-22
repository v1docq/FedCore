from __future__ import print_function

from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch import nn, Tensor
from torch.autograd import Variable
from fastai.torch_core import Module

from fedcore.architecture.settings.computational import default_device


def lambda_prepare(
    val: torch.Tensor, lambda_: Union[int, list, torch.Tensor]
) -> torch.Tensor:
    """Prepares lambdas for corresponding equation or bcond type.

    Args:
        val (_type_): operator tensor or bval tensor
        lambda_ (Union[int, list, torch.Tensor]): regularization parameters values

    Returns:
        torch.Tensor: torch.Tensor with lambda_ values,
        len(lambdas) = number of columns in val
    """

    if isinstance(lambda_, torch.Tensor):
        return lambda_

    if isinstance(lambda_, int):
        try:
            lambdas = torch.ones(val.shape[-1], dtype=val.dtype) * lambda_
        except BaseException:
            lambdas = torch.tensor(lambda_, dtype=val.dtype)
    elif isinstance(lambda_, list):
        lambdas = torch.tensor(lambda_, dtype=val.dtype)

    return lambdas.reshape(1, -1)


class ExpWeightedLoss(nn.Module):

    def __init__(self, time_steps, tolerance):
        self.n_t = time_steps
        self.tol = tolerance
        super().__init__()

    def forward(self, input_: Tensor, target: Tensor) -> torch.Tensor:
        """Computes causal loss, which is calculated with weights matrix:
        W = exp(-tol*(Loss_i)) where Loss_i is sum of the L2 loss from 0
        to t_i moment of time. This loss function should be used when one
        of the DE independent parameter is time.

        Args:
            input_ (torch.Tensor): predicted values.
            target (torch.Tensor): target values.

        Returns:
            loss (torch.Tensor): loss.
            loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
        """
        # res = torch.sum(input ** 2, dim=0).reshape(self.n_t, -1)
        res = torch.mean(input_, axis=0).reshape(self.n_t, -1)
        target = torch.mean(target, axis=0).reshape(self.n_t, -1)
        m = torch.triu(
            torch.ones((self.n_t, self.n_t), dtype=res.dtype), diagonal=1
        ).T.to(default_device())
        with torch.no_grad():
            w = torch.exp(-self.tol * (m @ res))
        loss = torch.mean(w * res)
        loss = torch.mean(torch.sqrt((loss - target) ** 2).flatten())
        return loss


class HuberLoss(nn.Module):
    """Huber loss

    Creates a criterion that uses a squared term if the absolute
    element-wise error falls below delta and a delta-scaled L1 term otherwise.
    This loss combines advantages of both :class:`L1Loss` and :class:`MSELoss`; the
    delta-scaled L1 region makes the loss less sensitive to outliers than :class:`MSELoss`,
    while the L2 region provides smoothness over :class:`L1Loss` near 0. See
    `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_ for more information.
    This loss is equivalent to nn.SmoothL1Loss when delta == 1.
    """

    def __init__(self, reduction="mean", delta=1.0):
        assert reduction in [
            "mean",
            "sum",
            "none",
        ], "You must set reduction to 'mean', 'sum' or 'none'"
        self.reduction, self.delta = reduction, delta
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        abs_diff = torch.abs(diff)
        mask = abs_diff < self.delta
        loss = torch.cat(
            [
                (0.5 * diff[mask] ** 2),
                self.delta * (abs_diff[~mask] - (0.5 * self.delta)),
            ]
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LogCoshLoss(nn.Module):
    def __init__(self, reduction="mean", delta=1.0):
        assert reduction in [
            "mean",
            "sum",
            "none",
        ], "You must set reduction to 'mean', 'sum' or 'none'"
        self.reduction, self.delta = reduction, delta
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.log(torch.cosh(input - target + 1e-12))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MaskedLossWrapper(Module):
    def __init__(self, crit):
        self.loss = crit

    def forward(self, inp, targ):
        inp = inp.flatten(1)
        targ = targ.flatten(1)
        mask = torch.isnan(targ)
        inp, targ = inp[~mask], targ[~mask]
        return self.loss(inp, targ)


class CenterLoss(Module):
    """Code in Pytorch has been slightly modified from:
    https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
    Based on paper: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        c_out (int): number of classes.
        logits_dim (int): dim 1 of the logits. By default same as c_out (for one hot encoded logits)

    """

    def __init__(self, c_out, logits_dim=None):
        if logits_dim is None:
            logits_dim = c_out
        self.c_out, self.logits_dim = c_out, logits_dim
        self.centers = nn.Parameter(torch.randn(c_out, logits_dim))
        self.classes = nn.Parameter(torch.arange(c_out).long(), requires_grad=False)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, logits_dim).
            labels: ground truth labels with shape (batch_size).
        """
        bs = x.shape[0]
        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(bs, self.c_out)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.c_out, bs)
            .T
        )
        distmat = torch.addmm(distmat, x, self.centers.T, beta=1, alpha=-2)

        labels = labels.unsqueeze(1).expand(bs, self.c_out)
        mask = labels.eq(self.classes.expand(bs, self.c_out))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / bs

        return loss


class CenterPlusLoss(Module):

    def __init__(self, loss, c_out, λ=1e-2, logits_dim=None):
        self.loss, self.c_out, self.λ = loss, c_out, λ
        self.centerloss = CenterLoss(c_out, logits_dim)

    def forward(self, x, labels):
        return self.loss(x, labels) + self.λ * self.centerloss(x, labels)

    def __repr__(self):
        return f"CenterPlusLoss(loss={self.loss}, c_out={self.c_out}, λ={self.λ})"


class FocalLoss(Module):
    """Weighted, multiclass focal loss"""

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper. Defaults to 2.
            reduction (str, optional): 'mean', 'sum' or 'none'. Defaults to 'mean'.
        """
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:

        log_p = F.log_softmax(x, dim=-1)
        pt = log_p[torch.arange(len(x)), y].exp()
        ce = self.nll_loss(log_p, y)
        loss = (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class TweedieLoss(Module):
    def __init__(self, p=1.5, eps=1e-8):
        """
        Tweedie loss as calculated in LightGBM
        Args:
            p: tweedie variance power (1 < p < 2)
            eps: small number to avoid log(zero).
        """
        assert 1 < p < 2, "make sure 1 < p < 2"
        self.p, self.eps = p, eps

    def forward(self, inp, targ):
        "Poisson and compound Poisson distribution, targ >= 0, inp > 0"
        inp = inp.flatten()
        targ = targ.flatten()
        torch.clamp_min_(inp, self.eps)
        a = targ * torch.exp((1 - self.p) * torch.log(inp)) / (1 - self.p)
        b = torch.exp((2 - self.p) * torch.log(inp)) / (2 - self.p)
        loss = -a + b
        return loss.mean()


class SMAPELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return 100 * torch.mean(
            2 * torch.abs(input - target) / (torch.abs(target) + torch.abs(input))
            + 1e-8
        )


class RMSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(input, target))
        return loss


class DistributionLoss(nn.Module):
    distribution_class: distributions.Distribution
    distribution_arguments: List[str]
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
    need_affine = True
    scale_dependent_idx = tuple()
    loc_dependent_idx = tuple()

    def __init__(
        self,
        reduction="mean",
    ):
        super().__init__()
        self.reduction = getattr(torch, reduction) if reduction else lambda x: x

    @classmethod
    def map_x_to_distribution(cls, x: torch.Tensor) -> distributions.Distribution:
        """
        Map the a tensor of parameters to a probability distribution.

        Args:
            x (torch.Tensor): parameters for probability distribution. Last dimension will index the parameters

        Returns:
            distributions.Distribution: torch probability distribution as defined in the
                class attribute ``distribution_class``
        """
        distr = cls._map_x_to_distribution(x)
        if cls.need_affine:
            scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
            distr = distributions.TransformedDistribution(distr, [scaler])
        return distr

    @classmethod
    def _map_x_to_distribution(cls, x):
        raise NotImplemented

    def forward(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        distribution = self.map_x_to_distribution(y_pred)
        loss = -distribution.log_prob(y_actual)
        loss = self.reduction(loss)
        return loss


class NormalDistributionLoss(DistributionLoss):
    """
    Normal distribution loss.
    """

    distribution_class = distributions.Normal
    distribution_arguments = ["loc", "scale"]
    scale_dependent_idx = (1,)
    loc_dependent_idx = (0,)
    need_affine = False

    @classmethod
    def _map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        loc = x[..., -2]
        scale = F.softplus(x[..., -1])
        distr = self.distribution_class(loc=loc, scale=scale)
        return distr


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(
        torch.unsqueeze(a[:, 0], 1), b[:, 0]
    )
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(
        torch.unsqueeze(a[:, 1], 1), b[:, 1]
    )

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = (
        torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)
        + area
        - iw * ih
    )

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        print("classifications", classifications.shape)
        print("regressions", regressions.shape)
        print("anchors", anchors.shape)
        print("annotations", annotations.shape)
        print(annotations)
        alpha = 0.25
        gamma = 1.5
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(
                anchors[0, :, :], bbox_annotation[:, :4]
            )  # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[
                positive_indices, assigned_annotations[positive_indices, 4].long()
            ] = 1

            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            alpha_factor = torch.where(
                torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor
            )
            focal_weight = torch.where(
                torch.eq(targets, 1.0), 1.0 - classification, classification
            )
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(
                targets * torch.log(classification)
                + (1.0 - targets) * torch.log(1.0 - classification)
            )

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(
                torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda()
            )

            classification_losses.append(
                cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0)
            )

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                ~positive_indices

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0,
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        return torch.stack(classification_losses).mean(
            dim=0, keepdim=True
        ), torch.stack(regression_losses).mean(dim=0, keepdim=True)


class FocalLoss1(nn.Module):
    def __init__(self, num_classes, device):
        super(FocalLoss1, self).__init__()
        self.num_classes = num_classes
        self.device = device

    def focal_loss(self, x, y):
        """Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        """
        alpha = 0.25
        gamma = 2

        t = F.one_hot(y.data, 1 + self.num_classes)  # [N,21]
        t = t[:, 1:]  # exclude background
        t = Variable(t)

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1 - pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, reduction="sum")

    def focal_loss_alt(self, x, y, alpha=0.25, gamma=1.5):
        """Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        """
        t = F.one_hot(y, self.num_classes + 1)
        t = t[:, 1:]

        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (2 * xt + 1).sigmoid()
        pt = pt.clamp(1e-7, 1.0)
        w = (0 + alpha) * (0 + t) + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / gamma
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        """

        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(
            masked_loc_preds, masked_loc_targets, reduction="sum"
        )

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        num_peg = pos_neg.data.long().sum()
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_peg), end=' | ')
        loss = loc_loss / num_pos + cls_loss / num_peg
        return loss
