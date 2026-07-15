"""
nnUNetTrainerFocalTverskyPriorDropout — custom nnU-Net v2 trainer for the
soft-prior lesion-detection cascade described in the manuscript.

Two changes vs the default nnUNetTrainer:
  1. Loss = Focal-Tversky + CE (recall-weighted: alpha < beta penalises false
     negatives more), replacing Dice + CE. Raises lesion sensitivity.
  2. Prior-channel dropout: with probability p, zero the soft-prior input
     channels for a training sample, so the model cannot fully rely on the
     priors and retains a PET(-CT)-only detection ability as a recall floor.

Channel layout is data-dependent, so the prior channels are configurable.
For a 3-channel input [PET, soft_slab, soft_MA] the priors are channels (1, 2);
for a 4-channel PET+CT input [PET, CT, soft_slab, soft_MA] they are (2, 3).

Hyperparameters overridable via environment variables:
  TVERSKY_ALPHA   (default 0.3)
  TVERSKY_BETA    (default 0.7)   # beta > alpha => recall-weighted
  TVERSKY_GAMMA   (default 1.0)
  PRIOR_DROPOUT_P (default 0.2)
  PRIOR_CHANNELS  (default "1,2") # comma-separated input-channel indices

This file is a research reference implementation released with the manuscript.
It contains no data, no patient identifiers, and no dataset-specific paths.
"""
import os
import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.dice import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


class MemoryEfficientTverskyLoss(nn.Module):
    """Tversky loss mirroring MemoryEfficientSoftDiceLoss tensor conventions.
    Tversky = TP / (TP + alpha*FP + beta*FN); loss = (1 - Tversky)**gamma."""
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1e-5,
                 ddp=True, alpha=0.3, beta=0.7, gamma=1.0):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes, dtype=torch.float32) if loss_mask is None \
                else (y_onehot * loss_mask).sum(axes, dtype=torch.float32)

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            tp = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            tp = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)

        fp = sum_pred - tp
        fn = sum_gt - tp

        if self.batch_dice:
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0, dtype=torch.float32)
                fp = AllGatherGrad.apply(fp).sum(0, dtype=torch.float32)
                fn = AllGatherGrad.apply(fn).sum(0, dtype=torch.float32)
            tp = tp.sum(0, dtype=torch.float32)
            fp = fp.sum(0, dtype=torch.float32)
            fn = fn.sum(0, dtype=torch.float32)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth).clamp_min(1e-8)
        ft = torch.pow((1.0 - tversky).clamp_min(1e-8), self.gamma)
        return ft.mean()


class Tversky_and_CE_loss(nn.Module):
    def __init__(self, tversky_kwargs, ce_kwargs, weight_ce=1, weight_tversky=1, ignore_label=None):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.weight_ce = weight_ce
        self.weight_tversky = weight_tversky
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.tv = MemoryEfficientTverskyLoss(apply_nonlin=softmax_helper_dim1, **tversky_kwargs)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label not implemented for one-hot targets'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        tv_loss = self.tv(net_output, target_dice, loss_mask=mask) if self.weight_tversky != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        return self.weight_ce * ce_loss + self.weight_tversky * tv_loss


def _parse_prior_channels(default=(1, 2)):
    env = os.environ.get('PRIOR_CHANNELS')
    if not env:
        return tuple(default)
    return tuple(int(x) for x in env.replace(' ', '').split(',') if x != '')


class nnUNetTrainerFocalTverskyPriorDropout(nnUNetTrainer):
    tversky_alpha = 0.3
    tversky_beta = 0.7
    tversky_gamma = 1.0
    prior_dropout_p = 0.2
    prior_channels = (1, 2)   # default: [PET, soft_slab, soft_MA]

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.tversky_alpha = float(os.environ.get('TVERSKY_ALPHA', self.tversky_alpha))
        self.tversky_beta = float(os.environ.get('TVERSKY_BETA', self.tversky_beta))
        self.tversky_gamma = float(os.environ.get('TVERSKY_GAMMA', self.tversky_gamma))
        self.prior_dropout_p = float(os.environ.get('PRIOR_DROPOUT_P', self.prior_dropout_p))
        self.prior_channels = _parse_prior_channels(self.prior_channels)
        self.print_to_log_file(
            f"[FocalTverskyPriorDropout] alpha={self.tversky_alpha} beta={self.tversky_beta} "
            f"gamma={self.tversky_gamma} prior_dropout_p={self.prior_dropout_p} "
            f"prior_channels={self.prior_channels}")

    def _build_loss(self):
        assert not self.label_manager.has_regions, "region-based training not supported by this trainer"
        loss = Tversky_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5,
             'do_bg': False, 'ddp': self.is_ddp,
             'alpha': self.tversky_alpha, 'beta': self.tversky_beta, 'gamma': self.tversky_gamma},
            {}, weight_ce=1, weight_tversky=1, ignore_label=self.label_manager.ignore_label)

        if self._do_i_compile():
            loss.tv = torch.compile(loss.tv)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        if self.prior_dropout_p > 0 and self.prior_channels and data.shape[1] > max(self.prior_channels):
            data = data.clone()
            drop = torch.rand(data.shape[0]) < self.prior_dropout_p
            if drop.any():
                for c in self.prior_channels:
                    data[drop, c] = 0.0
            batch = {**batch, 'data': data}
        return super().train_step(batch)
