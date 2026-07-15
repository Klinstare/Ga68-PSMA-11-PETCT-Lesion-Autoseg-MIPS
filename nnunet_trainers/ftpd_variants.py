"""
ftpd_variants.py — named hyperparameter variants of
nnUNetTrainerFocalTverskyPriorDropout used in the manuscript ablations.

Each subclass has a distinct class name so nnU-Net writes it to its own results
folder. The naming encodes: d<prior_dropout%>_b<tversky_beta*10>.

Channel layouts (see build_dataset.py --modality / --priors):
  [PET, soft_slab, soft_MA]           -> prior_channels = (1, 2)
  [PET, CT, soft_slab, soft_MA]       -> prior_channels = (2, 3)   (Stage II + CT)
  [PET, soft_MA]                      -> prior_channels = (1,)
  [PET, CT, soft_MA]                  -> prior_channels = (1, 2)   via MAonly_petct

Deploy: place this file (and nnUNetTrainerFocalTverskyPriorDropout.py) on the
PYTHONPATH under nnU-Net's trainer discovery path, e.g.
  <nnunetv2>/training/nnUNetTrainer/variants/loss/
so the classes are found by name via `nnUNetv2_train ... -tr <ClassName>`.
"""
# When installed into nnU-Net's trainer path (see README), the base trainer is
# importable via its full module path. If you instead keep these two files in a
# flat directory on PYTHONPATH, change this to:
#   from nnUNetTrainerFocalTverskyPriorDropout import nnUNetTrainerFocalTverskyPriorDropout as _Base
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerFocalTverskyPriorDropout import (
    nnUNetTrainerFocalTverskyPriorDropout as _Base,
)


# ---- PET + soft_slab + soft_MA (3 channels) --------------------------------
class nnUNetTrainerFTPD_d20_b70(_Base):
    """Baseline config (dropout 0.2, beta 0.7)."""
    prior_dropout_p = 0.2
    tversky_alpha = 0.3
    tversky_beta = 0.7
    prior_channels = (1, 2)


class nnUNetTrainerFTPD_d50_b80(_Base):
    """Recall-recovery config (dropout 0.5, beta 0.8)."""
    prior_dropout_p = 0.5
    tversky_alpha = 0.2
    tversky_beta = 0.8
    prior_channels = (1, 2)


class nnUNetTrainerFTPD_d80_b80(_Base):
    """Aggressive prior-dropout (0.8): PET recall floor."""
    prior_dropout_p = 0.8
    tversky_alpha = 0.2
    tversky_beta = 0.8
    prior_channels = (1, 2)


# ---- PET + CT + soft_slab + soft_MA (4 channels, Stage II with CT) ----------
class nnUNetTrainerFTPD_PETCT_d50_b80(_Base):
    """Stage-II with CT: input [PET, CT, soft_slab, soft_MA]; priors on ch (2,3)."""
    prior_dropout_p = 0.5
    tversky_alpha = 0.2
    tversky_beta = 0.8
    prior_channels = (2, 3)


# ---- PET + soft_MA (2 channels, MA-only) -----------------------------------
class nnUNetTrainerFTPD_MAonly_d50_b80(_Base):
    """MA-only: input [PET, soft_MA]; single prior channel (1)."""
    prior_dropout_p = 0.5
    tversky_alpha = 0.2
    tversky_beta = 0.8
    prior_channels = (1,)


# ---- PET + CT + soft_MA (3 channels, MA-only with CT) ----------------------
class nnUNetTrainerFTPD_MAonly_PETCT_d50_b80(_Base):
    """MA-only + CT: input [PET, CT, soft_MA]; single prior channel (2)."""
    prior_dropout_p = 0.5
    tversky_alpha = 0.2
    tversky_beta = 0.8
    prior_channels = (2,)
