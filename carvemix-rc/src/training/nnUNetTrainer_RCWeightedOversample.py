import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class nnUNetTrainer_RCWeightedOversample(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.66

    def _build_loss(self):
        ce_weights = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 3.0],
            dtype=torch.float32,
            device=self.device,
        )
        loss = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {"weight": ce_weights},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
