# Adopted from LLaVA trainer
import os, logging
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, ALL_LAYERNORM_LAYERS, logger, TRAINER_STATE_NAME

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

class VideoLLaMA2Trainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return super()._get_train_sampler()
    
    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)], "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)], "weight_decay": 0.0},
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer