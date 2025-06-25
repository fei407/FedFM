import math
from typing import List
from omegaconf import DictConfig
from collections import OrderedDict

import torch
from flwr.common.typing import NDArrays

from transformers import AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from .utils import print_state_dict_size


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig, rank_choices: List[int], group_id: str):
    """Load model with appropriate quantization config and other optimizations.
    """
    model = AutoModelForCausalLM.from_pretrained(model_cfg.name)

    for i, rank in enumerate(rank_choices):

        peft_config = LoraConfig(
            r=rank,
            lora_alpha=16,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

        adapter_name = f"group_{i}"
        if i == 0:
            model = get_peft_model(model, peft_config, adapter_name=adapter_name)
        else:
            model.add_adapter(adapter_name, peft_config)

    model.set_adapter(group_id)

    return model

def set_global_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    group_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k and "group_" in k}

    params_dict = zip(group_state_dict, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

    model.load_state_dict(state_dict, strict=False)


def set_local_parameters(model, parameters: NDArrays, group_id: str) -> None:
    """Change the parameters of the model using the given ones."""
    group_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k and "group_" in k}

    params_dict = zip(group_state_dict, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if group_id in k})

    model.load_state_dict(state_dict, strict=False)
    print_state_dict_size(state_dict, label="[Download Commm.]")


def get_local_parameters(model, group_id) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k and group_id in k}
    print_state_dict_size(state_dict, label="[Upload Commm.]")

    return [val.cpu().numpy() for _, val in state_dict.items()]


def get_global_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    group_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k and "group_" in k}

    return [val.cpu().numpy() for _, val in group_state_dict.items()]
