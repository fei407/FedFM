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


def get_global_model(model_cfg: DictConfig, rank_choices: List[int]):
    """Load model with appropriate quantization config and other optimizations.
    """
    global_model = AutoModelForCausalLM.from_pretrained(model_cfg.name)

    for i, rank in enumerate(rank_choices):

        peft_config = LoraConfig(
            r=rank,
            lora_alpha=16,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

        adapter_name = f"group_{i}"
        if i == 0:
            global_model = get_peft_model(global_model, peft_config, adapter_name=adapter_name)
        else:
            global_model.add_adapter(adapter_name, peft_config)

    return global_model

def get_local_model(model_cfg: DictConfig, local_rank):
    """Load model with appropriate quantization config and other optimizations.
    """
    model = AutoModelForCausalLM.from_pretrained(model_cfg.name)


    peft_config = LoraConfig(
        r=local_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, peft_config)


def set_global_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)
    print_state_dict_size(state_dict, label="[Download Commm.]")


def set_local_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)
    print_state_dict_size(state_dict, label="[Download Commm.]")


def get_local_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    print_state_dict_size(state_dict, label="[Upload Commm.]")

    return [val.cpu().numpy() for _, val in state_dict.items()]


def get_global_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = model.state_dict()
    group_state_dict = {k: v for k, v in state_dict.items() if "lora_" in k and "group_" in k}
    print_state_dict_size(group_state_dict, label="[Upload Commm.]")

    return [val.cpu().numpy() for _, val in group_state_dict.items()]
