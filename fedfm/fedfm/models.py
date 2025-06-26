import math
from typing import List
from omegaconf import DictConfig
from collections import OrderedDict

import torch
import torch.nn.init as init

from flwr.common.typing import NDArrays

from transformers import AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
)

from .utils import print_state_dict_size


def orthogonal_lora_init(model, init_A):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            in_features = module.lora_A["group_0"].weight.shape[1]
            out_features = module.lora_B["group_0"].weight.shape[0]

            with torch.no_grad():
                rand_mat = torch.empty(out_features, in_features)

                if init_A == "uniform":
                    init.uniform_(rand_mat, a=-0.5, b=0.5)
                elif init_A == "gaussian":
                    init.normal_(rand_mat, mean=0.0, std=0.02)
                elif init_A == "kaiming":
                    init.kaiming_uniform_(rand_mat, a=math.sqrt(5))
                else:
                    raise ValueError(f"Unknown init_A method: {init_A}")

                _, _, Vt = torch.linalg.svd(rand_mat, full_matrices=False)

                for adapter_name in module.lora_A.keys():
                    r = module.lora_A[adapter_name].weight.shape[0]
                    module.lora_A[adapter_name].weight.copy_(Vt[:r, :])

def check_lora_A_orthogonality(model, tol=1e-3):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            for adapter_name, A_layer in module.lora_A.items():
                A = A_layer.weight.data
                AA_t = A @ A.T
                identity = torch.eye(A.shape[0], device=A.device)
                deviation = torch.norm(AA_t - identity)

                print(f"[{name}] Adapter: {adapter_name} | ‖A·Aᵀ - I‖ = {deviation:.4e}")
                if deviation < tol:
                    print(" --> ✅ A is approximately orthogonal")
                else:
                    print(" --> ❌ A is NOT orthogonal")

def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig, rank_choices: List[int], group_id: str, peft_name, scaling_method, peft_init):
    """Load model with appropriate quantization config and other optimizations.
    """

    model = AutoModelForCausalLM.from_pretrained(model_cfg.name)

    if peft_name == "fft":
        pass
    elif peft_name == "lora":
        for i, rank in enumerate(rank_choices):
            if scaling_method == "fixed":  # γ = 2
                alpha, use_rslora = 2 * rank, False
            elif scaling_method == "normal":  # γ = α / r
                alpha, use_rslora = 16, False
            elif scaling_method == "sqrt":  # γ = α / √r
                alpha, use_rslora = 16, True
            else:
                raise ValueError(f"Unknown scaling_method '{scaling_method}', choose from ['fixed','normal','sqrt'].")

            peft_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                use_rslora=use_rslora,
            )

            adapter_name = f"group_{i}"
            if i == 0:
                model = get_peft_model(model, peft_config, adapter_name=adapter_name)
            else:
                model.add_adapter(adapter_name, peft_config)

        model.set_adapter(group_id)
    elif peft_name == "ffa":
        for i, rank in enumerate(rank_choices):
            if scaling_method == "fixed":  # γ = 2
                alpha, use_rslora = 2 * rank, False
            elif scaling_method == "normal":  # γ = α / r
                alpha, use_rslora = 16, False
            elif scaling_method == "sqrt":  # γ = α / √r
                alpha, use_rslora = 16, True
            else:
                raise ValueError(f"Unknown scaling_method '{scaling_method}', choose from ['fixed','normal','sqrt'].")

            peft_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                use_rslora=use_rslora,
            )

            adapter_name = f"group_{i}"
            if i == 0:
                model = get_peft_model(model, peft_config, adapter_name=adapter_name)
            else:
                model.add_adapter(adapter_name, peft_config)

        model.set_adapter(group_id)

        if peft_init != "vanilla":
            orthogonal_lora_init(model, peft_init)
            check_lora_A_orthogonality(model)
    else:
        raise ValueError("Unknown local training method.")

    return model

def set_global_parameters(model, parameters: NDArrays, peft_name) -> None:
    """Change the parameters of the model using the given ones."""
    full_state_dict = model.state_dict()
    if peft_name == "fft":
        selected_keys = list(full_state_dict.keys())
    elif peft_name == "lora":
        selected_keys = [k for k in full_state_dict if "lora_" in k and "group_" in k]
    elif peft_name == "ffa":
        selected_keys = [k for k in full_state_dict if "lora_B" in k and "group_" in k]
    else:
        raise ValueError(f"Invalid peft_name: {peft_name}. Please use 'fft', 'lora', or 'ffa'.")

    params_dict = zip(selected_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

    model.load_state_dict(state_dict, strict=False)


def set_local_parameters(model, parameters: NDArrays, group_id: str, peft_name, fl_method) -> None:
    """Change the parameters of the model using the given ones."""
    full_state_dict = model.state_dict()

    if peft_name == "fft":
        selected_keys = list(full_state_dict.keys())
    elif peft_name == "lora":
        if fl_method == "nbias":
            selected_keys = [
                k.replace(".lora_A.group_0", ".base_layer")
                for k in full_state_dict
                if "lora_A.group_0" in k
            ]
        else:
            selected_keys = [k for k in full_state_dict if "lora_" in k and f"group_" in k]
    elif peft_name == "ffa":
        selected_keys = [k for k in full_state_dict if "lora_B" in k and f"group_" in k]
    else:
        raise ValueError(f"Invalid peft_name: {peft_name}. Please use 'fft', 'lora', or 'ffa'.")

    params_dict = zip(selected_keys, parameters)
    if (peft_name == "lora" and fl_method != "nbias") or peft_name == "ffa":
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if group_id in k})
    else:
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)

    print_state_dict_size(state_dict, label="[Download Commm.]")


def get_local_parameters(model, group_id, peft_name) -> NDArrays:
    """Return the parameters of the current net."""
    local_state_dict = model.state_dict()

    if peft_name == "fft":
        state_dict = {k: v.clone().detach() for k, v in local_state_dict.items()}
    elif peft_name == "lora":
        state_dict = {k: v.clone().detach() for k, v in local_state_dict.items() if "lora_" in k and group_id in k}
    elif peft_name == "ffa":
        state_dict = {k: v.clone().detach() for k, v in local_state_dict.items() if "lora_B" in k and group_id in k}
    else:
        raise ValueError(f"Invalid peft_name: {peft_name}. Please use 'fft', 'lora', or 'ffa'.")

    print_state_dict_size(state_dict, label="[Upload Commm.]")

    return [val.cpu().numpy() for _, val in state_dict.items()]


def get_global_parameters(model, peft_name, fl_method) -> NDArrays:
    """Return the parameters of the current net."""
    global_state_dict = model.state_dict()
    if peft_name == "fft":
        group_state_dict =  {k: v.clone().detach() for k, v in global_state_dict.items()}
    elif peft_name == "lora":
        if fl_method == "nbias":
            group_state_dict = {
                (new_key := k.replace(".lora_A.group_0", ".base_layer")): global_state_dict[new_key].clone().detach()
                for k in global_state_dict
                if "lora_A.group_0" in k
            }
        else:
            group_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k and "group_" in k}
    elif peft_name == "ffa":
        group_state_dict = {
            k: v.clone().detach() for k, v in global_state_dict.items() if ("lora_B" in k and f"group_" in k)
        }
    else:
        raise ValueError(f"Invalid peft_name: {peft_name}. Please use 'fft', 'lora', or 'ffa'.")

    return [val.cpu().numpy() for _, val in group_state_dict.items()]
