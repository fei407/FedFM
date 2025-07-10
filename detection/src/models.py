import math
from typing import List
from omegaconf import DictConfig
from collections import OrderedDict

import torch
import torch.nn.init as init

from flwr.common.typing import NDArrays

from transformers import DeformableDetrForObjectDetection, AutoModelForObjectDetection, AutoConfig
from peft import (
    LoraConfig,
    TaskType,
    PeftModel
)

from .utils import print_state_dict_size
import torch.nn as nn

label_mapping= {
    "apple": 11, "avocado": 26, "banana": 44, "blueberry": 115, "cherry": 238, "grape": 509, "kiwi": 612, "lemon": 638,
    "lime": 646, "orange": 734, "peach": 773, "pear":775, "pineapple": 805, "raspberry": 871, "strawberry": 1024, "watermelon": 1171
}

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


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))

# generate fine-tuned model
def get_model(model_cfg: DictConfig, rank_choices: List[int], group_id: str, peft_name, scaling_method):
    """Load model with appropriate quantization config and other optimizations.
    """
    num_labels = len(label_mapping)
    new_label2id = {label: i for i, label in enumerate(label_mapping.keys())}
    new_id2label = {i: label for label, i in new_label2id.items()}

    config = AutoConfig.from_pretrained(
        model_cfg.name,
        num_labels=num_labels,
        label2id=new_label2id,
        id2label=new_id2label,
    )

    model = AutoModelForObjectDetection.from_pretrained(
        model_cfg.name,
        config=config,
        ignore_mismatched_sizes=True
    )

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
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                use_rslora=use_rslora,
            )

            adapter_name = f"group_{i}"
            if i == 0:
                model = PeftModel(model, peft_config, adapter_name=adapter_name)
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
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                use_rslora=use_rslora,
            )

            adapter_name = f"group_{i}"
            if i == 0:
                model = PeftModel(model, peft_config, adapter_name=adapter_name)
            else:
                model.add_adapter(adapter_name, peft_config)

        model.set_adapter(group_id)
        orthogonal_lora_init(model, "uniform")

        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False
    else:
        raise ValueError("Unknown local training method.")

    for name, param in model.named_parameters():
        if "class_embed" in name or "bbox_embed" in name:
            param.requires_grad = True

    return model

# save last parameters
def set_global_parameters(model, parameters: NDArrays, peft_name, fl_method) -> None:
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
            selected_keys = [k for k in full_state_dict if "lora_" in k and "group_" in k]
    elif peft_name == "ffa":
        selected_keys = [k for k in full_state_dict if "lora_B" in k and "group_" in k]
    else:
        raise ValueError(f"Invalid peft_name: {peft_name}. Please use 'fft', 'lora', or 'ffa'.")

    if peft_name != "fft":
        selected_keys += [k for k in full_state_dict if "class_embed" in k or "bbox_embed" in k]

    params_dict = zip(selected_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

    model.load_state_dict(state_dict, strict=False)

# clients receive aggregated model
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

    if peft_name != "fft":
        selected_keys += [k for k in full_state_dict if "class_embed" in k or "bbox_embed" in k]

    params_dict = zip(selected_keys, parameters)
    if (peft_name == "lora" and fl_method != "nbias") or peft_name == "ffa":
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if group_id in k or "class_embed" in k or "bbox_embed" in k})
    else:
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)

    print_state_dict_size(state_dict, label="[Download Commm.]")

# clients send local model
def get_local_parameters(model, group_id, peft_name) -> NDArrays:
    """Return the parameters of the current net."""
    local_state_dict = model.state_dict()

    if peft_name == "fft":
        state_dict = {k: v.clone().detach() for k, v in local_state_dict.items()}
    elif peft_name == "lora":
        state_dict = {k: v.clone().detach() for k, v in local_state_dict.items() if ("lora_" in k and group_id in k ) or "class_embed" in k or "bbox_embed" in k}
    elif peft_name == "ffa":
        state_dict = {k: v.clone().detach() for k, v in local_state_dict.items() if ("lora_B" in k and group_id in k) or "class_embed" in k or "bbox_embed" in k}
    else:
        raise ValueError(f"Invalid peft_name: {peft_name}. Please use 'fft', 'lora', or 'ffa'.")

    print_state_dict_size(state_dict, label="[Upload Commm.]")

    return [val.cpu().numpy() for _, val in state_dict.items()]

# init broadcast
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

            # Also include classification and bbox prediction layers for global update
            for k in global_state_dict:
                if "class_embed" in k or "bbox_embed" in k:
                    group_state_dict[k] = global_state_dict[k].clone().detach()
        else:
            group_state_dict = {k: v.clone().detach() for k, v in model.state_dict().items() if ("lora_" in k and "group_" in k) or "class_embed" in k or "bbox_embed" in k}
    elif peft_name == "ffa":
        group_state_dict = {
            k: v.clone().detach() for k, v in global_state_dict.items() if ("lora_B" in k and f"group_" in k) or "class_embed" in k or "bbox_embed" in k
        }
    else:
        raise ValueError(f"Invalid peft_name: {peft_name}. Please use 'fft', 'lora', or 'ffa'.")

    return [val.cpu().numpy() for _, val in group_state_dict.items()]
