# utils.py
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_trainable_params_dict(model):
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_p": total_p,
        "trainable_p": trainable_p,
    }

def print_trainable_params(model):
    params_dict = get_trainable_params_dict(model)
    total_p = params_dict["total_p"]
    trainable_p = params_dict["trainable_p"]
    print(
        f"total trainable params: {trainable_p:,} || "
        f"all params: {total_p:,} || "
        f"trainable%: {100 * trainable_p / total_p:.4f} \n"
    )

def print_state_dict_size(state_dict, label="[Comm.]"):
    total_bytes = 0
    for name, tensor in state_dict.items():
        size_mb = tensor.numel() * tensor.element_size() / 1e6
        total_bytes += size_mb
    print(f"INFO :      Total size of {label}: {total_bytes:.2f} MB")
