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
    cls_trainable_p = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and ("classifier" in n or "pooler" in n)
    )
    peft_trainable_p = trainable_p - cls_trainable_p
    return {
        "total_p": total_p,
        "trainable_p": trainable_p,
        "cls_trainable_p": cls_trainable_p,
        "peft_trainable_p": peft_trainable_p,
    }

def print_trainable_params(model):
    params_dict = get_trainable_params_dict(model)
    total_p = params_dict["total_p"]
    trainable_p = params_dict["trainable_p"]
    cls_trainable_p = params_dict["cls_trainable_p"]
    peft_trainable_p = params_dict["peft_trainable_p"]
    print(
        f"total trainable params: {trainable_p:,} || "
        f"all params: {total_p:,} || "
        f"trainable%: {100 * trainable_p / total_p:.4f} \n"
        f"peft trainable params: {peft_trainable_p:,} || "
        f"all params: {total_p:,} || "
        f"peft trainable%: {100 * peft_trainable_p / total_p:.4f}"
    )