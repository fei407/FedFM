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
    "N/A": -1,
    "person": 792, "bicycle": 93, "car": 206, "motorcycle": 702,  "airplane": 2, "bus": 172, "train": 1114, "truck": 1122, "boat": 117, "traffic_light": 1111,
    "fireplug": 444, "street_sign": 1025, "stop_sign": 1018, "parking_meter": 765, "bench": 89, "bird": 98, "cat": 224, "dog": 377, "horse": 568, "sheep": 942,
    "cow": 79, "elephant": 421, "bear": 75, "zebra": 1201, "giraffe": 495, "hat": 543, "backpack": 33, "umbrella": 1132, "shoe": 947, "eyeglasses": -1,
    "handbag": 34, "tie": 715, "suitcase": 35, "frisbee": 473, "ski": 963, "snowboard": 975, "sports_ball": -1, "kite": 610, "baseball_bat": 57, "baseball_glove": 59,
    "skateboard": 961, "surfboard": 1036, "tennis_racket": 1078, "bottle": 132, "plate": 817, "wineglass": 1189, "cup": 343, "fork": 468, "knife": 614, "spoon": 999,
    "bowl": 138, "banana": 44, "apple": 11, "sandwich": 911, "orange": 734, "broccoli": 153, "carrot": 216, "hotdog": -1, "pizza": 815, "donut": 386,
    "cake": 182, "chair": 231, "sofa": 981, "potted_plant": -1, "bed": 76, "mirror": 693, "dining_table": 366, "window": -1, "desk": 360, "toilet": 1096,
    "door": -1, "television": 1076, "laptop": 630, "mouse": 704, "remote_control": 880, "keyboard": 295, "cellphone": 229, "microwave": 686, "oven": 738, "toaster": 1094,
    "sink": 960, "refrigerator": 420, "blender": 111, "book": 126, "clock": 270, "vase": 1138, "scissors": 922, "teddy_bear": 1070, "hair_dryer": 533, "toothbrush": 1101,
    "tomato": 1098, "onion": 733, "eggplant": 418, "ginger": 494, "garlic": 486, "lettuce": 640, "cucumber": 341, "celery": 228, "potato": 837, "zucchini": 1202,
    "blueberry": 115, "strawberry": 1024, "cherry": 238, "coconut": 282, "peach": 773, "grape": 509, "kiwi_fruit": 640, "lemon": 638, "pineapple": 805, "watermelon": 1171,
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
    new_label2id = {label: i for i, label in enumerate(label_mapping.keys())}
    new_id2label = {i: label for label, i in new_label2id.items()}

    model = AutoModelForObjectDetection.from_pretrained(
        model_cfg.name,
        ignore_mismatched_sizes=True
    )

    num_old_classes = 91
    num_new_classes = 111
    old_head = model.class_embed
    in_features = old_head[0].in_features
    num_layers = len(old_head)

    # 创建一个共享的 Linear 层
    shared_head = nn.Linear(in_features, num_new_classes)

    # 拷贝原权重
    with torch.no_grad():
        shared_head.weight[:num_old_classes] = old_head[0].weight
        shared_head.bias[:num_old_classes] = old_head[0].bias

    # 用共享 Linear 填充 ModuleList
    new_head = nn.ModuleList([shared_head] * num_layers)

    model.class_embed = new_head
    model.config.label2id = new_label2id
    model.config.id2label = new_id2label

    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)
                    and ("q_proj" in n or "v_proj" in n)]  # 举例，只给注意力 q/v 做 LoRA

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
                target_modules=linear_names,
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
                target_modules=linear_names,
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
