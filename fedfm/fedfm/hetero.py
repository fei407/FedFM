import time
import torch
import math
from typing import List, Dict


def solve_linear(A: torch.Tensor, W: torch.Tensor, method: str = 'least_squares',
                 ridge_lamda: float = 1e-5) -> torch.Tensor:
    if method == 'simple':
        B = A.T @ W
    elif method == 'least_squares':
        B = torch.linalg.lstsq(A, W).solution
    elif method == 'pinv':
        B = torch.pinverse(A) @ W
    elif method == 'ridge':
        n_features = A.shape[1]
        ATA = A.T @ A
        I = torch.eye(n_features, device=A.device, dtype=A.dtype)
        B = torch.linalg.solve(ATA + ridge_lamda * I, A.T @ W)
    else:
        raise ValueError(
            f"Unknown method '{method}', choose from "
            "['least_squares','pinv','ridge'].")
    return B


def custom_aggregate(client_param_dicts: List[Dict[str, torch.Tensor]], num_examples_list: List[int], global_model, fl_method, peft_name, scaling_method, rmax) -> Dict[str, torch.Tensor]:

    total_dataset_size = sum(num_examples_list)
    global_state = global_model.state_dict()
    processed_clients_params = []

    # for i, client_params in enumerate(client_param_dicts):
    #     print(f"\nClient {i} parameter shapes:")
    #     for k, v in client_params.items():
    #         print(f"  {k}: {v.shape}")

    for client_dicts in client_param_dicts:
        received = client_dicts.copy()

        for key in list(received.keys()):
            if "lora_B" in key:
                base_name, adapter_name = key.split(".lora_B.")
                key_A = f"{base_name}.lora_A.{adapter_name}"
                key_B = f"{base_name}.lora_B.{adapter_name}"

                if fl_method in ["nbias", "svd"] and peft_name == "lora":
                    if key_A in received and key_B in received:
                        A = received.pop(key_A)
                        B = received.pop(key_B)
                        r = A.shape[0]

                        if scaling_method == "fixed":
                            scaling = 2
                        elif scaling_method == "normal":
                            scaling = 16.0 / r
                        elif scaling_method == "sqrt":
                            scaling = 16.0 / math.sqrt(r)
                        else:
                            raise ValueError(
                                f"Unknown scaling_method '{scaling_method}', choose from ['fixed','normal','sqrt'].")

                        received[f"{base_name}.lora_nbias.default.weight"] = (B @ A) * scaling
                elif fl_method == "zero-padding" and peft_name == "lora":
                    if key_A in received and key_B in received:
                        A = received.pop(key_A)
                        B = received.pop(key_B)
                        r = A.shape[0]

                        new_key_A = f"{base_name}.lora_A.weight"
                        new_key_B = f"{base_name}.lora_B.weight"

                        if r < rmax:
                            pad_A = torch.zeros((rmax - r, A.shape[1]), dtype=A.dtype, device=A.device)
                            A = torch.cat([A, pad_A], dim=0)  # [rmax × in_dim]

                            pad_B = torch.zeros((B.shape[0], rmax - r), dtype=B.dtype, device=B.device)
                            B = torch.cat([B, pad_B], dim=1)  # [out_dim × rmax]

                        if scaling_method == "fixed":
                            scaling = 2
                        elif scaling_method == "normal":
                            scaling = 16.0 / r
                        elif scaling_method == "sqrt":
                            scaling = 16.0 / math.sqrt(r)
                        else:
                            raise ValueError(
                                f"Unknown scaling_method '{scaling_method}', choose from ['fixed','normal','sqrt'].")

                        received[new_key_A] = A
                        received[new_key_B] = B * scaling
                elif peft_name == "ffa":
                    if key_B in received and key_A in global_state:
                        A_init = global_state[key_A]
                        B = received.pop(key_B)
                        r = A_init.shape[0]

                        if scaling_method == "fixed":
                            scaling = 2
                        elif scaling_method == "normal":
                            scaling = 16.0 / r
                        elif scaling_method == "sqrt":
                            scaling = 16.0 / math.sqrt(r)
                        else:
                            raise ValueError(
                                f"Unknown scaling_method '{scaling_method}', choose from ['fixed','normal','sqrt'].")

                        received[f"{base_name}.lora_nbias.default.weight"] = (B @ A_init) * scaling

        processed_clients_params.append(received)

    # c1, c2 = processed_clients_params[:2]
    #
    # ratios = []
    # for k, v in c2.items():
    #     if "lora_nbias" in k:
    #         r = v.norm().item() / c1[k].norm().item()
    #         ratios.append((k, r))
    #         print(f"{k}: {r}")
    #
    # avg = sum(r for _, r in ratios) / len(ratios) if ratios else 0
    # print(f"average: {avg:.2f}")

    aggregated_params = {k: torch.zeros_like(v) for k, v in processed_clients_params[0].items()}

    # for i, client_params in enumerate(processed_clients_params):
    #     print(f"\nClient {i} parameter shapes:")
    #     for k, v in client_params.items():
    #         print(f"  {k}: {v.shape}")

    for client_params, dataset_size in zip(processed_clients_params, num_examples_list):
        weight = dataset_size / total_dataset_size
        for k, v in client_params.items():
            aggregated_params[k] += v * weight

    return aggregated_params

def update_global_model(global_model, aggregated_params, fl_method,  peft_name, rank_choices, scaling_method):
    if peft_name == "lora" and fl_method == "zero-padding":
        start_time = time.time()
        reconstructed_params = {}
        for key in list(aggregated_params.keys()):
            if "lora_B" in key:
                base_name, _ = key.split(".lora_B.")
                key_B = key
                key_A = key.replace(".lora_B", ".lora_A")

                A_full = aggregated_params[key_A]
                B_full = aggregated_params[key_B]

                for i, r in enumerate(rank_choices):
                    adapter_name = f"group_{i}"

                    new_key_A = f"{base_name}.lora_A.{adapter_name}.weight"
                    new_key_B = f"{base_name}.lora_B.{adapter_name}.weight"

                    if scaling_method == "fixed":
                        scaling = 2
                    elif scaling_method == "normal":
                        scaling = 16.0 / r
                    elif scaling_method == "sqrt":
                        scaling = 16.0 / math.sqrt(r)
                    else:
                        raise ValueError(
                            f"Unknown scaling_method '{scaling_method}', choose from ['fixed','normal','sqrt'].")

                    reconstructed_params[new_key_A] = A_full[:r, :]
                    reconstructed_params[new_key_B] = B_full[:, :r] / scaling
            elif "classifier" in key:
                reconstructed_params[key] = aggregated_params[key]
            elif "lora_A" in key:
                continue
            else:
                print(f"⚠️warning：unexpected {key} is in aggregated_params.")
        global_model.load_state_dict(reconstructed_params, strict=False)
        end_time = time.time()
        print(f"[Zero-Padding] Total running time: {end_time - start_time:.4f} seconds")
    elif peft_name == "lora" and fl_method == "nbias":
        for key in list(aggregated_params.keys()):
            if "lora_nbias" in key:
                base_key = key.replace(".lora_nbias.default", ".base_layer")
                base_weight = global_model.state_dict()[base_key]
                lora_nbias = aggregated_params[key].to(base_weight.device)
                updated_weight = base_weight + lora_nbias
                base_weight.copy_(updated_weight)
            elif "classifier" in key:
                cls_params = aggregated_params[key]
                global_model.state_dict()[key].copy_(cls_params)
            else:
                print(f"⚠️warning：unexpected {key} is in aggregated_params.")
    elif peft_name == "lora" and fl_method == "svd":
        start_time = time.time()
        reconstructed_params = {}

        for key, value in aggregated_params.items():
            if "lora_nbias" in key:
                base_name = key.replace(".lora_nbias.default.weight", "")

                U, S, Vt = torch.linalg.svd(value, full_matrices=False)

                for i, r in enumerate(rank_choices):
                    adapter_name = f"group_{i}"

                    key_A = f"{base_name}.lora_A.{adapter_name}.weight"
                    key_B = f"{base_name}.lora_B.{adapter_name}.weight"

                    lora_B = U[:, :r] @ torch.diag(S[:r])  # [out_dim, r]
                    lora_A = Vt[:r, :]  # [r, in_dim]

                    if scaling_method == "fixed":
                        scaling = 2
                    elif scaling_method == "normal":
                        scaling = 16.0 / r
                    elif scaling_method == "sqrt":
                        scaling = 16.0 / math.sqrt(r)
                    else:
                        raise ValueError(
                            f"Unknown scaling_method '{scaling_method}', choose from ['fixed','normal','sqrt'].")

                    reconstructed_params[key_A] = lora_A
                    reconstructed_params[key_B] = lora_B / scaling
            elif "classifier" in key:
                reconstructed_params[key] = value
            else:
                print(f"⚠️warning：unexpected {key} is in aggregated_params.")

        global_model.load_state_dict(reconstructed_params, strict=False)
        end_time = time.time()
        print(f"[SVD] : Total running time: {end_time - start_time:.4f} seconds")
    elif peft_name == "ffa":
        start_time = time.time()
        solved_params = {}
        global_state = global_model.state_dict()

        for key, value in aggregated_params.items():
            if ".lora_nbias.default.weight" not in key:
                solved_params[key] = value
                continue

            base_name = key.replace(".lora_nbias.default.weight", "")

            for i, r in enumerate(rank_choices):
                adapter_name = f"group_{i}"

                key_A = f"{base_name}.lora_A.{adapter_name}.weight"
                key_B = f"{base_name}.lora_B.{adapter_name}.weight"

                A_init = global_state[key_A]

                if scaling_method == "fixed":
                    scaling = 2
                elif scaling_method == "normal":
                    scaling = 16.0 / r
                elif scaling_method == "sqrt":
                    scaling = 16.0 / math.sqrt(r)
                else:
                    raise ValueError(
                        f"Unknown scaling_method '{scaling_method}', choose from ['fixed','normal','sqrt'].")

                # print(f"solve_method: {solve_method}")
                B_T = solve_linear(A_init.T, value.T, "simple", 1e-5)
                B = B_T.T

                solved_params[key_B] = B / scaling

        global_model.load_state_dict(solved_params, strict=False)
        end_time = time.time()
        print(f"[Solve_B] : Total running time: {end_time - start_time:.4f} seconds")
    else:
        missing_keys, unexpected_keys = global_model.load_state_dict(aggregated_params, strict=False)

        if unexpected_keys:
            print(
                f"⚠️warning：unexpected {unexpected_keys} is in aggregated_params, and it can't be found in global_model!")

    return global_model
