import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, get_peft_model, LoraConfig
import torch.nn.init as init
import tkinter as tk
from tkinter import filedialog
from types import SimpleNamespace
import os
import math
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

args = SimpleNamespace(device="cuda" if torch.cuda.is_available() else "cpu", max_new=256, temperature=0.2, top_p=0.9)

model_name = "HuggingFaceTB/SmolLM2-135M"

def merge_groups(model):
    state = model.state_dict()
    processed_groups_params = []

    for i in range(3):
        received = {
            k: v for k, v in state.items() if "lora_B" in k and f"group_{i}" in k
        }

        for key in list(received.keys()):
            base_name, adapter_name = key.rsplit(".lora_B.", 1)
            key_A = f"{base_name}.lora_A.{adapter_name}"
            key_B = f"{base_name}.lora_B.{adapter_name}"

            if key_B in received and key_A in state:
                A = state[key_A]
                B = received.pop(key_B)
                r = A.shape[0]
                scaling = 16.0 / math.sqrt(r)

                nbias_key = f"{base_name}.lora_nbias.default.weight"
                received[nbias_key] = (B @ A) * scaling

        processed_groups_params.append(received)

    group_0, group_1, group_2 = processed_groups_params

    aggregated_params = {
        k: (group_0[k] + group_1[k] + group_2[k]) / 3
        for k in group_0
    }

    for key in list(aggregated_params.keys()):
        if "lora_nbias" in key:
            base_key = key.replace(".lora_nbias.default", ".base_layer")
            base_weight = state[base_key]
            lora_nbias = aggregated_params[key].to(base_weight.device)
            updated_weight = base_weight + lora_nbias
            base_weight.copy_(updated_weight)
        else:
            print(f"⚠️warning：unexpected {key} is in aggregated_params.")

    return model

@st.cache_resource
def load_raw_model(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return model, tokenizer

def render():
    st.subheader("💬 Chatbot Comparison")

    default_peft_path = "/home/fw407/workspace/results/ffa_vanilla_sqrt/peft_100"
    st.session_state.setdefault("peft_path", default_peft_path)

    path_col, browse_col = st.columns([4, 1])
    with path_col:
        st.session_state["peft_path"] = st.text_input("PEFT model path", st.session_state["peft_path"])
    with browse_col:
        if st.button("📂", key="pick_peft", use_container_width=True):
            root = tk.Tk(); root.withdraw()
            selected = filedialog.askdirectory(initialdir=st.session_state["peft_path"])
            root.destroy()
            if selected:
                st.session_state["peft_path"] = selected

    if "tokenizer" not in st.session_state or "raw_model" not in st.session_state:
        st.session_state["raw_model"], st.session_state["tokenizer"] = load_raw_model(model_name)

    if st.button("Load Fine-Tuned PEFT Model"):
        base_path = st.session_state["peft_path"]
        adapter_names = [f"group_{i}" for i in range(3)]

        missing = [name for name in adapter_names if
                   not os.path.exists(os.path.join(base_path, name, "adapter_config.json"))]
        if missing:
            st.error(f"Missing adapter_config.json in: {', '.join(missing)}")
        else:
            ft_model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(base_path, adapter_names[0]), adapter_name=adapter_names[0]).to(args.device).eval()

            # for name in adapter_names[1:]:
            #     ft_model.load_adapter(os.path.join(base_path, name), adapter_name=name)
            #
            # merge_groups(ft_model)
            # for adapter_name in list(ft_model.peft_config.keys()):
            #     ft_model.delete_adapter(adapter_name)
            #     print(f"✅ Removed adapters: {list(ft_model.peft_config.keys())}")

            st.session_state["finetuned_model"] = ft_model.merge_and_unload()

            # for name, param in ft_model.named_parameters():
            #     print(f"Parameter: {name}, Shape: {param.shape}, Dtype: {param.dtype}, Trainable: {param.requires_grad}, device: {param.device}")

            st.success("Finetuned model with multiple adapters loaded.")

    ###########
    user_input = st.chat_input("Ask a question")

    Q1 = "Give me a list of ten animal names."
    Q2 = "What are some things I should consider when choosing a pet?"
    Q3 = "How can we reduce air pollution?"
    # Q3 = "Generate a list of interesting riddles."

    if st.button(Q1):
        st.session_state["user_input"] = Q1
    if st.button(Q2):
        st.session_state["user_input"] = Q2
    if st.button(Q3):
        st.session_state["user_input"] = Q3

    if "user_input" in st.session_state and st.session_state["user_input"]:
        user_input = st.session_state["user_input"]
        st.session_state["user_input"] = ""

    tokenizer = st.session_state["tokenizer"]
    tokenizer.pad_token = tokenizer.eos_token

    if user_input:
        with st.chat_message("user"):
            st.write("**Question：** "+ user_input)
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        input_len = inputs["input_ids"].shape[-1]

        # Base Model
        raw_outputs = st.session_state["raw_model"].generate(
            **inputs,
            max_new_tokens=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        base_ans = tokenizer.decode(raw_outputs[0][input_len:], skip_special_tokens=True).strip()
        with st.chat_message("assistant"):
            st.write("**Base Model Answer：** "+ base_ans)

        # Finetuned Model
        ft_outputs = st.session_state["finetuned_model"].generate(
            **inputs,
            max_new_tokens=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        ft_ans = tokenizer.decode(ft_outputs[0][input_len:], skip_special_tokens=True).strip()
        with st.chat_message("assistant"):
            st.write("**Finetuned Model Answer：** "+ ft_ans)

if __name__ == "__main__":
    render()