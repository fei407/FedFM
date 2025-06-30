"""
Minimal web chat interface for SmolLM2-135M-Instruct using Gradio.

Prerequisites:
    pip install gradio torch transformers

Run:
    python smol_chat_web.py --host 0.0.0.0 --port 7860
Then open http://localhost:7860 in your browser.
"""
import copy
import argparse
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmolLM Chat WebUI")
    parser.add_argument("--checkpoint", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                        help="Model checkpoint on Hugging Face hub")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on (cuda | cpu)")
    parser.add_argument("--max-new", dest="max_new", type=int, default=50,
                        help="Max new tokens per response")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling top‑p value")
    parser.add_argument("--template", type=str, default="vicuna_v1.1",
                        help="Chat template")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8080,
                        help="Server port")
    return parser


def load_model(model_ckpt: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})

    model = AutoModelForCausalLM.from_pretrained(model_ckpt).to(device).eval()
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def make_chat_fn(tokenizer, model, device, max_new, temperature, top_p):
    """Return a Gradio-compatible chat function that maintains context."""

    def respond(message, history):
        """Generate a model reply given the latest user message and chat history."""
        # Build messages list for apply_chat_template
        messages = history + [{"role": "user", "content": message}]

        # Convert messages into a single prompt string with assistant tag appended
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Strip the prompt part to get only new tokens
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        reply = reply.replace("<|assistant|>", "").strip()
        return {"role": "assistant", "content": reply}

    return respond

def main():
    args = build_parser().parse_args()
    tokenizer, model = load_model(args.checkpoint, args.device)

    chat_fn = make_chat_fn(
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        max_new=args.max_new,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    demo = gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        title="💬 Chat with SmolLM2 (Federated Efficient Parameter Fined-Tuning)",
        description="This model is fine-tuned on Alpaca-GPT4 via heterogeneous federated learning across edge devices: NVIDIA Jetson AGX Orin, Orin Nano, and Raspberry Pi 5.",
        examples=[
            ["What is gravity?"],
            ["Where is the capital of France?"],
            ["Give me 3 healthy breakfast ideas."]
        ],
    )

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()