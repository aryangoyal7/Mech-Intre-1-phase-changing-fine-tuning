import torch
import matplotlib.pyplot as plt
import json
import os
from transformer_lens import HookedTransformer

def load_metrics(finetune_dir):
    with open(os.path.join(finetune_dir, "finetune_metrics.json"), "r") as f:
        return json.load(f)

def plot_phase_transitions(finetune_dir, save_path="phase_transition.png"):
    metrics = load_metrics(finetune_dir)
    steps = [m["step"] for m in metrics]
    task2_loss = [m["task2_test_loss"] for m in metrics]
    task1_loss = [m["task1_test_loss"] for m in metrics]
    task2_acc = [m["task2_test_acc"] for m in metrics]
    task1_acc = [m["task1_test_acc"] for m in metrics]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, task2_loss, label="Task 2 (Fine-tune) Loss")
    plt.plot(steps, task1_loss, label="Task 1 (Pre-train) Loss")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Losses During Fine-tuning")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(steps, task2_acc, label="Task 2 (Fine-tune) Acc")
    plt.plot(steps, task1_acc, label="Task 1 (Pre-train) Acc")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy During Fine-tuning")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(finetune_dir, save_path))
    plt.close()

def get_circuit_svd(model, layer=0, head=0):
    W_O = model.W_O[layer, head]
    W_V = model.W_V[layer, head]
    W_OV = W_V @ W_O
    
    W_Q = model.W_Q[layer, head]
    W_K = model.W_K[layer, head]
    W_QK = W_Q @ W_K.transpose(-1, -2)
    
    u_ov, s_ov, v_ov = torch.svd(W_OV)
    u_qk, s_qk, v_qk = torch.svd(W_QK)
    
    return s_ov, s_qk

def compare_svd(base_model, ft_model, layer=0, head=0, save_dir="."):
    s_ov_base, s_qk_base = get_circuit_svd(base_model, layer, head)
    s_ov_ft, s_qk_ft = get_circuit_svd(ft_model, layer, head)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(s_ov_base.detach().cpu().numpy(), label="Base")
    plt.plot(s_ov_ft.detach().cpu().numpy(), label="Fine-tuned")
    plt.title(f"OV Circuit SVs L{layer}H{head}")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(s_qk_base.detach().cpu().numpy(), label="Base")
    plt.plot(s_qk_ft.detach().cpu().numpy(), label="Fine-tuned")
    plt.title(f"QK Circuit SVs L{layer}H{head}")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"svd_L{layer}H{head}.png"))
    plt.close()

def dla(model, tokens, target_token):
    logits, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: name.endswith("hook_result")
    )
    unembed_direction = model.W_U[:, target_token]
    
    results = {}
    for layer in range(model.cfg.n_layers):
        head_results = cache["result", layer][0, -1] # [n_heads, d_model]
        for head in range(model.cfg.n_heads):
            head_out = head_results[head]
            contribution = (head_out * unembed_direction).sum().item()
            results[f"L{layer}H{head}"] = contribution
            
    return results

def activation_patching_head_out(base_model, ft_model, tokens, target_token):
    _, ft_cache = ft_model.run_with_cache(tokens)
    
    def head_patch_hook(z, hook, head_idx):
        z[:, :, head_idx, :] = ft_cache[hook.name][:, :, head_idx, :]
        return z
        
    patching_results = {}
    
    clean_logits = base_model(tokens)
    clean_prob = clean_logits[0, -1].softmax(-1)[target_token].item()
    
    for layer in range(base_model.cfg.n_layers):
        for head in range(base_model.cfg.n_heads):
            hook_name = f"blocks.{layer}.attn.hook_z"
            patched_logits = base_model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, lambda z, hook, h=head: head_patch_hook(z, hook, h))]
            )
            patched_prob = patched_logits[0, -1].softmax(-1)[target_token].item()
            patching_results[f"L{layer}H{head}"] = patched_prob - clean_prob
            
    return patching_results
