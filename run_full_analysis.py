import os
import torch
from data import get_dataloaders
from model import create_model
from train import train_baseline, finetune
from analysis import plot_phase_transitions, compare_svd, dla, activation_patching_head_out

def main():
    p = 113
    d_model = 128
    n_layers = 1
    n_heads = 4
    d_mlp = 512
    epochs_pretrain = 30
    ft_steps = 1500
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Dataset
    print("Initializing datasets...")
    dataloaders = get_dataloaders(p=p, batch_size=256)
    
    # 2. Model
    print("Creating model...")
    model = create_model(p, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp)
    
    # 3. Pre-train
    print("Starting pre-training...")
    pretrain_dir = "checkpoints/full_pretrain"
    train_baseline(
        model, 
        dataloaders['+']['train'], 
        dataloaders['+']['test'], 
        epochs=epochs_pretrain,
        device=device,
        save_dir=pretrain_dir,
        lr=1e-3,
        weight_decay=1.0  # Encourage stronger grokking style representations
    )
    
    # 4. Fine-tune
    print("Starting fine-tuning...")
    ft_dir = "checkpoints/full_finetune"
    finetune(
        model,
        dataloaders['-']['train'],
        dataloaders['-']['test'],
        dataloaders['+']['test'],
        steps=ft_steps,
        save_freq=50,
        eval_freq=50,
        device=device,
        save_dir=ft_dir,
        lr=1e-4,
        weight_decay=1.0
    )
    
    # 5. Analysis
    print("Running analysis phase transition plot...")
    plot_phase_transitions(ft_dir, "full_phase_transition.png")
    
    print("Running SVD comparison...")
    base_model = create_model(p, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp)
    base_model.load_state_dict(torch.load(f"{pretrain_dir}/final_model.pth"))
    base_model.to(device)
    
    ft_model = create_model(p, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp)
    ft_model.load_state_dict(torch.load(f"{ft_dir}/step_{ft_steps-1}.pth"))
    ft_model.to(device)
    
    for head in range(n_heads):
        compare_svd(base_model, ft_model, layer=0, head=head, save_dir=ft_dir)
    
    print("Running DLA and Activation Patching...")
    # sample sequence: [100, op_sub=114, 20, eq_token=115] -> target: 80
    op_sub = p + 1
    eq_token = p + 2
    tokens = torch.tensor([[100, op_sub, 20, eq_token]]).to(device)
    target_token = 80
    
    dla_res = dla(ft_model, tokens, target_token)
    print("DLA Results:", dla_res)
    
    patch_res = activation_patching_head_out(base_model, ft_model, tokens, target_token)
    print("Activation Patching Results:", patch_res)
    
    print("Full Analysis complete!")
    
if __name__ == "__main__":
    main()
