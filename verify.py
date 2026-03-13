import os
import torch
import shutil
from data import get_dataloaders
from model import create_model
from train import train_baseline, finetune
from analysis import plot_phase_transitions, compare_svd, dla, activation_patching_head_out

def main():
    p = 11
    d_model = 16
    n_layers = 1
    n_heads = 2
    d_mlp = 64
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Dataset
    print("Initializing datasets...")
    dataloaders = get_dataloaders(p=p, batch_size=32)
    
    # 2. Model
    print("Creating model...")
    model = create_model(p, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp)
    
    # 3. Pre-train
    print("Starting pre-training...")
    pretrain_dir = "checkpoints/verify_pretrain"
    train_baseline(
        model, 
        dataloaders['+']['train'], 
        dataloaders['+']['test'], 
        epochs=5,
        device=device,
        save_dir=pretrain_dir
    )
    
    # 4. Fine-tune
    print("Starting fine-tuning...")
    ft_dir = "checkpoints/verify_finetune"
    finetune(
        model,
        dataloaders['-']['train'],
        dataloaders['-']['test'],
        dataloaders['+']['test'],
        steps=50,
        save_freq=10,
        eval_freq=10,
        device=device,
        save_dir=ft_dir
    )
    
    # 5. Analysis
    print("Running analysis phase transition plot...")
    plot_phase_transitions(ft_dir, "test_phase_transition.png")
    
    print("Running SVD comparison...")
    base_model = create_model(p, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp)
    base_model.load_state_dict(torch.load(f"{pretrain_dir}/final_model.pth"))
    base_model.to(device)
    
    ft_model = create_model(p, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp)
    ft_model.load_state_dict(torch.load(f"{ft_dir}/step_49.pth"))
    ft_model.to(device)
    
    compare_svd(base_model, ft_model, layer=0, head=0, save_dir=ft_dir)
    
    print("Running DLA and Activation Patching...")
    # sample sequence: [2, op_sub=12, 1, eq_token=13] -> target: 1
    op_sub = p + 1
    eq_token = p + 2
    tokens = torch.tensor([[2, op_sub, 1, eq_token]]).to(device)
    target_token = 1
    
    dla_res = dla(ft_model, tokens, target_token)
    print("DLA Results:", dla_res)
    
    patch_res = activation_patching_head_out(base_model, ft_model, tokens, target_token)
    print("Activation Patching Results:", patch_res)
    
    print("Verification complete!")
    
if __name__ == "__main__":
    main()
