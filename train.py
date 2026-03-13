import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

def loss_fn(logits, targets):
    # logits shape: [batch, seq_len, d_vocab]
    # target shape: [batch]
    # We only care about the prediction at the very last token
    final_logits = logits[:, -1, :]
    return nn.functional.cross_entropy(final_logits, targets)

def get_accuracy(logits, targets):
    final_logits = logits[:, -1, :]
    preds = final_logits.argmax(dim=-1)
    return (preds == targets).float().mean()

def run_eval(model, dataloader, device):
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for batch, targets in dataloader:
            batch, targets = batch.to(device), targets.to(device)
            logits = model(batch)
            loss = loss_fn(logits, targets)
            total_loss += loss.item()
            total_acc += get_accuracy(logits, targets).item()
            
    return total_loss / len(dataloader), total_acc / len(dataloader)

def train_baseline(model, train_loader, test_loader, epochs, lr=1e-3, weight_decay=1e-2, device="cuda", save_dir="checkpoints/pretrain"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    os.makedirs(save_dir, exist_ok=True)
    
    model.to(device)
    metrics = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        for batch, targets in train_loader:
            batch, targets = batch.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += get_accuracy(logits, targets).item()
            
        train_loss = total_loss / len(train_loader)
        train_acc = total_acc / len(train_loader)
        
        # Eval
        model.eval()
        test_loss, test_acc = run_eval(model, test_loader, device)
        
        metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}")
            
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    return model

def finetune(model, train_loader, test_loader, task1_loader, steps, save_freq=10, eval_freq=10, lr=1e-4, weight_decay=1e-2, device="cuda", save_dir="checkpoints/finetune"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    os.makedirs(save_dir, exist_ok=True)
    
    model.to(device)
    metrics = []
    train_iter = iter(train_loader)
    
    for step in tqdm(range(steps)):
        model.train()
        try:
            batch, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch, targets = next(train_iter)
            
        batch, targets = batch.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        train_acc = get_accuracy(logits, targets).item()
        
        if step % eval_freq == 0 or step == steps - 1:
            model.eval()
            # Eval Task 2
            test_loss, test_acc = run_eval(model, test_loader, device)
            # Eval Task 1 (catastrophic forgetting)
            task1_loss, task1_acc = run_eval(model, task1_loader, device)
            
            metrics.append({
                "step": step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "task2_test_loss": test_loss,
                "task2_test_acc": test_acc,
                "task1_test_loss": task1_loss,
                "task1_test_acc": task1_acc
            })
            
        if step % save_freq == 0 or step == steps - 1:
            torch.save(model.state_dict(), os.path.join(save_dir, f"step_{step}.pth"))
            
    with open(os.path.join(save_dir, "finetune_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    return model
