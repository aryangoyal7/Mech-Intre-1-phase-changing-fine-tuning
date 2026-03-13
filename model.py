import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer

def create_model(p, d_model=128, n_layers=1, n_heads=4, d_mlp=512):
    # Vocab size: p numbers + 3 special tokens (+, -, =)
    d_vocab = p + 3
    
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_model // n_heads,
        d_mlp=d_mlp,
        d_vocab=d_vocab,
        act_fn="relu",
        normalization_type="LN",
        n_ctx=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        use_attn_result=True
    )
    
    model = HookedTransformer(cfg)
    return model
