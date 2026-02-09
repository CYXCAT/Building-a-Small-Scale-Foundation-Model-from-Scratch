import os
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from data import get_dataloader
from model import MiniGPT


def run_train(params, experiment_name="baseline", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = params["batch_size"]
    lr = params["lr"]
    max_epochs = params["max_epochs"]
    n_layers = params["n_layers"]
    embed_dim = params["embed_dim"]
    n_heads = params["n_heads"]
    ffn_mult = params.get("ffn_hidden_mult", 4)

    dataloader = get_dataloader(batch_size=batch_size, shuffle=True)
    model = MiniGPT(
        config.VOCAB_SIZE,
        config.SEQ_LEN,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_mult=ffn_mult,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)

    losses, perplexities = [], []
    for epoch in range(max_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"{experiment_name} epoch {epoch+1}/{max_epochs}")
        for batch in pbar:
            if isinstance(batch, torch.Tensor):
                ids = batch
            else:
                ids = batch["input_ids"]
            ids = ids.to(device)
            logits = model(ids)
            shift_logits = logits[:, :-1].contiguous().view(-1, config.VOCAB_SIZE)
            shift_labels = ids[:, 1:].contiguous().view(-1)
            loss = criterion(shift_logits, shift_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_value = loss.item()
            ppl_value = math.exp(loss_value)
            losses.append(loss_value)
            perplexities.append(ppl_value)
            pbar.set_postfix(loss=loss_value, ppl=ppl_value)

    os.makedirs(config.OUTCOME_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.OUTCOME_DIR, f"mini_gpt_{experiment_name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    return losses, perplexities
