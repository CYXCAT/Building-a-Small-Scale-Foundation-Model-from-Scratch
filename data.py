import torch
from torch.utils.data import Dataset, DataLoader
import os

import config


def _load_sequences():
    if os.path.exists(config.TOKENIZED_DATA_PATH):
        chunks = torch.load(config.TOKENIZED_DATA_PATH)
        out = []
        for c in chunks:
            t = c if isinstance(c, torch.Tensor) else torch.tensor(c)
            t = t.view(-1)
            L = t.size(0)
            if config.SEQ_MIN <= L <= config.SEQ_MAX:
                out.append(t)
            elif L > config.SEQ_MAX:
                for i in range(0, L - config.SEQ_MIN + 1, config.SEQ_LEN):
                    seg = t[i : i + config.SEQ_LEN]
                    if seg.size(0) >= config.SEQ_MIN:
                        out.append(seg)
        return out
    batches = torch.load(config.SAMPLE_DATA_PATH)
    out = []
    for batch in batches:
        ids = batch["input_ids"]
        B, L = ids.shape[0], ids.shape[1]
        for b in range(B):
            seq = ids[b]
            for start in range(0, L - config.SEQ_MIN + 1, config.SEQ_LEN):
                seg = seq[start : start + config.SEQ_LEN]
                if seg.size(0) >= config.SEQ_MIN:
                    out.append(seg)
    return out


class SeqDataset(Dataset):
    def __init__(self, sequences, seq_len, pad_token_id):
        self.sequences = sequences
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx].long()
        if s.dim() > 1:
            s = s.view(-1)
        if s.size(0) >= self.seq_len:
            s = s[: self.seq_len]
        else:
            s = torch.cat(
                [s, torch.full((self.seq_len - s.size(0),), self.pad_token_id, dtype=s.dtype)]
            )
        return s


def get_dataloader(batch_size, shuffle=True):
    sequences = _load_sequences()
    if not sequences:
        raise FileNotFoundError(
            f"Neither {config.TOKENIZED_DATA_PATH} nor {config.SAMPLE_DATA_PATH} found or empty."
        )
    dataset = SeqDataset(
        sequences, config.SEQ_LEN, config.PAD_TOKEN_ID
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=False,
    )
