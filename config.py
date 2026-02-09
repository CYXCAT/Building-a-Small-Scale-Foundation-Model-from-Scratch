import os

PREPROCESSING_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data-Collection-and-Preprocessing-for-Foundation-Model-Pre-Training",
)
TOKENIZED_DATA_PATH = os.path.join(PREPROCESSING_DIR, "tokenized_data.pt")
SAMPLE_DATA_PATH = os.path.join(PREPROCESSING_DIR, "sample_dataset.pt")

VOCAB_SIZE = 50257
PAD_TOKEN_ID = 50256

SEQ_MIN = 32
SEQ_MAX = 128
SEQ_LEN = 64

# baseline
BASELINE = {
    "lr": 5e-4,
    "batch_size": 32,
    "n_layers": 2,
    "embed_dim": 128,
    "n_heads": 4,
    "max_epochs": 1,
    "ffn_hidden_mult": 4,
}

EXPERIMENTS = [
    {"name": "lr_1e-3", "overrides": {"lr": 1e-3}},
    {"name": "batch_size_16", "overrides": {"batch_size": 16}},
    {"name": "n_layers_1", "overrides": {"n_layers": 1}},
    {"name": "embed_dim_64", "overrides": {"embed_dim": 64}},
]

OUTCOME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outcome")
