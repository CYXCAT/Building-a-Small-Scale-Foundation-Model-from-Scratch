import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import torch
from train import run_train


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help="experiment name to run, or 'all'",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="1,2,3,4,5",
        help="comma-separated cuda device ids, e.g. 1,2,3,4,5",
    )
    return parser.parse_args()


def _prepare_experiments():
    experiments = [{"name": "baseline", "params": config.BASELINE}]
    for exp in config.EXPERIMENTS:
        params = {**config.BASELINE, **exp["overrides"]}
        experiments.append({"name": exp["name"], "params": params})
    return experiments


def _resolve_devices(num_exps, devices_arg):
    raw_ids = [s.strip() for s in devices_arg.split(",") if s.strip()]
    requested = [f"cuda:{i}" for i in raw_ids] if raw_ids else []
    if not torch.cuda.is_available():
        return ["cpu"] * num_exps
    available = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if not available:
        return ["cpu"] * num_exps
    if not requested:
        requested = available
    usable = [d for d in requested if d in available]
    if not usable:
        usable = available
    return (usable * num_exps)[:num_exps]


def _plot_single(experiment_name, losses, perplexities):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label=experiment_name)
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.legend()
    plt.title(f"Training loss - {experiment_name}")
    plt.text(
        0.01,
        0.98,
        f"experiment: {experiment_name}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
    )
    plt.tight_layout()
    loss_path = os.path.join(config.OUTCOME_DIR, f"loss_curve_{experiment_name}.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved {loss_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(perplexities, label=experiment_name)
    plt.xlabel("Step")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.title(f"Perplexity - {experiment_name}")
    plt.text(
        0.01,
        0.98,
        f"experiment: {experiment_name}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
    )
    plt.tight_layout()
    ppl_path = os.path.join(config.OUTCOME_DIR, f"perplexity_curve_{experiment_name}.png")
    plt.savefig(ppl_path)
    plt.close()
    print(f"Saved {ppl_path}")


def _plot_comparison(results):
    if len(results) <= 1:
        return
    plt.figure(figsize=(8, 5))
    for label, data in results.items():
        plt.plot(data["losses"], label=label)
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.legend()
    plt.title("Training loss comparison")
    plt.tight_layout()
    loss_path = os.path.join(config.OUTCOME_DIR, "loss_curves.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved {loss_path}")

    plt.figure(figsize=(8, 5))
    for label, data in results.items():
        plt.plot(data["perplexities"], label=label)
    plt.xlabel("Step")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.title("Perplexity comparison")
    plt.tight_layout()
    ppl_path = os.path.join(config.OUTCOME_DIR, "perplexity_curves.png")
    plt.savefig(ppl_path)
    plt.close()
    print(f"Saved {ppl_path}")


def _write_tables(results):
    rows = []
    for label, data in results.items():
        rows.append({
            "experiment": label,
            "final_loss": data["losses"][-1],
            "final_perplexity": data["perplexities"][-1],
        })
    lines = ["experiment\tfinal_loss\tfinal_perplexity"]
    for r in rows:
        lines.append(f"{r['experiment']}\t{r['final_loss']:.6f}\t{r['final_perplexity']:.4f}")
    table_str = "\n".join(lines)
    print("\nComparison table:")
    print(table_str)
    table_path = os.path.join(config.OUTCOME_DIR, "results_table.txt")
    with open(table_path, "w") as f:
        f.write(table_str)
    print(f"\nSaved {table_path}")

    csv_path = os.path.join(config.OUTCOME_DIR, "results_table.csv")
    with open(csv_path, "w") as f:
        f.write("experiment,final_loss,final_perplexity\n")
        for r in rows:
            f.write(f"{r['experiment']},{r['final_loss']:.6f},{r['final_perplexity']:.4f}\n")
    print(f"Saved {csv_path}")


def main():
    args = _parse_args()
    os.makedirs(config.OUTCOME_DIR, exist_ok=True)
    experiments = _prepare_experiments()
    if args.only != "all":
        experiments = [e for e in experiments if e["name"] == args.only]
        if not experiments:
            raise ValueError(f"Unknown experiment name: {args.only}")

    results = {}
    devices = _resolve_devices(len(experiments), args.devices)
    for idx, exp in enumerate(experiments):
        name = exp["name"]
        print(f"Running experiment: {name} on {devices[idx]}")
        losses, ppl = run_train(
            exp["params"],
            experiment_name=name,
            device=devices[idx],
        )
        results[name] = {"losses": losses, "perplexities": ppl}
        _plot_single(name, losses, ppl)

    _plot_comparison(results)
    _write_tables(results)


if __name__ == "__main__":
    main()
