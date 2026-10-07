"""Anchored/unanchored validation trajectories using Table 1's resolved settings."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

from ncam.checkpoints import load_model
from ncam.experiments.training import fit_fold
from ncam.training import initialize_runtime
from paper_reproduce._common import launch_job, parse_args, run_gpu_jobs
from paper_reproduce._sensitivity import SensitivityExperiments, configure_parser, variant_fold


def configure_plot_parser(parser):
    configure_parser(parser)
    parser.add_argument("--folds", nargs="+", help="Select existing calendar fold IDs")
    parser.add_argument(
        "--plot-only", action="store_true", help="Replot trajectories.csv"
    )
    parser.add_argument("--job", help=argparse.SUPPRESS)


def run_job(job):
    _, reference = load_model(job["source"])
    device = initialize_runtime(int(reference["seed"]))
    fold = variant_fold(
        reference,
        {"model": job["model"], "record_validation_predictions": True},
        device,
    )
    model = fit_fold(fold, device)
    history = model.training_history
    target = fold.val.targets.cpu().numpy().astype(np.float64)
    labeled = np.isfinite(target)
    if not labeled.any():
        raise ValueError("No observed validation references to plot")
    rows = []
    for epoch, nll, point in zip(
        history["epochs"],
        history["val_loss"],
        history["validation_predictions"],
        strict=True,
    ):
        residual = point[labeled].astype(np.float64) - target[labeled]
        rows.append(
            {
                "dataset": reference["dataset"],
                "fold": reference.get("calendar_fold_id", "random"),
                "seed": reference["seed"],
                "model": job["model"],
                "epoch": epoch,
                "val_sensor_nll": nll,
                "val_reference_mae": float(np.abs(residual).mean()),
                "val_reference_rmse": float(np.sqrt(np.square(residual).mean())),
                "val_sensor_rows": len(target),
                "val_reference_rows": int(labeled.sum()),
            }
        )
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "curve.csv", index=False)


def plot_curves(curves, output):
    """Keep each fold/seed visible; do not select an illustrative run by MAE."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for (dataset, fold, seed), group in curves.groupby(
        ["dataset", "fold", "seed"], dropna=False, sort=False
    ):
        if set(group.model) != {"anchored", "unanchored"}:
            raise ValueError("Every plot requires both anchored and unanchored traces")
        figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
        for model, color in (("anchored", "#2369a1"), ("unanchored", "#c44d37")):
            trace = group.loc[group.model.eq(model)].sort_values("epoch")
            for axis, metric in zip(
                axes, ("val_sensor_nll", "val_reference_mae"), strict=True
            ):
                axis.plot(
                    trace.epoch, trace[metric], label=model.capitalize(), color=color
                )
        for axis, label in zip(
            axes, ("Validation sensor NLL", "Validation reference MAE"), strict=True
        ):
            axis.set(xlabel="Epoch", ylabel=label)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
        fold_label = "random split" if pd.isna(fold) else str(fold)
        figure.suptitle(f"{dataset} · {fold_label} · seed {int(seed)}", fontsize=11)
        stem = f"{dataset}_{'random' if pd.isna(fold) else fold}_seed_{int(seed)}"
        figure.savefig(output / f"{stem}.pdf")
        figure.savefig(output / f"{stem}.png", dpi=200)
        plt.close(figure)


def main():
    args = parse_args(
        __file__, __doc__, seeds=(42,), configure_parser=configure_plot_parser
    )
    if args.job:
        run_job(json.loads(args.job))
        return
    if args.plot_only:
        plot_curves(pd.read_csv(args.output / "trajectories.csv"), args.output)
        return
    reference = SensitivityExperiments(args)
    available = {key[1] for key in reference.configs if key[1] is not None}
    if args.folds and set(args.folds) - available:
        raise ValueError("Requested folds must exist in Table 1")
    jobs = []
    for key, source in reference.sources.items():
        if args.folds and key[1] not in args.folds:
            continue
        for model in ("anchored", "unanchored"):
            directory = (
                args.output / key[0] / (key[1] or "random") / f"seed_{key[2]}" / model
            )
            jobs.append(
                {"source": str(source), "model": model, "directory": str(directory)}
            )
    run_gpu_jobs(
        jobs, lambda gpu, job: launch_job("paper_reproduce.plot_non_identifiability", gpu, job)
    )
    curves = pd.concat(
        [pd.read_csv(Path(job["directory"]) / "curve.csv") for job in jobs],
        ignore_index=True,
    )
    curves.to_csv(args.output / "trajectories.csv", index=False)
    plot_curves(curves, args.output)


if __name__ == "__main__":
    main()
