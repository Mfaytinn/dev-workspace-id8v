"""Train full-sensor reference models and all Table 1 baselines."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ncam.experiments.calendar_protocol import (
    apply_calendar_spec,
    calendar_plan,
    calendar_spec,
    prepare_calendar_fold,
)
from ncam.experiments.comparison import run_baseline_comparison
from ncam.experiments.preparation import load_dataset
from ncam.experiments.training import fit_fold
from ncam.training import initialize_runtime
from paper_reproduce._common import (
    DEFAULT_OUTPUT_ROOT,
    base_config,
    ensure_data,
    gpu_subprocess_env,
    model_directory,
    run_gpu_jobs,
)


def point_errors(prediction, target):
    residual = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    if not np.isfinite(residual).all():
        raise ValueError("Point predictions and targets must be finite")
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def aggregate_performance(raw):
    # Pool squared errors and absolute errors across folds within each seed.
    per_seed = []
    for (dataset, seed, method), group in raw.groupby(["dataset", "seed", "method"]):
        weights = group.test_rows.to_numpy()
        per_seed.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "rmse": np.sqrt(np.average(group.rmse**2, weights=weights)),
                "mae": np.average(group.mae, weights=weights),
                "test_rows": weights.sum(),
            }
        )
    per_seed = pd.DataFrame(per_seed)
    summary = per_seed.groupby(["dataset", "method"])[["rmse", "mae"]].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{key}_{stat}" for key, stat in summary.columns]
    summary["n_seeds"] = per_seed.groupby(["dataset", "method"]).seed.nunique()
    return per_seed, summary.reset_index()


def worker(output, dataset, fold_id, seed):
    cfg = base_config(dataset)
    ensure_data(dataset, cfg)
    spec = calendar_spec()
    cfg = apply_calendar_spec(cfg, spec)
    frame, cfg = load_dataset(dataset, cfg)
    planned = {
        name: (parts, info)
        for name, parts, info in calendar_plan(frame, cfg, spec=spec)
    }
    parts, info = planned[fold_id]
    device = initialize_runtime(seed)
    cfg["seed"] = seed
    fold = prepare_calendar_fold(parts, cfg, fold_id, device, metadata=info)
    directory = model_directory(output, dataset, fold_id, seed)
    model = fit_fold(fold, device, save_path=directory / "model.pt")
    rows = run_baseline_comparison(
        model,
        fold.test.x,
        fold.test.sensors,
        fold.test.targets,
        fold.frames.test[fold.config["b_cols"]].to_numpy(dtype=np.float32),
        fold.train.sensors,
        fold.config["b_cols"],
        device,
        fold.config,
        fold.config["obs_transform"],
        train_keys=fold.frames.train[["date", "location_id"]],
        test_keys=fold.frames.test[["date", "location_id"]],
    )
    pd.DataFrame(rows).assign(
        dataset=dataset, fold=fold_id, seed=seed, test_rows=len(fold.test.x)
    ).to_csv(directory / "performance.csv", index=False)


def launch(args, gpu, job):
    subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "paper_reproduce.run_temporal_calendar",
            "--output",
            str(args.output),
            "--worker",
            *map(str, job),
        ],
        env=gpu_subprocess_env(gpu),
        check=True,
    )


def load_reference_performance(output, *, datasets, seeds):
    frames = []
    for dataset in datasets:
        for seed in seeds:
            paths = sorted(
                Path(output).glob(f"models/{dataset}/*/seed_{seed}/performance.csv")
            )
            if not paths:
                raise FileNotFoundError(
                    f"Run paper_reproduce.run_temporal_calendar first: {dataset}, seed {seed}"
                )
            frames.extend(
                pd.read_csv(path, float_precision="round_trip") for path in paths
            )
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "calendar")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("seneurcity", "cairsense"),
        default=["seneurcity", "cairsense"],
    )
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        dataset, fold, seed = args.worker
        worker(args.output, dataset, fold, int(seed))
        return
    spec, jobs = calendar_spec(), []
    for dataset in args.datasets:
        cfg = base_config(dataset)
        ensure_data(dataset, cfg)
        frame, cfg = load_dataset(dataset, apply_calendar_spec(cfg, spec))
        jobs.extend(
            (dataset, name, seed)
            for name, _, _ in calendar_plan(frame, cfg, spec=spec)
            for seed in spec["seeds"]
        )
    run_gpu_jobs(jobs, lambda gpu, job: launch(args, gpu, job))
    raw = load_reference_performance(
        args.output, datasets=args.datasets, seeds=spec["seeds"]
    )
    raw.to_csv(args.output / "performance.csv", index=False)
    _, summary = aggregate_performance(raw)
    summary.to_csv(args.output / "table.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
