"""Table 1: median point errors on the reference models and retained baselines."""

import argparse
import shutil

import numpy as np
import pandas as pd

from ncam.experiments.comparison import run_baseline_comparison
from ncam.experiments.config import load_config
from ncam.experiments.preparation import load_dataset, prepare_toy_fold
from ncam.experiments.training import fit_fold
from ncam.training import initialize_runtime
from paper_reproduce._common import (
    base_config,
    configure_calendar_parser,
    ensure_data,
    launch_job,
    model_directory,
    parse_args,
    run_gpu_jobs,
    save_results,
    write_table,
)
from paper_reproduce.run_temporal_calendar import (
    aggregate_performance,
    load_reference_performance,
)


def configure_parser(parser):
    configure_calendar_parser(parser)
    parser.add_argument("--job", help=argparse.SUPPRESS)


def run_toy(job):
    cfg = dict(
        base_config("toy_spatial"),
        seed=job["seed"],
        n_splits=1,
        veli=load_config("configs/tables/veli.yaml")["veli"],
    )
    ensure_data("toy_spatial", cfg)
    device = initialize_runtime(cfg["seed"])
    frame, cfg = load_dataset("toy_spatial", cfg)
    fold = prepare_toy_fold(frame, cfg, device)
    directory = model_directory(job["output"], "toy_spatial", None, cfg["seed"])
    model = fit_fold(fold, device, save_path=directory / "model.pt")
    rows = run_baseline_comparison(
        model,
        fold.test.x,
        fold.test.sensors,
        fold.test.targets,
        fold.frames.test[cfg["b_cols"]].to_numpy(dtype=np.float32),
        fold.train.sensors,
        cfg["b_cols"],
        device,
        fold.config,
        cfg["obs_transform"],
    )
    pd.DataFrame(rows).assign(
        dataset="toy_spatial",
        seed=cfg["seed"],
        fold="random",
        test_rows=len(fold.test.x),
    ).to_csv(directory / "performance.csv", index=False)


def main():
    import json

    args = parse_args(__file__, __doc__, configure_parser=configure_parser)
    if args.job:
        run_toy(json.loads(args.job))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    real = [name for name in args.datasets if name != "toy_spatial"]
    if real:
        frames.append(
            load_reference_performance(
                args.calendar_study, datasets=real, seeds=args.seeds
            )
        )
        for dataset in real:
            for seed in args.seeds:
                for source in sorted(
                    args.calendar_study.glob(f"models/{dataset}/*/seed_{seed}/model.pt")
                ):
                    destination = model_directory(
                        args.output, dataset, source.parent.parent.name, seed
                    )
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination / "model.pt")
                    shutil.copyfile(
                        source.with_name("performance.csv"),
                        destination / "performance.csv",
                    )
    if "toy_spatial" in args.datasets:
        jobs = [{"seed": seed, "output": str(args.output)} for seed in args.seeds]
        run_gpu_jobs(
            jobs,
            lambda gpu, job: launch_job(
                "paper_reproduce.table1_predictivePerformance", gpu, job
            ),
        )
        frames.extend(
            pd.read_csv(
                model_directory(args.output, "toy_spatial", None, seed)
                / "performance.csv",
                float_precision="round_trip",
            )
            for seed in args.seeds
        )
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(args.output / "fold_results.csv", index=False)
    per_seed, _ = aggregate_performance(raw)
    summary = save_results(args, [per_seed], ["dataset", "method"])
    summary = summary.loc[~summary.method.str.startswith("Single-sensor")].copy()
    reference = summary.loc[summary.method.eq("anchored")].set_index("dataset")
    for metric in ("rmse", "mae"):
        summary[f"{metric}_delta_pct"] = (
            summary[f"{metric}_mean"] / summary.dataset.map(reference[f"{metric}_mean"])
            - 1
        ) * 100
    write_table(args, summary)


if __name__ == "__main__":
    main()
