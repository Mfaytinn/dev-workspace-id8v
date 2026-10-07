"""Refit changed settings using Table 1's reference configurations and rows."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ncam.checkpoints import load_model
from ncam.evaluation import posterior_point_prediction
from ncam.experiments.calendar_protocol import prepare_calendar_fold, reconstruct_fold
from ncam.experiments.preparation import load_dataset, prepare_toy_fold
from ncam.experiments.training import fit_fold
from ncam.training import initialize_runtime
from paper_reproduce._common import (
    DEFAULT_TABLE1_RESULTS,
    launch_job,
    run_gpu_jobs,
    save_results,
)
from paper_reproduce.run_temporal_calendar import aggregate_performance, point_errors

DEFAULT_SEEDS = (0, 1, 42)


def configure_parser(parser):
    parser.add_argument(
        "--table1-results",
        type=Path,
        default=DEFAULT_TABLE1_RESULTS,
        help="Table 1 results and reference model files",
    )


def variant_config(reference, overrides):
    fitted = {
        "x_norm_mean",
        "x_norm_std",
        "log_floor_per_sensor",
        "coordinate_bounds",
        "coordinate_clipping",
        "veli",
    }
    cfg = {key: value for key, value in reference.items() if key not in fitted}
    cfg.update(overrides)
    cfg.update(model_candidates={}, baselines=False)
    if "anchor_sensor" in overrides:
        cfg["anchor_idx"] = cfg["b_cols"].index(cfg["anchor_sensor"])
    return cfg


def variant_fold(reference, overrides, device):
    original = reconstruct_fold(reference, device)
    cfg = variant_config(reference, overrides)
    if reference["dataset"] == "toy_spatial":
        frame, cfg = load_dataset("toy_spatial", cfg, saved_config=cfg)
        return prepare_toy_fold(frame, cfg, device)
    info = {
        "test_end": reference["calendar_test_end"],
        "windows": reference["calendar_windows"],
    }
    return prepare_calendar_fold(
        original.frames, cfg, reference["calendar_fold_id"], device, metadata=info
    )


def select_methods(frame, anchor):
    raw = f"Single-sensor baseline ({anchor})"
    selected = frame.loc[frame.method.isin(("anchored", raw))].copy()
    selected["method"] = selected.method.replace({raw: "raw_anchor"})
    return selected


class SensitivityExperiments:
    def __init__(self, args):
        self.args, self.configs, self.sources, self.references = args, {}, {}, {}
        args.output.mkdir(parents=True, exist_ok=True)
        for dataset in args.datasets:
            for seed in args.seeds:
                paths = sorted(
                    args.table1_results.glob(f"models/{dataset}/*/seed_{seed}/model.pt")
                )
                if not paths:
                    raise FileNotFoundError(
                        "Run Table 1 first, or provide --table1-results"
                    )
                for path in paths:
                    _, cfg = load_model(path)
                    fold = cfg.get("calendar_fold_id")
                    key = (dataset, fold, seed)
                    self.configs[key] = cfg
                    self.sources[key] = path
                    self.references[key] = pd.read_csv(
                        path.with_name("performance.csv"), float_precision="round_trip"
                    )

    def sensors(self, dataset):
        return next(
            cfg["b_cols"] for key, cfg in self.configs.items() if key[0] == dataset
        )

    def evaluate(self, cells):
        frames, jobs = [], []
        for dataset, cell, seed, overrides, labels in cells:
            for key, cfg in self.configs.items():
                if key[0] != dataset or key[2] != seed:
                    continue
                if all(
                    cfg.get(name, 1.0 if name == "train_subsample_frac" else None)
                    == value
                    for name, value in overrides.items()
                ):
                    frame = select_methods(self.references[key], cfg["anchor_sensor"])
                    frames.append(frame.assign(**labels))
                else:
                    directory = (
                        self.args.output
                        / dataset
                        / (key[1] or "random")
                        / cell
                        / f"seed_{seed}"
                    )
                    jobs.append(
                        {
                            "source": str(self.sources[key]),
                            "directory": str(directory),
                            "overrides": overrides,
                            "labels": labels,
                        }
                    )
        run_gpu_jobs(jobs, lambda gpu, job: launch_job("paper_reproduce._sensitivity", gpu, job))
        frames.extend(
            pd.read_csv(
                Path(job["directory"]) / "performance.csv", float_precision="round_trip"
            )
            for job in jobs
        )
        return pd.concat(frames, ignore_index=True)


def worker(job):
    _, reference = load_model(job["source"])
    device = initialize_runtime(int(reference["seed"]))
    fold = variant_fold(reference, job["overrides"], device)
    model = fit_fold(fold, device)
    with torch.no_grad():
        mean, variance = model.predict_posterior(fold.test.x, fold.test.sensors)
        point = (
            posterior_point_prediction(
                mean.ravel(), variance.ravel(), fold.config["obs_transform"]
            )
            .cpu()
            .numpy()
        )
    target = fold.test.targets.cpu().numpy()
    anchor = fold.config.get(
        "anchor_sensor", fold.config["b_cols"][int(fold.config["anchor_idx"])]
    )
    common = {
        "dataset": reference["dataset"],
        "fold": reference.get("calendar_fold_id", "random"),
        "seed": reference["seed"],
        "test_rows": len(target),
        "anchor_sensor": anchor,
        **job["labels"],
    }
    frame = pd.DataFrame(
        [
            {"method": fold.config["model"], **point_errors(point, target), **common},
            {
                "method": "raw_anchor",
                **point_errors(
                    fold.frames.test[anchor].to_numpy(dtype=np.float32), target
                ),
                **common,
            },
        ]
    )
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "performance.csv", index=False)


def save_sensitivity_results(args, raw, keys):
    """Pool fold squared errors by test rows, then summarize variability by seed."""
    raw.to_csv(args.output / "fold_results.csv", index=False)
    rows = []
    for labels, group in raw.groupby(keys, sort=False):
        labels = labels if isinstance(labels, tuple) else (labels,)
        real = group.loc[group.dataset.ne("toy_spatial")]
        toy = group.loc[group.dataset.eq("toy_spatial")]
        if not real.empty:
            per_seed, _ = aggregate_performance(real)
            rows.append(per_seed.assign(**dict(zip(keys, labels, strict=True))))
        if not toy.empty:
            rows.append(toy)
    return save_results(args, rows, keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help=argparse.SUPPRESS)
    worker(json.loads(parser.parse_args().job))


if __name__ == "__main__":
    main()
