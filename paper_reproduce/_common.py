"""Small execution and reporting helpers for the paper scripts."""

import argparse
import os
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import TypeVar

import pandas as pd
import torch

from ncam.experiments.config import load_config

DATASETS = ("toy_spatial", "seneurcity", "cairsense")
BASE_CONFIGS = {name: f"configs/{name}.yaml" for name in DATASETS}
DEFAULT_OUTPUT_ROOT = Path("results/tables")
DEFAULT_CALENDAR_STUDY = DEFAULT_OUTPUT_ROOT / "calendar"
DEFAULT_TABLE1_RESULTS = DEFAULT_OUTPUT_ROOT / "table1_predictivePerformance"
Job, Result = TypeVar("Job"), TypeVar("Result")


def parse_args(
    script, description, seeds=(0, 1, 42), datasets=DATASETS, *, configure_parser=None
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--datasets", nargs="+", choices=datasets, default=list(datasets)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=(0, 1, 42), default=list(seeds)
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    if configure_parser:
        configure_parser(parser)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    args.datasets = list(dict.fromkeys(args.datasets))
    args.output = args.output_root / Path(script).stem
    return args


def configure_calendar_parser(parser):
    parser.add_argument(
        "--calendar-study",
        type=Path,
        default=DEFAULT_CALENDAR_STUDY,
        help="Reference models from the calendar training script",
    )


def base_config(dataset):
    return dict(load_config(BASE_CONFIGS[dataset]), dataset=dataset)


def ensure_data(dataset, cfg):
    if dataset == "toy_spatial" and not Path(cfg["data_path"]).is_file():
        subprocess.run(
            [
                "uv",
                "run",
                "--locked",
                "python",
                "-m",
                "paper_reproduce.toy_spatial_generate",
                "--out-path",
                cfg["data_path"],
            ],
            check=True,
        )
    if dataset == "seneurcity":
        if not list(Path(cfg["data_dir"]).glob(cfg["pattern"])):
            raise FileNotFoundError(
                "Supply the prepared Antwerp CSVs in data/seneurcity_preprocessed; see README.md"
            )
    elif not Path(cfg["data_path"]).is_file():
        raise FileNotFoundError(
            f"Supply the prepared input at {cfg['data_path']}; see README.md"
        )


def model_directory(root, dataset, fold, seed):
    return Path(root) / "models" / dataset / (fold or "random") / f"seed_{seed}"


def launch_job(module, gpu, job):
    """Pass a job directly to a fresh process; no job files or run logs."""
    import json

    subprocess.run(
        ["uv", "run", "--locked", "python", "-m", module, "--job", json.dumps(job)],
        env=gpu_subprocess_env(gpu),
        check=True,
    )


def visible_gpus() -> list[str | None]:
    """Return every CUDA device visible to this process, or one CPU worker."""
    count = torch.cuda.device_count()
    if not count:
        return [None]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible.split(",") if visible else [str(index) for index in range(count)]


def gpu_subprocess_env(gpu: str | None) -> dict[str, str]:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def run_gpu_jobs(
    jobs: list[Job], worker: Callable[[str | None, Job], Result]
) -> list[Result]:
    """Run independent jobs with at most one active process per visible GPU."""
    devices = visible_gpus()
    queue: Queue[Job] = Queue()
    for job in jobs:
        queue.put(job)
    print(f"Running {len(jobs)} jobs on GPUs {devices}", flush=True)

    def drain(gpu: str | None) -> list[Result]:
        results = []
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                return results
            results.append(worker(gpu, job))

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(drain, gpu) for gpu in devices]
        return [result for future in futures for result in future.result()]


def save_results(args, frames, keys, values=("rmse", "mae")) -> pd.DataFrame:
    raw = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output / "results.csv", index=False)
    # Average repeated splits within each seed before computing seed variability.
    per_seed = raw.groupby([*keys, "seed"], sort=False)[list(values)].mean()
    summary = per_seed.groupby(keys, sort=False).agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = per_seed.groupby(keys, sort=False).size()
    return summary.reset_index()


def write_table(args, frame: pd.DataFrame) -> None:
    frame.to_csv(args.output / "table.csv", index=False)
    print(f"Saved {args.output / 'table.csv'}")
