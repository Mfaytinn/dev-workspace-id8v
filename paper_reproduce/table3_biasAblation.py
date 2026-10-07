"""Table 3: reuse Table 1 anchored results and refit only the no-bias model."""

import pandas as pd

from paper_reproduce._common import parse_args, save_results, write_table
from paper_reproduce._sensitivity import SensitivityExperiments, configure_parser
from paper_reproduce.run_temporal_calendar import aggregate_performance


def main():
    args = parse_args(__file__, __doc__, configure_parser=configure_parser)
    experiments = SensitivityExperiments(args)
    cells = [
        (dataset, "nobias", seed, {"model": "nobias"}, {})
        for dataset in args.datasets
        for seed in args.seeds
    ]
    raw = experiments.evaluate(cells)
    raw = raw.loc[raw.method.eq("nobias")].copy()
    raw.to_csv(args.output / "fold_results.csv", index=False)
    variants, _ = aggregate_performance(raw)
    source = pd.read_csv(
        args.table1_results / "results.csv", float_precision="round_trip"
    )
    anchored = source.loc[
        source.method.eq("anchored")
        & source.dataset.isin(args.datasets)
        & source.seed.isin(args.seeds)
    ]
    summary = save_results(args, [anchored, variants], ["dataset", "method"])
    base = summary.loc[summary.method.eq("anchored")].set_index("dataset")
    for metric in ("rmse", "mae"):
        summary[f"{metric}_delta_pct"] = (
            summary[f"{metric}_mean"] / summary.dataset.map(base[f"{metric}_mean"]) - 1
        ) * 100
    write_table(args, summary)


if __name__ == "__main__":
    main()
