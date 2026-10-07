"""Table 4: hidden-width and learning-rate sensitivity around Table 1 models."""

from paper_reproduce._common import (
    parse_args,
    write_table,
)
from paper_reproduce._sensitivity import (
    DEFAULT_SEEDS,
    SensitivityExperiments,
    configure_parser,
    save_sensitivity_results,
)


def main():
    args = parse_args(
        __file__, __doc__, seeds=DEFAULT_SEEDS, configure_parser=configure_parser
    )
    experiments = SensitivityExperiments(args)
    cells = []
    for dataset in args.datasets:
        widths = (
            (32, 64, 128, 256, 512) if dataset == "toy_spatial" else (2, 4, 8, 16, 32)
        )
        rates = (
            (1e-5, 2e-5, 4e-5, 1e-4, 2e-4)
            if dataset == "toy_spatial"
            else (5e-5, 1e-4, 2e-4, 5e-4, 1e-3)
        )
        for seed in args.seeds:
            for axis, grid in (("hidden_dim", widths), ("lr", rates)):
                for value in grid:
                    overrides = {axis: value}
                    if axis == "hidden_dim":
                        overrides.update(
                            prior_hidden_dim=value,
                            rel_hidden_dim=value,
                            bias_hidden_dim=max(1, value // 2),
                        )
                    cells.append(
                        (
                            dataset,
                            f"{axis}_{value:g}",
                            seed,
                            overrides,
                            {"axis": axis, "value": value},
                        )
                    )
    raw = experiments.evaluate(cells)
    raw = raw.loc[raw.method.eq("anchored")]
    summary = save_sensitivity_results(args, raw, ["dataset", "axis", "value"])
    summary.to_csv(args.output / "grid.csv", index=False)
    summary = summary.groupby(["dataset", "axis"], sort=False)[
        ["rmse_mean", "mae_mean"]
    ].agg(["min", "max"])
    summary.columns = [
        f"{metric.removesuffix('_mean')}_{stat}" for metric, stat in summary.columns
    ]
    summary = summary.reset_index()
    for metric in ("rmse", "mae"):
        summary[f"{metric}_span_pct"] = (
            summary[f"{metric}_max"] / summary[f"{metric}_min"] - 1
        ) * 100
    write_table(args, summary)


if __name__ == "__main__":
    main()
