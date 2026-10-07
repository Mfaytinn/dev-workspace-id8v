"""Table 5: four training fractions on Table 1 models and fixed held-out folds."""

from paper_reproduce._common import parse_args, write_table
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
        for seed in args.seeds:
            cells.extend(
                (
                    dataset,
                    f"fraction_{fraction:g}",
                    seed,
                    {"train_subsample_frac": fraction},
                    {"train_fraction": fraction},
                )
                for fraction in (1.0, 0.5, 0.25, 0.125)
            )
    raw = experiments.evaluate(cells)
    raw = raw.loc[raw.method.eq("anchored")]
    write_table(
        args, save_sensitivity_results(args, raw, ["dataset", "train_fraction"])
    )


if __name__ == "__main__":
    main()
