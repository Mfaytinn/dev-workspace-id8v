"""Table 6: each Table 1 input sensor as anchor versus its raw readings."""

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
        for sensor in experiments.sensors(dataset):
            cells.extend(
                (
                    dataset,
                    f"anchor_{sensor}",
                    seed,
                    {"anchor_sensor": sensor},
                    {"anchor": sensor},
                )
                for seed in args.seeds
            )
    raw = experiments.evaluate(cells)
    write_table(
        args, save_sensitivity_results(args, raw, ["dataset", "anchor", "method"])
    )


if __name__ == "__main__":
    main()
