import argparse
from pathlib import Path
import runpy
from typing import Optional

from .multiverse import DEFAULT_SEED, MultiverseAnalysis, add_ids_to_multiverse_grid
from .logger import logger

DEFAULT_CONFIG_FILE = "multiverse.toml"
DEFAULT_PYTHON_SCRIPT_NAME = "multiverse.py"


def run_cli(dimensions: Optional[dict] = None, **kwargs) -> None:
    """Run a multiverse analysis from the command line.

    You will only need to use this function if you want to directly pass in
    dimensions from within Python (e.g. if you generate dimensions via code).

    To run the CLI without providing dimensions, simply run
    `python -m multiversum`.

    Args:
        dimensions (dict, optional): Manually specify dimensions. Set to None to
            use normal default / allow specification as an argument.
            Defaults to None.
        **kwargs: Additional keyword arguments to pass to the MultiverseAnalysis

    Raises:
        FileNotFoundError: If the dimensions file is not found.
        NotADirectoryError: If the output directory is not found.
    """
    parser = argparse.ArgumentParser("multiversum")

    parser.add_argument(
        "--mode",
        help=(
            "How to run the multiverse analysis. "
            "(continue: continue from previous run, "
            "full: run all universes, "
            "test: run only a small subset of universes)"
        ),
        choices=["full", "continue", "test"],
        default="full",
    )

    def verify_file(string):
        if Path(string).is_file():
            return string
        else:
            raise FileNotFoundError(string)

    parser.add_argument(
        "--config",
        help=(
            "Relative path to a TOML or JSON file with a config for the multiverse."
            f"Defaults to {DEFAULT_CONFIG_FILE}."
        ),
        default=None,
        type=verify_file,
    )

    parser.add_argument(
        "--notebook",
        help=("Relative path to the notebook to run."),
        default="./universe.ipynb",
        type=str,
    )

    parser.add_argument(
        "--output-dir",
        help=("Relative path to output directory for the results."),
        default="./output",
        type=str,
    )

    parser.add_argument(
        "--seed",
        help=("The seed to use for the analysis."),
        default=str(DEFAULT_SEED),
        type=int,
    )

    parser.add_argument(
        "--u-id",
        help=(
            "Examine only a single universe with the given universe id (or starting with the provided id)."
        ),
        default=None,
        type=str,
    )

    args = parser.parse_args()
    logger.debug(f"Parsed arguments: {args}")

    if args.config is not None:
        config_file = Path(args.config)
    elif Path(DEFAULT_CONFIG_FILE).is_file():
        config_file = Path(DEFAULT_CONFIG_FILE)
    else:
        config_file = None

    if config_file is None and dimensions is None:
        # Check whether multiverse.py might exist and run it
        if Path(DEFAULT_PYTHON_SCRIPT_NAME).is_file():
            logger.info(
                f"Detected {DEFAULT_PYTHON_SCRIPT_NAME}. Running it.\n"
                f"Please create a {DEFAULT_CONFIG_FILE}, if you don't want to run the script."
            )

            runpy.run_path(DEFAULT_PYTHON_SCRIPT_NAME)
            return

    multiverse_analysis = MultiverseAnalysis(
        dimensions=dimensions,
        config_file=config_file,
        notebook=Path(args.notebook),
        output_dir=Path(args.output_dir),
        new_run=(args.mode != "continue"),
        seed=args.seed,
        **kwargs,
    )

    multiverse_grid = multiverse_analysis.generate_grid(save=True)
    logger.info(f"Generated N = {len(multiverse_grid)} universes")

    if args.u_id is not None:
        # Search for this particular universe
        multiverse_dict = add_ids_to_multiverse_grid(multiverse_grid)
        matching_values = [
            key for key in multiverse_dict.keys() if key.startswith(args.u_id)
        ]
        assert len(matching_values) == 1, (
            f"The id {args.u_id} matches {len(matching_values)} universe ids."
        )
        logger.info(f"Running only universe: {matching_values[0]}")
        multiverse_grid = [multiverse_dict[matching_values[0]]]

    logger.info(
        f"~ Starting Run No. {multiverse_analysis.run_no} (Seed: {multiverse_analysis.seed}) ~"
    )

    # Run the analysis for the first universe
    if args.mode == "test":
        logger.info("Test Run")
        multiverse_analysis.visit_universe(multiverse_grid[0])
        if len(multiverse_grid) > 1:
            multiverse_analysis.visit_universe(
                multiverse_grid[len(multiverse_grid) - 1]
            )
    elif args.mode == "continue":
        logger.info("Continuing Previous Run")
        missing_universes = multiverse_analysis.check_missing_universes()[
            "missing_universes"
        ]

        # Run analysis only for missing universes
        multiverse_analysis.examine_multiverse(multiverse_grid=missing_universes)
    else:
        logger.info("Full Run")
        # Run analysis for all universes
        multiverse_analysis.examine_multiverse(multiverse_grid=multiverse_grid)

    multiverse_analysis.aggregate_data(save=True)

    multiverse_analysis.check_missing_universes()
