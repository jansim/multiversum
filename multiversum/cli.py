import click
from pathlib import Path

from .multiverse import DEFAULT_SEED, MultiverseAnalysis, add_ids_to_multiverse_grid
from .logger import logger

DEFAULT_CONFIG_FILES = ["multiverse.toml", "multiverse.json", "multiverse.py"]


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["full", "continue", "test"]),
    default="full",
    help="How to run the multiverse analysis. (continue: continue from previous run, full: run all universes, test: run only a small subset of universes)",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help=f"Relative path to a TOML, JSON or Python file with a config for the multiverse. Defaults to searching for {', '.join(DEFAULT_CONFIG_FILES)} (in that order).",
)
@click.option(
    "--notebook",
    type=click.Path(),
    default="./universe.ipynb",
    help="Relative path to the notebook to run.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./output",
    help="Relative path to output directory for the results.",
)
@click.option(
    "--seed", type=int, default=DEFAULT_SEED, help="The seed to use for the analysis."
)
@click.option(
    "--u-id",
    type=str,
    default=None,
    help="Examine only a single universe with the given universe id (or starting with the provided id).",
)
@click.pass_context
def cli(
    ctx,
    mode,
    config,
    notebook,
    output_dir,
    seed,
    u_id,
):
    """Run a multiverse analysis from the command line."""
    logger.debug(f"Parsed arguments: {ctx.params}")

    if config is not None:
        config_file = Path(config)
    else:
        config_file = None
        for file in DEFAULT_CONFIG_FILES:
            if Path(file).is_file():
                config_file = Path(file)
                break

    multiverse_analysis = MultiverseAnalysis(
        config_file=config_file,
        notebook=Path(notebook),
        output_dir=Path(output_dir),
        new_run=(mode != "continue"),
        seed=seed,
    )

    multiverse_grid = multiverse_analysis.generate_grid(save=True)
    logger.info(f"Generated N = {len(multiverse_grid)} universes")

    if u_id is not None:
        # Search for this particular universe
        multiverse_dict = add_ids_to_multiverse_grid(multiverse_grid)
        matching_values = [
            key for key in multiverse_dict.keys() if key.startswith(u_id)
        ]
        assert len(matching_values) == 1, (
            f"The id {u_id} matches {len(matching_values)} universe ids."
        )
        logger.info(f"Running only universe: {matching_values[0]}")
        multiverse_grid = [multiverse_dict[matching_values[0]]]

    logger.info(
        f"~ Starting Run No. {multiverse_analysis.run_no} (Seed: {multiverse_analysis.seed}) ~"
    )

    # Run the analysis for the first universe
    if mode == "test":
        logger.info("Test Run")
        multiverse_analysis.visit_universe(multiverse_grid[0])
        if len(multiverse_grid) > 1:
            multiverse_analysis.visit_universe(
                multiverse_grid[len(multiverse_grid) - 1]
            )
    elif mode == "continue":
        logger.info("Continuing Previous Run")
        missing_universes = multiverse_analysis.check_missing_universes()[
            "missing_universes"
        ]

        # Run analysis only for missing universes
        multiverse_analysis.examine_multiverse(missing_universes)
    else:
        logger.info("Full Run")
        # Run analysis for all universes
        multiverse_analysis.examine_multiverse(multiverse_grid)

    multiverse_analysis.aggregate_data(save=True)

    multiverse_analysis.check_missing_universes()


if __name__ == "__main__":
    cli()
