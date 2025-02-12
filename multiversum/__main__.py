from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from .logger import logger
from .multiverse import (
    DEFAULT_CONFIG_FILES,
    DEFAULT_SEED,
    DEFAULT_UNIVERSE_FILES,
    MultiverseAnalysis,
    add_ids_to_multiverse_grid,
)


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["full", "continue", "test"]),
    default="full",
    help="How to run the multiverse analysis. (continue: continue from previous run, full: run all universes, test: run a minimal set of universes where each unique option appears at least once)",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help=f"Relative path to a TOML, JSON or Python file with a config for the multiverse. Defaults to searching for {', '.join(DEFAULT_CONFIG_FILES)} (in that order).",
)
@click.option(
    "--universe",
    type=click.Path(),
    default=None,
    help=f"Relative path to the universe file to run. Defaults to searching for {', '.join(DEFAULT_UNIVERSE_FILES)} (in that order).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./output",
    help="Relative path to output directory for the results.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help=f"The seed to use for the analysis (Defaults to {DEFAULT_SEED}).",
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
    universe,
    output_dir,
    seed,
    u_id,
):
    """Run a multiverse analysis from the command line."""
    logger.debug(f"Parsed arguments: {ctx.params}")

    multiverse_analysis = MultiverseAnalysis(
        config=config,
        universe=universe,
        output_dir=Path(output_dir),
        new_run=(mode != "continue"),
        seed=seed,
    )

    multiverse_grid = multiverse_analysis.generate_grid(save=True)

    # Set panel style based on mode
    MODE_DESCRIPTIONS = {
        "full": "Full Run",
        "continue": "Continuing Previous Run",
        "test": "Test Run",
    }

    MODE_STYLES = {"full": "green", "continue": "yellow", "test": "magenta"}

    # Initialize rich console
    console = Console()
    console.print(
        Panel.fit(
            f"Generated [bold cyan]N = {len(multiverse_grid)}[/bold cyan] universes\n"
            f"Mode: [bold {MODE_STYLES[mode]}]{MODE_DESCRIPTIONS[mode]}[/bold {MODE_STYLES[mode]}]\n"
            f"Run No.: [bold cyan]{multiverse_analysis.run_no}[/bold cyan]\n"
            f"Seed: [bold cyan]{multiverse_analysis.seed}[/bold cyan]",
            title="multiversum: Multiverse Analysis",
            border_style=MODE_STYLES[mode],
        )
    )

    if u_id is not None:
        # Search for this particular universe
        multiverse_dict = add_ids_to_multiverse_grid(multiverse_grid)
        matching_values = [
            key for key in multiverse_dict.keys() if key.startswith(u_id)
        ]
        assert len(matching_values) == 1, (
            f"The id {u_id} matches {len(matching_values)} universe ids."
        )
        console.print(
            f"[bold yellow]Running only universe:[/bold yellow] {matching_values[0]}"
        )
        multiverse_grid = [multiverse_dict[matching_values[0]]]

    # Run the analysis for the first universe
    if mode == "test":
        minimal_grid = multiverse_analysis.generate_minimal_grid()
        console.print(
            f"Generated minimal test grid with [bold cyan]{len(minimal_grid)}[/bold cyan] universes"
        )
        multiverse_analysis.examine_multiverse(minimal_grid)
    elif mode == "continue":
        missing_universes = multiverse_analysis.check_missing_universes()[
            "missing_universes"
        ]

        # Run analysis only for missing universes
        multiverse_analysis.examine_multiverse(missing_universes)
    else:
        # Run analysis for all universes
        multiverse_analysis.examine_multiverse(multiverse_grid)

    with console.status("[bold green]Aggregating data...[/bold green]"):
        multiverse_analysis.aggregate_data(save=True)

    multiverse_analysis.check_missing_universes()


if __name__ == "__main__":
    cli()
