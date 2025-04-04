import os
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from .cli_helpers import (
    create_summary_table,
    parse_partial_percentages,
    split_multiverse_grid,
)
from .helpers import calculate_cpu_count
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
    type=click.Choice(["full", "continue", "test", "partial-parallel", "finalize"]),
    default="full",
    help=(
        "How to run the multiverse analysis."
        "(continue: continue from previous run, "
        "full: run all universes, "
        "test: run a minimal set of universes where each unique option appears at least once, "
        "partial-parallel: run a partial range of universes in parallel see --partial"
        "finalize: finalize the previous run e.g. after running in partial-parallel mode)"
    ),
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
@click.option(
    "--grid-only",
    is_flag=True,
    default=False,
    help="Only export the multiverse grid without running the analysis.",
)
@click.option(
    "--grid-format",
    type=click.Choice(["json", "csv", "none"]),
    default="json",
    help="Format of the exported multiverse grid (json, csv, or none to skip export).",
)
@click.option(
    "--n-jobs",
    type=int,
    default=-2,
    help="Number of CPUs to use for parallel processing. -1 uses all CPUs, -2 uses all but one CPU (default), and 1 disables parallel processing.",
)
@click.option(
    "--partial",
    type=str,
    default=None,
    help="Run only a specific percentage range of universes. Format: 'start%,end%' (e.g. '0%,50%' or '0,20%'). Set mode to 'partial-parallel' to run in parallel and avoid race conditions.",
)
@click.option(
    "--retry-errors",
    is_flag=True,
    default=False,
    help="When mode=continue, retry universes that previously resulted in errors.",
)
@click.option(
    "--retry-error-type",
    type=str,
    default=None,
    help="When mode=continue, retry only universes that failed with this specific error type.",
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
    grid_only,
    grid_format,
    n_jobs,
    partial,
    retry_errors,
    retry_error_type,
):
    """Run a multiverse analysis from the command line."""
    # Initialize rich console
    console = Console()

    logger.debug(f"Parsed arguments: {ctx.params}")

    # Check retry flags are only used with continue mode
    if (retry_errors or retry_error_type) and mode != "continue":
        logger.warning(
            f"The --retry-errors and --retry-error-type flags are only used with --mode continue, "
            f"but mode is set to '{mode}'. These flags will be ignored."
        )

    multiverse_analysis = MultiverseAnalysis(
        config=config,
        universe=universe,
        output_dir=Path(output_dir),
        new_run=(mode not in ["continue", "partial-parallel", "finalize"]),
        seed=seed,
    )

    # Calculate actual CPU count to use
    actual_n_jobs = calculate_cpu_count(n_jobs)
    total_cpus = os.cpu_count() or 1

    # Generate the grid with the specified export format
    multiverse_grid = multiverse_analysis.generate_grid(
        save_format=grid_format,
    )

    # If export-only is specified, exit after exporting the grid
    if grid_only:
        if grid_format == "none":
            logger.warning(
                "Using --export-grid-only without specifying an export format. Nothing will happen."
            )
        else:
            console.print(
                Panel.fit(
                    f"Exported [bold cyan]N = {len(multiverse_grid)}[/bold cyan]\n"
                    f"Format: [bold green]{grid_format}[/bold green]",
                    title="multiversum: Grid Export Only",
                    border_style="green",
                )
            )
        return

    if mode != "finalize":
        # Set panel style based on mode
        MODE_DESCRIPTIONS = {
            "full": "Full Run",
            "continue": "Continuing Previous Run",
            "test": "Test Run",
            "partial-parallel": "Parallel Run (Partial)",
        }

        MODE_STYLES = {
            "full": "green",
            "continue": "yellow",
            "test": "magenta",
            "partial-parallel": "blue",
        }

        console.print(
            Panel.fit(
                f"Generated [bold cyan]N = {len(multiverse_grid)}[/bold cyan] universes\n"
                f"Mode: [bold {MODE_STYLES[mode]}]{MODE_DESCRIPTIONS[mode]}[/bold {MODE_STYLES[mode]}]\n"
                f"Run No.: [bold cyan]{multiverse_analysis.run_no}[/bold cyan]\n"
                f"Seed: [bold cyan]{multiverse_analysis.seed}[/bold cyan]\n"
                f"CPUs: [bold cyan]Using {actual_n_jobs}/{total_cpus} (n-jobs: {n_jobs})[/bold cyan]",
                title="multiversum: Multiverse Analysis",
                border_style=MODE_STYLES[mode],
            )
        )

        # Only u_id or split can be provided, not both
        assert u_id is None or partial is None

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

        # Apply split if provided
        if partial is not None:
            try:
                start_pct, end_pct = parse_partial_percentages(partial)
                multiverse_grid, start_idx, end_idx = split_multiverse_grid(
                    multiverse_grid, start_pct, end_pct
                )
                console.print(
                    f"[bold yellow]Running only {end_pct - start_pct:.1%} of universes:[/bold yellow] "
                    f"from {start_pct:.1%} to {end_pct:.1%} (indices {start_idx} to {end_idx})"
                )
                if mode != "partial-parallel":
                    console.print(
                        "Detected a partial run. If you want to run multiple partial multiversum analyses in parallel, "
                        "you may want to use mode='partial-parallel' to avoid race conditions."
                    )
            except ValueError as e:
                logger.error(f"Invalid split format: {e}")
                ctx.exit(1)

        if mode == "partial-parallel":
            if partial is None:
                logger.error(
                    "You must provide a partial range when using mode='partial-parallel'."
                )
                ctx.exit(1)
            console.print(
                "Running in manual parallel mode with partial ranges. Will not create a new run, "
                "but rather reuse the previous one to avoid starting multiple."
            )

            multiverse_analysis.examine_multiverse(
                multiverse_grid, n_jobs=actual_n_jobs
            )

            console.print(
                "Finished running the partial analysis. "
                "Use mode=finalize to finalize the run when all partial analyses are finished."
            )
        # Run the analysis for the first universe
        elif mode == "test":
            minimal_grid = multiverse_analysis.generate_minimal_grid()
            console.print(
                f"Generated minimal test grid with [bold cyan]{len(minimal_grid)}[/bold cyan] universes"
            )
            multiverse_analysis.examine_multiverse(minimal_grid, n_jobs=actual_n_jobs)
        elif mode == "continue":
            missing_info = multiverse_analysis.check_missing_universes()
            missing_universes = missing_info["missing_universes"]

            if retry_error_type is not None:
                if retry_errors:
                    logger.warning(
                        "Both --retry-errors and --retry-error-type provided. "
                        "Only --retry-error-type will be used."
                    )
                if retry_error_type in missing_info["error_universes_by_type"]:
                    error_universes = missing_info["error_universes_by_type"][
                        retry_error_type
                    ]
                    missing_universes.extend(error_universes)
                    console.print(
                        f"[bold yellow]Found {len(error_universes)} universes "
                        f"that failed with error type '{retry_error_type}'[/bold yellow]"
                    )
                else:
                    logger.warning(
                        f"No universes found with error type '{retry_error_type}'"
                    )
            elif retry_errors:
                # Add all errored universes to the list of universes to run
                missing_universes.extend(missing_info["error_universes"])

            # Run analysis only for missing/errored universes
            multiverse_analysis.examine_multiverse(
                missing_universes, n_jobs=actual_n_jobs
            )
        else:
            # Run analysis for all universes
            multiverse_analysis.examine_multiverse(
                multiverse_grid, n_jobs=actual_n_jobs
            )

    multiverse_analysis.check_missing_universes()

    # Aggregate data
    agg_data = None
    with console.status("[bold green]Aggregating data...[/bold green]"):
        agg_data = multiverse_analysis.aggregate_data(save=True)

    # Display a summary table of the analysis results
    table = create_summary_table(agg_data)
    if table:
        console.print(table)


if __name__ == "__main__":
    cli()
