"""
This module contains helper functions to orchestrate a multiverse analysis.
"""

import contextlib
import io
import json
import runpy
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import pandas as pd
import papermill as pm
from joblib import Parallel, cpu_count, delayed

from .helpers import (
    add_ids_to_multiverse_grid,
    add_universe_info_to_df,
    calculate_cpu_count,
    generate_minimal_multiverse_grid,
    generate_multiverse_grid,
    generate_universe_id,
    search_files,
)
from .logger import logger

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .parallel import rich_joblib

DEFAULT_SEED = 80539
ERRORS_DIR_NAME = "errors"
SCRIPT_GLOBAL_OVERWRITE_NAME = "MULTIVERSUM_OVERRIDE_SETTINGS"
DEFAULT_CONFIG_FILES = ["multiverse.toml", "multiverse.json", "multiverse.py"]
DEFAULT_UNIVERSE_FILES = ["universe.ipynb", "universe.py"]
DEFAULT_STOP_ON_ERROR = True

ERROR_TYPE_COLUMN = "mv_error_type"
ERROR_MESSAGE_COLUMN = "mv_error"


@dataclass
class Config:
    """
    Configuration for the multiverse analysis.

    Attributes:
        dimensions: A dictionary where keys are dimension names and values are lists of possible values for each dimension.
        constraints: Optional dictionary where keys are dimension names and values are lists of constraints. Each constraint is a dictionary with:
            - value: The value of the dimension that the constraint applies to.
            - allowed_if: A dictionary of dimension-value pairs that must be present for the constraint to be allowed.
            - forbidden_if: A dictionary of dimension-value pairs that must not be present for the constraint to be allowed.
            Only one of allowed_if and forbidden_if can be present in a constraint.
            Example:
                constraints = {
                    "dimension1": [
                        {
                            "value": "value1",
                            "allowed_if": {"dimension2": "value2"}
                        },
                        {
                            "value": "value3",
                            "forbidden_if": {"dimension4": "value4"}
                        }
                    ]
                }
        seed: Optional seed for random number generation.
        stop_on_error: Optional flag to stop on error.
        cell_timeout: Optional timeout (in seconds) for each cell in the notebook.
    """

    dimensions: Dict[str, Any]
    constraints: Optional[Dict[str, List[Dict[str, Any]]]] = None
    seed: Optional[int] = None
    stop_on_error: Optional[bool] = None
    cell_timeout: Optional[int] = None


class MissingUniverseInfo(TypedDict):
    """
    Information about missing or errored universes in a multiverse analysis.

    Attributes:
        missing_universe_ids: List of IDs of universes not yet run
        extra_universe_ids: List of IDs found in data but not in multiverse grid
        error_universe_ids: List of IDs of universes that errored out
        missing_universes: List of dictionaries containing the settings for missing universes
        error_universes: List of dictionaries containing the settings for errored universes
        error_universes_by_type: Dictionary mapping error types to lists of errored universe settings
    """

    missing_universe_ids: List[str]
    extra_universe_ids: List[str]
    error_universe_ids: List[str]
    missing_universes: List[Dict[str, str]]
    error_universes: List[Dict[str, str]]
    error_universes_by_type: Dict[str, List[Dict[str, str]]]


class MultiverseAnalysis:
    """
    This class orchestrates a multiverse analysis.

    Attributes:
        dimensions: A dictionary where keys are dimension names and values are lists of possible values for each dimension.
        constraints: Optional dictionary where keys are dimension names and values are lists of constraints. Each constraint is a dictionary with:
            - value: The value of the dimension that the constraint applies to.
            - allowed_if: A dictionary of dimension-value pairs that must be present for the constraint to be allowed.
            - forbidden_if: A dictionary of dimension-value pairs that must not be present for the constraint to be allowed.
            Only one of allowed_if and forbidden_if can be present in a constraint.
            Example:
                constraints = {
                    "dimension1": [
                        {
                            "value": "value1",
                            "allowed_if": {"dimension2": "value2"}
                        },
                        {
                            "value": "value3",
                            "forbidden_if": {"dimension4": "value4"}
                        }
                    ]
                }
        seed: The seed to use for the analysis.
        cell_timeout: A timeout (in seconds) for each cell in the notebook.
        stop_on_error: Whether to stop the analysis if an error occurs.
        run_no: The number of the current run.
        new_run: Whether this is a new run or not.
        output_dir: The directory to store the output in.
        universe_file: The Path to the universe file to run.
        grid: Optional list of dictionaries containing the settings for different universes.
    """

    dimensions = None
    constraints = None
    seed = DEFAULT_SEED
    cell_timeout = None
    stop_on_error = DEFAULT_STOP_ON_ERROR

    run_no: int
    new_run: bool
    output_dir: Path
    universe_file: Path

    grid: Optional[List[Dict[str, Any]]] = None

    def __init__(
        self,
        dimensions: Optional[Dict] = None,
        config: Union[Path, Config, None] = None,
        universe: Path = None,
        output_dir: Path = Path("./output"),
        run_no: Optional[int] = None,
        new_run: bool = True,
        seed: Optional[int] = None,
        stop_on_error: Optional[bool] = None,
        cell_timeout: Optional[int] = None,
    ) -> None:
        """
        Initializes a new MultiverseAnalysis instance.

        Args:
            dimensions: A dictionary where keys are dimension names and values are lists of possible values for each dimension.
                Each dimension corresponds to a decision.
            config: A Path to a TOML, JSON or Python file containing the
                analysis configuration. Supported configuration options can be
                found in the Config class. If a Python file is used, it should
                contain a dictionary / config object named "config".
                Will automatically search for multiverse.toml / .json / .py.
            universe: The Path to the universe_file to run. Either an
                ipython / jupyter notebook (.ipynb) or a python script (.py).
            output_dir: The directory to store the output in.
            run_no: The number of the current run. Defaults to an automatically
                incrementing integer number if new_run is True or the last run if
                new_run is False.
            new_run: Whether this is a new run or not. Defaults to True.
            seed: The seed to use for the analysis.
            stop_on_error: Whether to stop the analysis if an error occurs.
            cell_timeout: A timeout (in seconds) for each cell in the notebook.
        """
        # Check for configuration file and parse it
        config_file = search_files(file=config, default_files=DEFAULT_CONFIG_FILES)
        if isinstance(config_file, Path):
            if config_file.suffix == ".toml":
                with open(config_file, "rb") as fp:
                    config = tomllib.load(fp)
            elif config_file.suffix == ".json":
                with open(config_file, "r") as fp:
                    config = json.load(fp)
            elif config_file.suffix == ".py":
                results = runpy.run_path(str(config_file))
                config = results["config"]
            else:
                raise ValueError(
                    "Only .toml, .json and .py files are supported as config."
                )
        # Convert config to Config object
        if isinstance(config, dict):
            config = Config(**config)

        # Read settings from config (or args)
        self.read_config_value(config, "dimensions", dimensions)
        self.read_config_value(config, "seed", seed)
        self.read_config_value(config, "stop_on_error", stop_on_error)
        self.read_config_value(config, "cell_timeout", cell_timeout)
        self.read_config_value(config, "constraints")

        universe_file = search_files(
            file=universe, default_files=DEFAULT_UNIVERSE_FILES
        )
        self.universe_file = universe_file

        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_no = (
            run_no if run_no is not None else self.read_counter(increment=new_run)
        )

        if self.dimensions is None:
            raise ValueError(
                "Dimensions need to be specified either directly or in a config."
            )

    def read_config_value(
        self, config: Optional[Config], key: str, overwrite_value: Optional[Any] = None
    ):
        config_value = getattr(config, key) if config is not None else None

        if overwrite_value is not None:
            if config_value is not None:
                logger.warning(
                    f"Overwriting config value {key} ({config_value}) with {overwrite_value} as it was passed directly."
                )
            setattr(self, key, overwrite_value)
        elif config_value is not None:
            # Use value from config
            setattr(self, key, config_value)

    def get_run_dir(self, sub_directory: Optional[str] = None) -> Path:
        """
        Get the directory for the current run.

        Args:
            sub_directory: An optional sub-directory to append to the run directory.

        Returns:
            A Path object for the current run directory.
        """
        run_dir = self.output_dir / "runs" / str(self.run_no)
        target_dir = run_dir / sub_directory if sub_directory is not None else run_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def read_counter(self, increment: bool) -> int:
        """
        Read the counter from the output directory.

        Args:
            increment: Whether to increment the counter after reading.

        Returns:
            The current value of the counter.
        """

        # Use a self-incrementing counter via counter.txt
        counter_filepath = self.output_dir / "counter.txt"
        if counter_filepath.is_file():
            with open(counter_filepath, "r") as fp:
                run_no = int(fp.read())
        else:
            run_no = 0
        if increment:
            run_no += 1
        with open(counter_filepath, "w") as fp:
            fp.write(str(run_no))

        return run_no

    def generate_grid(self, save_format: str = "json") -> List[Dict[str, Any]]:
        """
        Generate the multiverse grid from the stored dimensions.

        Args:
            save_format: Format to save the multiverse grid in.
                Options: "json", "csv", "none".

        Returns:
            A list of dicts containing the settings for different universes.
        """
        self.grid = generate_multiverse_grid(self.dimensions, self.constraints)

        save_format = save_format.lower()
        if save_format == "json":
            with open(self.output_dir / "multiverse_grid.json", "w") as fp:
                json.dump(self.grid, fp, indent=2)
        elif save_format == "csv":
            grid_df = pd.DataFrame(self.grid)
            grid_df.to_csv(self.output_dir / "multiverse_grid.csv", index=False)
        elif save_format != "none":
            logger.warning(
                f"Unknown save_format: {save_format}. Multiverse grid not saved."
            )

        return self.grid

    def aggregate_data(
        self, include_errors: bool = True, save: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate the data from all universes into a single DataFrame.

        Args:
            include_errors: Whether to include error information.
            save: Whether to save the aggregated data to a file.

        Returns:
            A pandas DataFrame containing the aggregated data from all universes.
        """
        data_dir = self.get_run_dir(sub_directory="data")
        csv_files = list(data_dir.glob("*.csv"))

        if include_errors:
            error_dir = self.get_run_dir(sub_directory=ERRORS_DIR_NAME)
            csv_files += list(error_dir.glob("*.csv"))

        if len(csv_files) == 0:
            logger.warning("No data files to aggregate, returning empty dataframe.")
            df = pd.DataFrame({"mv_universe_id": []})
        else:
            df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

        if include_errors:
            # Ensure error columns exist even if there are no errors
            for col in [ERROR_TYPE_COLUMN, ERROR_MESSAGE_COLUMN]:
                if col not in df.columns:
                    df[col] = pd.Series(dtype="string")

        if save:
            df.to_csv(data_dir / ("agg_" + str(self.run_no) + "_run_outputs.csv.gz"))

        return df

    def check_missing_universes(self) -> MissingUniverseInfo:
        """
        Check if any universes from the multiverse have not yet been visited.

        Returns:
            A dictionary containing:
                - missing_universe_ids: IDs of universes not yet run
                - extra_universe_ids: IDs found in data but not in multiverse grid
                - error_universe_ids: IDs of universes that errored out
                - missing_universes: Dictionaries for the missing universes
                - error_universes: Dictionaries for the errored universes
                - error_universes_by_type: Dictionary mapping error types to lists of errored universes
        """
        multiverse_dict = add_ids_to_multiverse_grid(
            self.generate_grid(save_format="none")
        )
        all_universe_ids = set(multiverse_dict.keys())

        # Get all runs including errors
        all_data = self.aggregate_data(include_errors=True, save=False)

        # Split into successful and error runs
        error_data = all_data[all_data["mv_error_type"].notna()]
        success_data = all_data[all_data["mv_error_type"].isna()]

        # Determine universe IDs
        universe_ids_with_data = set(success_data["mv_universe_id"])
        error_universe_ids = set(error_data["mv_universe_id"])
        missing_universe_ids = all_universe_ids - (
            universe_ids_with_data | error_universe_ids
        )
        extra_universe_ids = universe_ids_with_data - all_universe_ids

        # Get universe dicts
        missing_universes = [multiverse_dict[u_id] for u_id in missing_universe_ids]
        error_universes = [
            multiverse_dict[u_id]
            for u_id in error_universe_ids
            if u_id in multiverse_dict
        ]
        error_universes_by_type = {}
        for error_type in error_data["mv_error_type"].unique():
            type_ids = set(
                error_data[error_data["mv_error_type"] == error_type]["mv_universe_id"]
            )
            error_universes_by_type[error_type] = [
                multiverse_dict[u_id] for u_id in type_ids if u_id in multiverse_dict
            ]

        if (
            len(missing_universe_ids) > 0
            or len(extra_universe_ids) > 0
            or len(error_universe_ids) > 0
        ):
            logger.warning(
                f"Found {len(missing_universe_ids)} missing / "
                f"{len(error_universe_ids)} errored / "
                f"{len(extra_universe_ids)} additional universe ids!"
            )

        return {
            "missing_universe_ids": list(missing_universe_ids),
            "extra_universe_ids": list(extra_universe_ids),
            "error_universe_ids": list(error_universe_ids),
            "missing_universes": missing_universes,
            "error_universes": error_universes,
            "error_universes_by_type": error_universes_by_type,
        }

    def examine_multiverse(
        self, multiverse_grid: List[Dict[str, Any]] = None, n_jobs: int = -2
    ) -> None:
        """
        Run the analysis for all universes in the multiverse.

        Args:
            multiverse_grid: A list of dictionaries containing the settings for different universes.
            n_jobs: The number of jobs to run in parallel. Defaults to -2 (all CPUs but one).
                Use -1 for all CPUs, positive integers for specific number of CPUs,
                or 1 to disable parallel processing.

        Returns:
            None
        """
        if multiverse_grid is None:
            multiverse_grid = self.grid or self.generate_grid(save_format="none")

        n_jobs = n_jobs if n_jobs > 0 else calculate_cpu_count(n_jobs)

        # Run analysis for all universes
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(),
            TaskProgressColumn(),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:
            task_id = progress.add_task("Running", total=len(multiverse_grid))
            if n_jobs == 1:
                logger.info("Running in single-threaded mode.")
                for universe_params in multiverse_grid:
                    self.visit_universe(universe_params)
                    progress.update(task_id, advance=1)
                    # Somehow automatic updating is not working in single threaded mode, so we manually refresh
                    progress.refresh()
            else:
                logger.info(f"Running in parallel mode (n_jobs = {n_jobs}).")
                with rich_joblib(progress, task_id):
                    # For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
                    # Thus for n_jobs = -2, all CPUs but one are used
                    Parallel(n_jobs=n_jobs)(
                        delayed(self.visit_universe)(universe_params)
                        for universe_params in multiverse_grid
                    )

    def visit_universe(self, universe_dimensions: Dict[str, str]) -> None:
        """
        Run the complete analysis for a single universe.

        Output from the analysis will be saved to a file in the run's output
        directory.

        Args:
            universe_dimensions: A dictionary containing the parameters
                for the universe.

        Returns:
            None
        """
        # Generate universe ID
        universe_id = generate_universe_id(universe_dimensions)
        logger.debug(f"Visiting universe: {universe_id}")

        # Clean up any old error fiels
        error_path = self._get_error_filepath(universe_id)
        if error_path.is_file():
            logger.warning(
                f"Removing old error file: {error_path}. This should only happen during a re-run."
            )
            error_path.unlink()

        universe_filetype = self.universe_file.suffix

        # Generate final command
        output_dir = self.get_run_dir(sub_directory="universes")
        output_filename = f"nb_{self.run_no}-{universe_id}{universe_filetype}"
        output_path = output_dir / output_filename

        # Ensure output dir exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare settings dictionary
        settings = {
            "universe_id": universe_id,
            "dimensions": universe_dimensions,
            "run_no": self.run_no,
            "output_dir": str(self.output_dir),
            "seed": self.seed,
        }
        settings_str = json.dumps(settings, sort_keys=True)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                if universe_filetype == ".ipynb":
                    self.execute_notebook_via_api(
                        input_path=str(self.universe_file),
                        output_path=str(output_path),
                        parameters={
                            "settings": settings_str,
                        },
                    )
                elif universe_filetype == ".py":
                    self.execute_python_script(
                        input_path=str(self.universe_file),
                        output_path=str(output_path),
                        parameters=settings,
                    )
                else:
                    raise ValueError("Universe file must be a .ipynb or .py file.")

                for warning in w:
                    logger.warning(
                        f"Warning in universe {universe_id}: {warning.message}"
                    )
        except Exception as e:
            error_filename = "E_" + output_filename
            logger.error(f"Error in universe {universe_id} ({error_filename})")
            # Rename notebook file to indicate error
            error_output_path = output_dir / error_filename
            output_path.rename(error_output_path)
            if self.stop_on_error:
                raise e
            else:
                logger.exception(e)
                self.save_error(universe_id, universe_dimensions, e)

    def _get_error_filepath(self, universe_id: str) -> Path:
        error_dir = self.get_run_dir(sub_directory=ERRORS_DIR_NAME)
        error_filename = "e_" + str(self.run_no) + "-" + universe_id + ".csv"

        return error_dir / error_filename

    def save_error(self, universe_id: str, dimensions: dict, error: Exception) -> None:
        """
        Save an error to a file.

        Args:
            universe_id: The ID of the universe that caused the error.
            error: The exception that was raised.

        Returns:
            None
        """
        error_type = type(error).__name__
        if error_type == "PapermillExecutionError":
            error_type = error.ename

        df_error = add_universe_info_to_df(
            pd.DataFrame(
                {
                    "mv_error_type": [error_type],
                    "mv_error": [str(error)],
                }
            ),
            universe_id=universe_id,
            run_no=self.run_no,
            dimensions=dimensions,
        )
        error_path = self._get_error_filepath(universe_id)
        df_error.to_csv(error_path, index=False)

    def execute_notebook_via_api(
        self, input_path: str, output_path: str, parameters: Dict[str, str]
    ) -> None:
        """
        Executes a notebook via the papermill python API.

        Args:
            input_path: The path to the input notebook.
            output_path: The path to the output notebook.
            parameters: A dictionary containing the parameters for the notebook.

        Returns:
            None
        """
        pm.execute_notebook(
            input_path,
            output_path,
            parameters=parameters,
            progress_bar=False,
            kernel_manager_class="multiversum.IPCKernelManager.IPCKernelManager",
            execution_timeout=self.cell_timeout,
        )

    def execute_python_script(
        self, input_path: str, output_path: Optional[str], parameters: Dict[str, Any]
    ):
        global_dict = {SCRIPT_GLOBAL_OVERWRITE_NAME: parameters}

        if output_path is not None:
            # Capture output
            script_output_capture = io.StringIO()
            with contextlib.redirect_stdout(script_output_capture):
                runpy.run_path(input_path, init_globals=global_dict)
        else:
            # Keep output as-is
            runpy.run_path(input_path, init_globals=global_dict)

        # Copy input file to output file if no output file is specified
        if output_path is not None:
            with open(input_path, "r") as input_file:
                with open(output_path, "w") as output_file:
                    # Prepend brief statement and parameters
                    output_file.write("# Generated by multiversum\n")
                    output_file.write(
                        "# Note: This file is only for illustrative purposes and the analysis itself may behave slightly differently.\n"
                    )
                    output_file.write(
                        f"{SCRIPT_GLOBAL_OVERWRITE_NAME} = {json.dumps(parameters, indent=4)}\n\n"
                    )
                    # Prepend output from running the script
                    script_output = script_output_capture.getvalue()
                    script_output_escaped = script_output.replace("\n", "\n# ")
                    output_file.write(f"# Output:\n# {script_output_escaped}\n\n")

                    # Copy over script
                    output_file.write(input_file.read())

    def generate_minimal_grid(self) -> List[Dict[str, Any]]:
        """
        Generate a minimal multiverse grid that contains each unique option at least once.

        This creates a smaller grid compared to the full factorial design, where each unique
        option in each dimension appears at least once. This can be useful for testing or
        quick validation of all options.

        Returns:
            A list of dicts containing the settings for different universes.
        """
        return generate_minimal_multiverse_grid(self.dimensions, self.constraints)

    def cpu_count(self) -> int:
        """
        Get the number of CPUs available.

        Returns:
            The number of CPUs available on the system.
        """
        return cpu_count()
