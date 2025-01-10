"""
This module contains helper functions to orchestrate a multiverse analysis.
"""

import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from hashlib import md5
import subprocess
import json
import warnings
import pandas as pd
import papermill as pm
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from .parallel import tqdm_joblib
from .logger import logger
from .helpers import add_universe_info_to_df

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DEFAULT_SEED = 80539
ERRORS_DIR_NAME = "errors"


def generate_multiverse_grid(dimensions: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Generate a full grid from a dictionary of dimensions.

    Args:
        dimensions: A dictionary containing Lists with options.

    Returns:
        A list of dicts containing all different combinations of the options.
    """
    if not dimensions:
        raise ValueError("No (or empty) dimensions provided.")

    keys, values = zip(*dimensions.items())
    assert all(isinstance(k, str) for k in keys)
    assert all(isinstance(v, list) for v in values)
    if any(len(dim) != len(set(dim)) for dim in values):
        raise ValueError("Dimensions must not contain duplicate values.")

    # from https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    multiverse_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return multiverse_grid


def generate_universe_id(universe_parameters: Dict[str, Any]) -> str:
    """
    Generate a unique ID for a given universe.

    Args:
        universe_parameters: A dictionary containing the parameters for the universe.

    Returns:
        A unique ID for the universe.
    """
    # Note: Getting stable hashes seems to be easier said than done in Python
    # See https://stackoverflow.com/questions/5884066/hashing-a-dictionary/22003440#22003440
    return md5(
        json.dumps(universe_parameters, sort_keys=True).encode("utf-8")
    ).hexdigest()


def add_ids_to_multiverse_grid(
    multiverse_grid: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generates a dictionary of universe IDs mapped to their corresponding parameters.

    Args:
        multiverse_grid: A list of dictionaries, where each dictionary contains parameters for a universe.

    Returns:
        A dictionary where the keys are generated universe IDs and the values are the corresponding parameters.
    """
    return {generate_universe_id(u_params): u_params for u_params in multiverse_grid}


class MissingUniverseInfo(TypedDict):
    missing_universe_ids: List[str]
    extra_universe_ids: List[str]
    missing_universes: List[Dict[str, str]]


class MultiverseAnalysis:
    """
    This class orchestrates a multiverse analysis.

    Attributes:
        dimensions: A dictionary containing the dimensions of the multiverse.
        notebook: The Path to the notebook to run.
        config_file: A Path to a JSON file containing the dimensions.
        output_dir: The directory to store the output in.
        run_no: The number of the current run.
        new_run: Whether this is a new run or not.
        seed: The seed to use for the analysis.
        stop_on_error: Whether to stop the analysis if an error occurs.
        cell_timeout: A timeout (in seconds) for each cell in the notebook.
    """

    dimensions = None
    notebook = None
    config_file = None
    output_dir = None
    run_no = None
    new_run = None
    seed = None
    grid = None
    cell_timeout = None
    stop_on_error = True

    def __init__(
        self,
        dimensions: Optional[Dict] = None,
        notebook: Path = Path("./universe.ipynb"),
        config_file: Optional[Path] = None,
        output_dir: Path = Path("./output"),
        run_no: Optional[int] = None,
        new_run: bool = True,
        seed: Optional[int] = DEFAULT_SEED,
        stop_on_error: bool = True,
        cell_timeout: Optional[int] = None,
    ) -> None:
        """
        Initializes a new MultiverseAnalysis instance.

        Args:
            dimensions: A dictionary containing the dimensions of the multiverse.
                Each dimension corresponds to a decision.
                Alternatively a Path to a JSON file containing the dimensions.
            notebook: The Path to the notebook to run.
            config_file: A Path to a JSON file containing the dimensions.
            output_dir: The directory to store the output in.
            run_no: The number of the current run. Defaults to an automatically
                incrementing integer number if new_run is True or the last run if
                new_run is False.
            new_run: Whether this is a new run or not. Defaults to True.
            seed: The seed to use for the analysis.
            stop_on_error: Whether to stop the analysis if an error occurs.
            cell_timeout: A timeout (in seconds) for each cell in the notebook.
        """
        if isinstance(config_file, Path):
            if config_file.suffix == ".toml":
                with open(config_file, "rb") as fp:
                    config = tomllib.load(fp)
            elif config_file.suffix == ".json":
                with open(config_file, "r") as fp:
                    config = json.load(fp)
            else:
                raise ValueError("Only .toml and .json files are supported as config.")

            if "dimensions" in config:
                assert dimensions is None
                self.dimensions = config["dimensions"]

            if "stop_on_error" in config:
                self.stop_on_error = config["stop_on_error"]

        if dimensions is not None:
            self.dimensions = dimensions

        self.notebook = notebook
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.run_no = (
            run_no if run_no is not None else self.read_counter(increment=new_run)
        )
        self.stop_on_error = stop_on_error
        self.cell_timeout = cell_timeout

        if self.dimensions is None:
            raise ValueError(
                "Dimensions need to be specified either directly or in a config."
            )

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

    def generate_grid(self, save: bool = True) -> List[Dict[str, Any]]:
        """
        Generate the multiverse grid from the stored dimensions.

        Args:
            save: Whether to save the multiverse grid to a JSON file.

        Returns:
            A list of dicts containing the settings for different universes.
        """
        self.grid = generate_multiverse_grid(self.dimensions)
        if save:
            with open(self.output_dir / "multiverse_grid.json", "w") as fp:
                json.dump(self.grid, fp, indent=2)
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

        if save:
            df.to_csv(data_dir / ("agg_" + str(self.run_no) + "_run_outputs.csv.gz"))

        return df

    def check_missing_universes(self) -> MissingUniverseInfo:
        """
        Check if any universes from the multiverse have not yet been visited.

        Returns:
            A dictionary containing the missing universe ids, additional
                universe ids (i.e. not in the current multiverse_grid)
                and the dictionaries for the missing universes.
        """
        multiverse_dict = add_ids_to_multiverse_grid(self.generate_grid(save=False))
        all_universe_ids = set(multiverse_dict.keys())

        aggregated_data = self.aggregate_data(include_errors=False, save=False)
        universe_ids_with_data = set(aggregated_data["mv_universe_id"])

        missing_universe_ids = all_universe_ids - universe_ids_with_data
        extra_universe_ids = universe_ids_with_data - all_universe_ids
        missing_universes = [multiverse_dict[u_id] for u_id in missing_universe_ids]

        if len(missing_universe_ids) > 0 or len(extra_universe_ids) > 0:
            warnings.warn(
                f"Found missing {len(missing_universe_ids)} / "
                f"additional {len(extra_universe_ids)} universe ids!"
            )

        return {
            "missing_universe_ids": missing_universe_ids,
            "extra_universe_ids": extra_universe_ids,
            "missing_universes": missing_universes,
        }

    def examine_multiverse(
        self, multiverse_grid: List[Dict[str, Any]] = None, n_jobs: int = -2
    ) -> None:
        """
        Run the analysis for all universes in the multiverse.

        Args:
            multiverse_grid: A list of dictionaries containing the settings for different universes.
            n_jobs: The number of jobs to run in parallel. Defaults to -2 (all CPUs but one).

        Returns:
            None
        """
        if multiverse_grid is None:
            multiverse_grid = self.grid or self.generate_grid(save=False)

        # Run analysis for all universes
        if n_jobs == 1:
            logger.info("Running in single-threaded mode (njobs = 1).")
            for universe_params in tqdm(multiverse_grid, desc="Visiting Universes"):
                self.visit_universe(universe_params)
        else:
            logger.info(
                f"Running in parallel mode (njobs = {n_jobs}; {cpu_count()} CPUs detected)."
            )
            with tqdm_joblib(
                tqdm(desc="Visiting Universes", total=len(multiverse_grid), smoothing=0)
            ) as progress_bar:  # noqa: F841
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
            warnings.warn(
                f"Removing old error file: {error_path}. This should only happen during a re-run."
            )
            error_path.unlink()

        # Generate final command
        output_dir = self.get_run_dir(sub_directory="notebooks")
        output_filename = "m_" + str(self.run_no) + "-" + universe_id + ".ipynb"

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
            self.execute_notebook_via_api(
                input_path=str(self.notebook),
                output_path=str(output_dir / output_filename),
                parameters={
                    "settings": settings_str,
                },
            )
        except Exception as e:
            logger.error(f"Error in universe {universe_id} ({output_filename})")
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

    def execute_notebook_via_cli(
        self, input_path: str, output_path: str, parameters: Dict[str, str]
    ):
        """
        Executes a notebook via the papermill command line interface.

        Args:
            input_path: The path to the input notebook.
            output_path: The path to the output notebook.
            parameters: A dictionary containing the parameters for the notebook.

        Returns:
            None
        """
        call_params = [
            "papermill",
            input_path,
            output_path,
        ]
        if self.cell_timeout is not None:
            call_params.append("--execution-timeout")
            call_params.append(str(self.cell_timeout))

        for key, value in parameters.items():
            call_params.append("-p")
            call_params.append(key)
            call_params.append(value)

        logger.info(" ".join(call_params))
        # Call papermill render
        process = subprocess.run(call_params, capture_output=True, text=True)
        logger.info(process.stdout)
        logger.info(process.stderr)

    def execute_notebook_via_api(
        self, input_path: str, output_path: str, parameters: Dict[str, str]
    ):
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
