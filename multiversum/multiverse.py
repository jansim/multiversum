"""
This module contains helper functions to orchestrate a multiverse analysis.
"""

import itertools
from pathlib import Path
from typing import Dict, List, Optional, TypedDict
from hashlib import md5
import subprocess
import json
import warnings
import pandas as pd
import papermill as pm
from tqdm import tqdm
from joblib import Parallel, delayed
from .parallel import tqdm_joblib


def generate_multiverse_grid(dimensions: Dict[str, List[str]]):
    """
    Generate a full grid from a dictionary of dimensions.

    Args:
    - dimensions: A dictionary containing Lists with options.

    Returns:
    - A list of dicts containing all different combinations of the options.
    """
    # from https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    keys, values = zip(*dimensions.items())
    multiverse_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return multiverse_grid


class MissingUniverseInfo(TypedDict):
    missing_universe_ids: List[str]
    extra_universe_ids: List[str]
    missing_universes: List[Dict[str, str]]


class MultiverseAnalysis:
    def __init__(
        self,
        dimensions: Dict,
        output_dir: Path = Path("./output"),
        run_no: Optional[int] = None,
        new_run: bool = True,
        seed: Optional[int] = 2023,
    ) -> None:
        """
        Initializes a new MultiverseAnalysis instance.

        This is a helper class to run a full multiverse analysis.

        Args:
        - dimensions: A dictionary containing the dimensions of the multiverse.
            Each dimension corresponds to a decision.
        - output_dir: The directory to store the output in.
        - run_no: The number of the current run. Defaults to an automatically
            incrementing integer number if new_run is True or the last run if
            new_run is False.
        - new_run: Whether this is a new run or not. Defaults to True.
        - seed: The seed to use for the analysis.
        """
        self.dimensions = dimensions
        self.output_dir = output_dir
        self.seed=seed
        self.run_no = (
            run_no if run_no is not None else self.read_counter(increment=new_run)
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
        - increment: Whether to increment the counter after reading.

        Returns:
        - The current value of the counter.
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

    def generate_grid(self, save=True):
        """
        Generate the multiverse grid from the stored dimensions.

        Args:
        - save: Whether to save the multiverse grid to a JSON file.

        Returns:
        - A list of dicts containing the settings for different universes.
        """
        self.grid = generate_multiverse_grid(self.dimensions)
        if save:
            with open(self.output_dir / "multiverse_grid.json", "w") as fp:
                json.dump(self.grid, fp, indent=2)
        return self.grid

    def aggregate_data(self, save=True) -> pd.DataFrame:
        """
        Aggregate the data from all universes into a single DataFrame.

        Args:
        - save: Whether to save the aggregated data to a file.

        Returns:
        - A pandas DataFrame containing the aggregated data from all universes.
        """
        data_dir = self.get_run_dir(sub_directory="data")
        csv_files = data_dir.glob("*.csv")

        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

        if save:
            df.to_csv(data_dir / ("agg_" + str(self.run_no) + "_run_outputs.csv.gz"))

        return df

    def check_missing_universes(self) -> MissingUniverseInfo:
        """
        Check if any universes from the multiverse have not yet been visited.

        Returns:
        - A dictionary containing the missing universe ids, additional
            universe ids (i.e. not in the current multiverse_grid)
            and the dictionaries for the missing universes.
        """
        multiverse_grid = self.generate_grid(save=False)
        multiverse_dict = {
            self.generate_universe_id(u_params): u_params
            for u_params in multiverse_grid
        }
        all_universe_ids = set(multiverse_dict.keys())

        aggregated_data = self.aggregate_data(save=False)
        universe_ids_with_data = set(aggregated_data["universe_id"])

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

    def generate_universe_id(self, universe_parameters):
        # Note: Getting stable hashes seems to be easier said than done in Python
        # See https://stackoverflow.com/questions/5884066/hashing-a-dictionary/22003440#22003440
        return md5(
            json.dumps(universe_parameters, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def examine_multiverse(self, multiverse_grid, n_jobs=-2):
        # Run analysis for all universes
        with tqdm_joblib(
            tqdm(desc="Visiting Universes", total=len(multiverse_grid))
        ) as progress_bar:  # noqa: F841
            # For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
            # Thus for n_jobs = -2, all CPUs but one are used
            Parallel(n_jobs=n_jobs)(
                delayed(self.visit_universe)(universe_params)
                for universe_params in multiverse_grid
            )

    def visit_universe(self, universe_parameters: Dict[str, str]):
        """
        Run the complete analysis for a single universe.

        Output from the analysis will be saved to a file in the run's output
        directory.

        Args:
            universe_parameters: A dictionary containing the parameters
            for the universe.

        Returns:
            None
        """
        # Generate universe ID
        universe_id = self.generate_universe_id(universe_parameters)

        # Generate parameter string
        universe_param_string = json.dumps(universe_parameters, sort_keys=True)

        # Generate final command
        output_dir = self.get_run_dir(sub_directory="notebooks")
        output_filename = "m_" + str(self.run_no) + "-" + universe_id + ".ipynb"

        # Ensure output dir exists
        output_dir.mkdir(parents=True, exist_ok=True)

        execute_notebook_via_api(
            input_path="universe_analysis.ipynb",
            output_path=str(output_dir / output_filename),
            parameters={
                "universe_id": universe_id,
                "run_no": str(self.run_no),
                "universe": universe_param_string,
                "output_dir": str(self.output_dir),
                "seed": str(self.seed),
            },
        )


def execute_notebook_via_cli(
    input_path: str, output_path: str, parameters: Dict[str, str]
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
    for key, value in parameters.items():
        call_params.append("-p")
        call_params.append(key)
        call_params.append(value)

    print(" ".join(call_params))
    # Call papermill render
    process = subprocess.run(call_params, capture_output=True, text=True)
    print(process.stdout)
    print(process.stderr)


def execute_notebook_via_api(
    input_path: str, output_path: str, parameters: Dict[str, str]
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
        input_path, output_path, parameters=parameters, progress_bar=False
    )
