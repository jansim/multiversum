import itertools
import json
import os
from hashlib import md5
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .logger import logger


def add_universe_info_to_df(
    data: pd.DataFrame,
    universe_id: str,
    run_no: int,
    dimensions: dict,
    execution_time: Optional[float] = None,
) -> pd.DataFrame:
    """
    Add general universe / run info to the dataframe.

    Args:
        data: Dataframe to add the info to.
        universe_id: Universe ID.
        run_no: Run number.
        dimensions: Dictionary with dimensions.
        execution_time: Execution time.
    """
    if len(data.index) == 0:
        logger.warning(
            "Index of data is empty, adding one entry using universe_id to be able to add data."
        )
        data.index = [universe_id]

    index = count()
    data.insert(next(index), "mv_universe_id", universe_id)
    data.insert(next(index), "mv_run_no", run_no)
    data.insert(next(index), "mv_execution_time", execution_time)

    # Add info about dimensions
    dimensions_sorted = sorted(dimensions.keys())
    for dimension in dimensions_sorted:
        value = dimensions[dimension]
        if isinstance(value, (list, dict)):
            value = json.dumps(value, sort_keys=True)
        data.insert(next(index), f"mv_dim_{dimension}", value)
    return data


def validate_dimensions(dimensions: Dict[str, Any]) -> Tuple[Tuple[str, ...], List]:
    """
    Validate the dimensions dictionary for multiverse grid generation.

    Args:
        dimensions: A dictionary where keys are dimension names and values are lists
            of possible values for each dimension.

    Returns:
        A tuple containing:
            - A tuple of dimension names (keys)
            - A list of dimension values (values)

    Raises:
        ValueError: If dimensions are empty or contain invalid values.
    """
    if not dimensions:
        raise ValueError("No (or empty) dimensions provided.")

    keys, values = zip(*dimensions.items())
    if not all(isinstance(k, str) for k in keys):
        raise ValueError("All dimension names must be strings.")
    if not all(isinstance(v, list) for v in values):
        raise ValueError("All dimension values must be lists.")

    values_conv = []
    for dim in values:
        dim_conv = []
        for v in dim:
            if isinstance(v, (list, dict)):
                v_hashable = json.dumps(v, sort_keys=True)
                dim_conv.append(v_hashable)
            else:
                dim_conv.append(v)
        values_conv.append(dim_conv)

    if any(len(dim) != len(set(dim)) for dim in values_conv):
        raise ValueError("Dimensions must not contain duplicate values.")

    return keys, values


def generate_minimal_multiverse_grid(
    dimensions: Dict[str, Any],
    constraints: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a minimal multiverse grid that contains each unique option at least once.

    This creates a smaller grid compared to the full factorial design, where each unique
    option in each dimension appears at least once. This can be useful for testing or
    quick validation of all options.

    Args:
        dimensions: A dictionary where keys are dimension names and values are lists
            of possible values for each dimension.
        constraints: Optional dictionary where keys are dimension names and values are lists of constraints.
            Each constraint is a dictionary with:
                - value: The value of the dimension that the constraint applies to.
                - allowed_if: A dictionary of dimension-value pairs that must be present for the constraint to be allowed.
                - forbidden_if: A dictionary of dimension-value pairs that must not be present for the constraint to be allowed.
            Only one of allowed_if and forbidden_if can be present in a constraint.

    Returns:
        A list of dicts containing the settings for different universes.
    """
    keys, values_conv = validate_dimensions(dimensions)

    # Get the dimension with the most options
    max_options = max(len(options) for options in values_conv)

    minimal_grid = []
    # Create one universe for each index up to the max number of options
    for i in range(max_options):
        universe = {}
        for dim_name, options in zip(keys, values_conv):
            # Use modulo to cycle through options
            universe[dim_name] = options[i % len(options)]
        minimal_grid.append(universe)

    if constraints:
        # Apply constraints to filter out invalid combinations
        minimal_grid = apply_constraints(minimal_grid, constraints)

        # Check if we need to add more combinations to ensure all valid values are represented
        missing_values = find_missing_values(minimal_grid, dimensions, constraints)

        if missing_values:
            # Generate additional combinations to include missing values
            full_grid = generate_multiverse_grid(dimensions, constraints)

            # Add combinations from full grid that contain missing values until all values are represented
            for universe in full_grid:
                if not any(universe == existing for existing in minimal_grid):
                    for dim, values in missing_values.items():
                        if universe[dim] in values:
                            minimal_grid.append(universe)
                            # Update missing values
                            missing_values = find_missing_values(
                                minimal_grid, dimensions, constraints
                            )
                            if not missing_values:
                                break
                    if not missing_values:
                        break

    return minimal_grid


def find_missing_values(
    grid: List[Dict[str, Any]],
    dimensions: Dict[str, Any],
    constraints: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, set]:
    """
    Find values from dimensions that are not represented in the grid but are valid according to constraints.

    Args:
        grid: List of dictionaries containing the current grid combinations
        dimensions: Dictionary of dimension names and their possible values
        constraints: Optional dictionary of constraints

    Returns:
        Dictionary mapping dimension names to sets of their missing valid values
    """
    missing_values = {}

    # Generate full grid with constraints to know all valid combinations
    full_grid = generate_multiverse_grid(dimensions, constraints)

    # For each dimension, find values that appear in the full grid but not in the minimal grid
    for dim_name, values in dimensions.items():
        values_in_full = {universe[dim_name] for universe in full_grid}
        values_in_minimal = {universe[dim_name] for universe in grid}
        missing = values_in_full - values_in_minimal
        if missing:
            missing_values[dim_name] = missing

    return missing_values


def generate_multiverse_grid(
    dimensions: Dict[str, List[str]],
    constraints: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a full grid from a dictionary of dimensions.

    Args:
        dimensions: A dictionary containing Lists with options.
        constraints: An optional dictionary containing constraints for dimensions.

    Returns:
        A list of dicts containing all different combinations of the options.
    """
    keys, values_conv = validate_dimensions(dimensions)

    # from https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    multiverse_grid = [dict(zip(keys, v)) for v in itertools.product(*values_conv)]

    if constraints:
        multiverse_grid = apply_constraints(multiverse_grid, constraints)

    return multiverse_grid


def apply_constraints(
    multiverse_grid: List[Dict[str, Any]], constraints: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Apply constraints to filter out nonsensical dimension combinations.

    Args:
        multiverse_grid: A list of dictionaries containing the settings for different universes.
        constraints: A dictionary containing constraints for dimensions.
            Keys in the dict correspond to dimensions, values are lists of constraints.
            Each constraint is a dictionary of the following structure:
                - value: The value of the dimension that the constraint applies to.
                - allowed_if: A dictionary of dimension-value pairs that must be present for the constraint to be allowed.
                - forbidden_if: A dictionary of dimension-value pairs that must not be present for the constraint to be allowed.
            Only one of allowed_if and forbidden_if can be present in a constraint.

    Returns:
        A filtered list of dictionaries containing the settings for different universes.
    """

    def is_allowed(universe: Dict[str, Any], constraint: Dict[str, Any]) -> bool:
        if "allowed_if" in constraint:
            for key, value in constraint["allowed_if"].items():
                if universe.get(key) != value:
                    return False
        if "forbidden_if" in constraint:
            for key, value in constraint["forbidden_if"].items():
                if universe.get(key) == value:
                    return False
        return True

    filtered_grid = []
    for universe in multiverse_grid:
        valid = True
        for dimension, dimension_constraints in constraints.items():
            for constraint in dimension_constraints:
                if universe[dimension] == constraint["value"] and not is_allowed(
                    universe, constraint
                ):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            filtered_grid.append(universe)

    return filtered_grid


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


def search_files(file: Any, default_files: List[str]) -> Optional[Path]:
    if file is not None and (isinstance(file, str) or isinstance(file, Path)):
        file_path = Path(file)
        if file_path.is_file():
            return file_path
        else:
            raise FileNotFoundError
    else:
        for default_file in default_files:
            default_file_path = Path(default_file)
            if default_file_path.is_file():
                return default_file_path

    return None


def calculate_cpu_count(n_jobs: int) -> int:
    """
    Calculate the actual number of CPUs to use based on the n_jobs parameter.

    Args:
        n_jobs: Number of jobs specified by the user.
               -1 means use all available CPUs.
               -2 means use all but one CPU.
               Positive values are used as is.

    Returns:
        The actual number of CPUs to use.
    """
    assert n_jobs != 0, "n_jobs cannot be 0."

    cpu_count = os.cpu_count() or 1  # Default to 1 if cpu_count returns None

    if n_jobs < 0:
        # For negative values, use cpu_count + n_jobs (e.g., -1 → all CPUs, -2 → all but one)
        return max(1, cpu_count + n_jobs + 1)
    else:
        if n_jobs > cpu_count:
            logger.warning(
                f"Requested {n_jobs} jobs, but only {cpu_count} CPUs detected."
            )
            return cpu_count
        return n_jobs
