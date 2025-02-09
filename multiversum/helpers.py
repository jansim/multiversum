import itertools
import json
from hashlib import md5
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        data.insert(next(index), f"mv_dim_{dimension}", dimensions[dimension])
    return data


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
    if not dimensions:
        raise ValueError("No (or empty) dimensions provided.")

    keys, values = zip(*dimensions.items())
    assert all(isinstance(k, str) for k in keys)
    assert all(isinstance(v, list) for v in values)
    # If we have lists of lists for dimensions (as is the case for sub-universes),
    # we need to convert them to tuples to make them hashable
    values_conv = [
        [tuple(v) if isinstance(v, list) else v for v in dim] for dim in values
    ]

    if any(len(dim) != len(set(dim)) for dim in values_conv):
        raise ValueError("Dimensions must not contain duplicate values.")

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
