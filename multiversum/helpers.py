import pandas as pd
from itertools import count
from typing import Optional

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
