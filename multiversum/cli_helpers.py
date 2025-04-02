from typing import List, Optional, Tuple

import pandas as pd
from rich.table import Table


def parse_partial_percentages(partial_str: str) -> Tuple[float, float]:
    """
    Parse a string representing a percentage range for splitting the multiverse.

    Args:
        partial_str: A string in the format "start%,end%" or "start,end%", e.g. "0%,50%" or "0,20%"

    Returns:
        A tuple of two floats representing (start_percentage, end_percentage)

    Raises:
        ValueError: If the split string is not in the correct format
    """
    parts = partial_str.split(",")
    if len(parts) != 2:
        raise ValueError("Split must be in the format 'start%,end%' or 'start,end%'.")

    start_str, end_str = parts

    if not end_str.strip().endswith("%"):
        raise ValueError(
            "Non-percentages for splitting are currently not supported, but may be in the future. Please add a '%' sign to the end of the second number."
        )

    # Clean and convert to floats
    start_str = start_str.strip().rstrip("%")
    end_str = end_str.strip().rstrip("%")

    try:
        start_pct = float(start_str) / 100
        end_pct = float(end_str) / 100
    except ValueError:
        raise ValueError("Split percentages must be valid numbers")

    # Validate ranges
    if not (0 <= start_pct <= 1) or not (0 <= end_pct <= 1):
        raise ValueError("Split percentages must be between 0% and 100%")

    if start_pct >= end_pct:
        raise ValueError("Start percentage must be less than end percentage")

    return start_pct, end_pct


def split_multiverse_grid(
    multiverse_grid: List, start_pct: float, end_pct: float
) -> Tuple[List, int, int]:
    """
    Split a grid based on percentage range.

    Args:
        multiverse_grid: The grid to split
        start_pct: Start percentage as a float (0-1)
        end_pct: End percentage as a float (0-1)

    Returns:
        A tuple containing (split_grid, start_idx, end_idx)
            - split_grid: The split portion of the grid
            - start_idx: Start index in the original grid
            - end_idx: End index in the original grid
    """
    start_idx = int(len(multiverse_grid) * start_pct)
    end_idx = int(len(multiverse_grid) * end_pct)
    return multiverse_grid[start_idx:end_idx], start_idx, end_idx


def create_summary_table(agg_data: pd.DataFrame) -> Optional[Table]:
    """
    Create a rich table summarizing the multiverse analysis results.

    Args:
        agg_data: Aggregated data from the multiverse analysis

    Returns:
        A formatted table with the analysis summary, or None if data is empty
    """
    if agg_data.empty:
        return None

    # Create a summary table
    table = Table(title="Multiverse Analysis Summary", show_header=False)

    # Basic statistics
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", style="green")

    # Total universes
    total_universes = len(agg_data)
    table.add_row("Length of Agg. Data", str(total_universes))
    if "mv_universe_id" in agg_data.columns:
        table.add_row(
            "Universes in Agg. Data", str(agg_data["mv_universe_id"].nunique())
        )

    # Success rate
    table.add_section()
    error_col = "mv_error" if "mv_error" in agg_data.columns else None
    if error_col:
        failed = agg_data[error_col].notna().sum()
        success_rate = f"{(total_universes - failed) / total_universes:.1%}"
        table.add_row("Success Rate", success_rate)
        table.add_row("Failed Universes", str(failed))
    else:
        table.add_row("Success Rate", "N/A")
        table.add_row("Failed Universes", "N/A")

    # Execution time stats if available
    if "mv_execution_time" in agg_data.columns:
        table.add_section()
        exec_times = agg_data["mv_execution_time"].dropna()
        if not exec_times.empty:
            table.add_row(
                "Avg Execution Time (Min; Max)",
                f"{exec_times.mean():.2f}s ({exec_times.min():.2f}s; {exec_times.max():.2f}s)",
            )
            table.add_row("Total Execution Time", f"{exec_times.sum():.2f}s")

    return table
