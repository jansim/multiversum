from rich.table import Table


def create_summary_table(agg_data):
    """
    Create a rich table summarizing the multiverse analysis results.

    Args:
        agg_data (pandas.DataFrame): Aggregated data from the multiverse analysis

    Returns:
        rich.table.Table: A formatted table with the analysis summary, or None if data is empty
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
    table.add_row("Universes in Agg. Data", str(agg_data["mv_universe_id"].nunique()))

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
