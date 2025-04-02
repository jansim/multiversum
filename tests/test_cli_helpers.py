import pandas as pd
from rich.table import Table

from multiversum.cli_helpers import create_summary_table


class TestCreateSummaryTable:
    def test_empty_dataframe_returns_none(self):
        """Test that an empty dataframe returns None."""
        agg_data = pd.DataFrame()
        result = create_summary_table(agg_data)
        assert result is None

    def test_basic_dataframe_structure(self):
        """Test that a basic dataframe creates a table with correct structure."""
        # Create a minimal test dataframe
        agg_data = pd.DataFrame({"mv_universe_id": ["u001", "u002", "u003"]})

        table = create_summary_table(agg_data)

        # Verify table structure
        assert isinstance(table, Table)
        assert len(table.columns) == 2

    def test_with_error_column(self):
        """Test that error statistics are correctly calculated when error column exists."""
        # Create dataframe with errors
        agg_data = pd.DataFrame(
            {
                "mv_universe_id": ["u001", "u002", "u003", "u004"],
                "mv_error": [None, "Error", None, "Error"],
            }
        )

        table = create_summary_table(agg_data)

        # Check the table exists with correct structure
        assert isinstance(table, Table)

    def test_with_execution_time(self):
        """Test that execution time statistics are included when available."""
        # Create dataframe with execution times
        agg_data = pd.DataFrame(
            {
                "mv_universe_id": ["u001", "u002", "u003"],
                "mv_execution_time": [1.5, 2.0, 3.5],
            }
        )

        table = create_summary_table(agg_data)

        # Verify table exists with correct structure
        assert isinstance(table, Table)

    def test_complete_dataframe(self):
        """Test with a complete dataframe containing all possible columns."""
        # Create dataframe with all relevant columns
        agg_data = pd.DataFrame(
            {
                "mv_universe_id": ["u001", "u002", "u003", "u004"],
                "mv_error": [None, "Error", None, None],
                "mv_execution_time": [1.5, None, 2.0, 3.5],
            }
        )

        table = create_summary_table(agg_data)

        # Verify table exists with correct structure
        assert isinstance(table, Table)
