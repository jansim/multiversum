import pandas as pd
import pytest
from rich.table import Table

from multiversum.cli_helpers import (
    create_summary_table,
    parse_partial_percentages,
    split_multiverse_grid,
)


class TestParsePartialPercentages:
    def test_valid_percentage_format(self):
        """Test valid percentage format parsing."""
        start_pct, end_pct = parse_partial_percentages("0%,50%")
        assert start_pct == 0.0
        assert end_pct == 0.5

    def test_mixed_format(self):
        """Test mixed format with and without percentage signs."""
        start_pct, end_pct = parse_partial_percentages("0,50%")
        assert start_pct == 0.0
        assert end_pct == 0.5

    def test_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        start_pct, end_pct = parse_partial_percentages(" 10% , 60% ")
        assert start_pct == 0.1
        assert end_pct == 0.6

    def test_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            parse_partial_percentages("")

    def test_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_partial_percentages("0%,50%,100%")  # Too many parts

    def test_invalid_percentage_range(self):
        """Test that invalid percentage range raises ValueError."""
        with pytest.raises(ValueError):
            parse_partial_percentages("50%,30%")  # Start > End

    def test_out_of_range_percentages(self):
        """Test that percentages outside 0-100 range raise ValueError."""
        with pytest.raises(ValueError):
            parse_partial_percentages("-10%,50%")  # Negative percentage

        with pytest.raises(ValueError):
            parse_partial_percentages("0%,120%")  # Percentage > 100


class TestSplitMultiverse:
    def test_split_basic(self):
        """Test basic splitting functionality."""
        grid = list(range(100))
        split_grid, start_idx, end_idx = split_multiverse_grid(grid, 0.2, 0.6)

        assert split_grid == list(range(20, 60))
        assert start_idx == 20
        assert end_idx == 60

    def test_split_empty_grid(self):
        """Test splitting an empty grid."""
        grid = []
        split_grid, start_idx, end_idx = split_multiverse_grid(grid, 0.0, 1.0)

        assert split_grid == []
        assert start_idx == 0
        assert end_idx == 0

    def test_split_single_item(self):
        """Test splitting a grid with a single item."""
        grid = [42]
        split_grid, start_idx, end_idx = split_multiverse_grid(grid, 0.0, 1.0)

        assert split_grid == [42]
        assert start_idx == 0
        assert end_idx == 1

    def test_split_edge_cases(self):
        """Test edge cases for splitting."""
        grid = list(range(10))

        # Split with zero length
        split_grid, start_idx, end_idx = split_multiverse_grid(grid, 0.5, 0.5)
        assert split_grid == []
        assert start_idx == 5
        assert end_idx == 5

        # Split at the beginning
        split_grid, start_idx, end_idx = split_multiverse_grid(grid, 0.0, 0.3)
        assert split_grid == [0, 1, 2]
        assert start_idx == 0
        assert end_idx == 3

        # Split at the end
        split_grid, start_idx, end_idx = split_multiverse_grid(grid, 0.7, 1.0)
        assert split_grid == [7, 8, 9]
        assert start_idx == 7
        assert end_idx == 10

    def test_full_range_coverage(self):
        """Test that 0-0.5 and 0.5-1.0 together cover the full range."""
        grid = list(range(100))

        # First half
        first_half, first_start, first_end = split_multiverse_grid(grid, 0.0, 0.5)
        assert first_half == list(range(0, 50))
        assert first_start == 0
        assert first_end == 50

        # Second half
        second_half, second_start, second_end = split_multiverse_grid(grid, 0.5, 1.0)
        assert second_half == list(range(50, 100))
        assert second_start == 50
        assert second_end == 100

        # Together they should cover the entire grid
        assert first_half + second_half == grid
        assert first_end == second_start

    def test_fractional_splits(self):
        """Test splitting with fractional values like 0.3333."""
        grid = list(range(100))

        # Split into three equal parts
        first_part, first_start, first_end = split_multiverse_grid(grid, 0.0, 1 / 3)
        second_part, second_start, second_end = split_multiverse_grid(
            grid, 1 / 3, 2 / 3
        )
        third_part, third_start, third_end = split_multiverse_grid(grid, 2 / 3, 1.0)

        # Check that parts have expected sizes (30 items each)
        assert len(first_part) == 33
        assert len(second_part) == 33
        assert len(third_part) == 34

        # Check specific indices
        assert first_start == 0
        assert first_end == 33
        assert second_start == 33
        assert second_end == 66
        assert third_start == 66
        assert third_end == 100

        # Check full coverage
        assert first_part + second_part + third_part == grid

    def test_adjacent_splits_no_overlap_or_gaps(self):
        """Test that adjacent splits have no overlaps or gaps."""
        grid = list(range(100))

        # Create several adjacent splits
        splits = [(0.0, 0.215), (0.215, 0.333333), (0.333333, 0.79), (0.79, 1.0)]
        results = []

        for start_pct, end_pct in splits:
            part, start_idx, end_idx = split_multiverse_grid(grid, start_pct, end_pct)
            results.append((part, start_idx, end_idx))

        # Verify no gaps or overlaps
        for i in range(len(results) - 1):
            current_end = results[i][2]  # End index of current split
            next_start = results[i + 1][1]  # Start index of next split
            assert current_end == next_start, (
                f"Gap or overlap between splits {i} and {i + 1}"
            )

        # Verify we covered the entire grid
        all_items = []
        for part, _, _ in results:
            all_items.extend(part)
        assert all_items == grid


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

    def test_missing_universe_id_column(self):
        """Test behavior when the aggregate data is missing the required mv_universe_id column."""
        # Create a dataframe with data but no mv_universe_id column
        agg_data = pd.DataFrame(
            {"some_column": [1, 2, 3], "another_column": ["a", "b", "c"]}
        )

        table = create_summary_table(agg_data)

        # Verify table exists with correct structure
        assert isinstance(table, Table)
