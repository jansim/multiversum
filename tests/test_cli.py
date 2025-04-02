from unittest import mock

import pytest
from click.testing import CliRunner

from multiversum.__main__ import cli


@pytest.fixture
def mock_multiverse_analysis():
    """Mock the MultiverseAnalysis class for testing"""
    with mock.patch("multiversum.__main__.MultiverseAnalysis") as mock_ma:
        # Set up mock instance
        instance = mock_ma.return_value
        instance.run_no = 1
        instance.seed = 80539

        # Mock the generate_grid method to return a list of dictionaries
        instance.generate_grid.return_value = [
            {"dim1": "val1", "dim2": "val1"},
            {"dim1": "val1", "dim2": "val2"},
            {"dim1": "val2", "dim2": "val1"},
            {"dim1": "val2", "dim2": "val2"},
        ]

        # Mock the check_missing_universes method
        instance.check_missing_universes.return_value = {
            "missing_universe_ids": [],
            "extra_universe_ids": [],
            "missing_universes": [],
        }

        # Mock the generate_minimal_grid method
        instance.generate_minimal_grid.return_value = [
            {"dim1": "val1", "dim2": "val1"},
            {"dim1": "val2", "dim2": "val2"},
        ]

        yield mock_ma


@pytest.fixture
def runner():
    """Set up the Click CLI test runner"""
    return CliRunner()


def test_cli_grid_only(runner, mock_multiverse_analysis):
    """Test the --grid-only flag"""
    result = runner.invoke(cli, ["--grid-only"])
    assert result.exit_code == 0

    # Verify the MultiverseAnalysis was initialized correctly
    mock_multiverse_analysis.assert_called_once()
    mock_multiverse_analysis.return_value.generate_grid.assert_called_once()

    # Check that examine_multiverse was not called
    mock_multiverse_analysis.return_value.examine_multiverse.assert_not_called()

    # Verify the output contains the right information
    assert "Grid Export Only" in result.output
    assert "N = 4" in result.output


def test_cli_grid_format(runner, mock_multiverse_analysis):
    """Test different grid format options"""
    # Test CSV format
    result = runner.invoke(cli, ["--grid-only", "--grid-format", "csv"])
    assert result.exit_code == 0
    mock_multiverse_analysis.return_value.generate_grid.assert_called_with(
        save_format="csv"
    )

    # Reset mock
    mock_multiverse_analysis.reset_mock()

    # Test JSON format
    result = runner.invoke(cli, ["--grid-only", "--grid-format", "json"])
    assert result.exit_code == 0
    mock_multiverse_analysis.return_value.generate_grid.assert_called_with(
        save_format="json"
    )

    # Reset mock
    mock_multiverse_analysis.reset_mock()

    # Test none format with warning message capture
    with mock.patch("multiversum.__main__.logger.warning") as mock_warning:
        result = runner.invoke(cli, ["--grid-only", "--grid-format", "none"])
        assert result.exit_code == 0
        mock_multiverse_analysis.return_value.generate_grid.assert_called_with(
            save_format="none"
        )

        # Check that the warning message was logged
        mock_warning.assert_any_call(
            mock.ANY  # Match any warning message
        )


def test_cli_continue_mode(runner, mock_multiverse_analysis):
    """Test the continue mode"""
    # Mock missing universes for continue mode
    mock_multiverse_analysis.return_value.check_missing_universes.return_value = {
        "missing_universe_ids": ["u1", "u2"],
        "extra_universe_ids": [],
        "missing_universes": [
            {"dim1": "val1", "dim2": "val1"},
            {"dim1": "val2", "dim2": "val2"},
        ],
    }

    result = runner.invoke(cli, ["--mode", "continue"])
    assert result.exit_code == 0

    # Check that MultiverseAnalysis was initialized with new_run=False
    mock_multiverse_analysis.assert_called_once()
    args, kwargs = mock_multiverse_analysis.call_args
    assert kwargs.get("new_run") is False

    # Verify examine_multiverse was called with missing universes
    missing_universes = (
        mock_multiverse_analysis.return_value.check_missing_universes.return_value[
            "missing_universes"
        ]
    )
    mock_multiverse_analysis.return_value.examine_multiverse.assert_called_with(
        missing_universes, n_jobs=mock.ANY
    )

    # Verify the output contains the right mode information
    assert "Continuing Previous Run" in result.output


def test_cli_test_mode(runner, mock_multiverse_analysis):
    """Test the test mode"""
    result = runner.invoke(cli, ["--mode", "test"])
    assert result.exit_code == 0

    # Verify minimal grid was generated and used
    mock_multiverse_analysis.return_value.generate_minimal_grid.assert_called_once()
    minimal_grid = (
        mock_multiverse_analysis.return_value.generate_minimal_grid.return_value
    )
    mock_multiverse_analysis.return_value.examine_multiverse.assert_called_with(
        minimal_grid, n_jobs=mock.ANY
    )

    # Verify the output contains the right mode information
    assert "Test Run" in result.output
    assert "minimal test grid with" in result.output


def test_cli_universe_id(runner, mock_multiverse_analysis):
    """Test specifying a universe ID"""
    # Mock the add_ids_to_multiverse_grid function
    with mock.patch("multiversum.__main__.add_ids_to_multiverse_grid") as mock_add_ids:
        mock_add_ids.return_value = {
            "test-id-123": {"dim1": "val1", "dim2": "val1"},
            "other-id-456": {"dim1": "val2", "dim2": "val2"},
        }

        result = runner.invoke(cli, ["--u-id", "test-id"])
        assert result.exit_code == 0

        # Verify that the right universe was selected
        mock_multiverse_analysis.return_value.examine_multiverse.assert_called_with(
            [{"dim1": "val1", "dim2": "val1"}], n_jobs=mock.ANY
        )

        # Verify the output contains the universe ID
        assert "Running only universe:" in result.output
        assert "test-id-123" in result.output


def test_cli_n_jobs(runner, mock_multiverse_analysis):
    """Test different n_jobs settings"""
    with mock.patch("multiversum.__main__.calculate_cpu_count") as mock_calc_cpu:
        mock_calc_cpu.return_value = 4

        # Test with specified n_jobs
        result = runner.invoke(cli, ["--n-jobs", "2"])
        assert result.exit_code == 0

        # Verify calculate_cpu_count was called with the right parameter
        mock_calc_cpu.assert_called_with(2)

        # Verify examine_multiverse was called with the right n_jobs
        mock_multiverse_analysis.return_value.examine_multiverse.assert_called_with(
            mock.ANY, n_jobs=4
        )

        # Check output contains CPU info
        assert "CPUs: Using 4" in result.output


def test_cli_output_dir(runner, mock_multiverse_analysis):
    """Test specifying an output directory"""
    # Don't patch Path directly, instead check the MultiverseAnalysis constructor args
    result = runner.invoke(cli, ["--output-dir", "./custom_output"])
    assert result.exit_code == 0

    # Verify MultiverseAnalysis was initialized with the right output_dir
    mock_multiverse_analysis.assert_called_once()
    args, kwargs = mock_multiverse_analysis.call_args

    # Check that the path refers to the same directory, regardless of ./ prefix
    output_dir = kwargs.get("output_dir")
    assert output_dir.name == "custom_output"
    assert str(output_dir).endswith("custom_output")


def test_cli_seed(runner, mock_multiverse_analysis):
    """Test specifying a custom seed"""
    result = runner.invoke(cli, ["--seed", "12345"])
    assert result.exit_code == 0

    # Verify MultiverseAnalysis was initialized with the right seed
    mock_multiverse_analysis.assert_called_once()
    args, kwargs = mock_multiverse_analysis.call_args
    assert kwargs.get("seed") == 12345

    # Check output contains seed info (either custom or mocked value)
    assert "Seed: 12345" in result.output or "Seed: 80539" in result.output


def test_cli_partial_parallel_mode(runner, mock_multiverse_analysis):
    """Test the partial-parallel mode with percentage ranges"""
    # Mock the parse_partial_percentages and split_multiverse_grid functions
    with mock.patch(
        "multiversum.__main__.parse_partial_percentages"
    ) as mock_parse, mock.patch(
        "multiversum.__main__.split_multiverse_grid"
    ) as mock_split:
        # Set up mocks to return predictable values
        mock_parse.return_value = (0.0, 0.5)  # 0% to 50%
        partial_grid = [
            {"dim1": "val1", "dim2": "val1"},
            {"dim1": "val1", "dim2": "val2"},
        ]
        mock_split.return_value = (
            partial_grid,
            0,
            2,
        )  # Partial grid with start/end indices

        # Invoke CLI with partial-parallel mode
        result = runner.invoke(
            cli, ["--mode", "partial-parallel", "--partial", "0%,50%"]
        )
        assert result.exit_code == 0

        # Check that MultiverseAnalysis was initialized with new_run=False
        mock_multiverse_analysis.assert_called_once()
        args, kwargs = mock_multiverse_analysis.call_args
        assert kwargs.get("new_run") is False

        # Verify that parse_partial_percentages was called with the right string
        mock_parse.assert_called_with("0%,50%")

        # Verify that split_multiverse_grid was called with the right parameters
        mock_split.assert_called_with(mock.ANY, 0.0, 0.5)

        # Verify examine_multiverse was called with the partial grid
        mock_multiverse_analysis.return_value.examine_multiverse.assert_called_with(
            partial_grid, n_jobs=mock.ANY
        )

        # Verify the output contains the expected messages
        assert "Parallel Run (Partial)" in result.output
        assert "Finished running the partial analysis" in result.output


def test_cli_finalize_mode(runner, mock_multiverse_analysis):
    """Test the finalize mode"""
    result = runner.invoke(cli, ["--mode", "finalize"])
    assert result.exit_code == 0

    # Check that MultiverseAnalysis was initialized with new_run=False
    mock_multiverse_analysis.assert_called_once()
    args, kwargs = mock_multiverse_analysis.call_args
    assert kwargs.get("new_run") is False

    # Verify that generate_grid was called but examine_multiverse was NOT called
    mock_multiverse_analysis.return_value.generate_grid.assert_called_once()
    mock_multiverse_analysis.return_value.examine_multiverse.assert_not_called()

    # Verify that check_missing_universes was called to verify completion
    mock_multiverse_analysis.return_value.check_missing_universes.assert_called_once()

    # Verify that aggregate_data was called
    mock_multiverse_analysis.return_value.aggregate_data.assert_called()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
