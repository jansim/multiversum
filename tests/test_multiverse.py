import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from multiversum import (
    Config,
    MultiverseAnalysis,
    Universe,
)
from multiversum.helpers import (
    add_universe_info_to_df,
    find_missing_values,
    generate_minimal_multiverse_grid,
    generate_multiverse_grid,
    generate_universe_id,
    validate_dimensions,
)

ROOT_DIR = Path(__file__).parent.parent
TEST_DIR = ROOT_DIR / "tests"
TEMP_DIR = TEST_DIR / "temp"

shutil.rmtree(TEMP_DIR, ignore_errors=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def get_temp_dir(name):
    new_dir = TEMP_DIR / name
    new_dir.mkdir()
    return new_dir


def count_files(dir, glob):
    return len(list(dir.glob(glob)))


class TestGenerateMultiverseGrid:
    def test_grid(self):
        assert generate_multiverse_grid({"x": [1, 2], "y": [3, 4]}) == [
            {"x": 1, "y": 3},
            {"x": 1, "y": 4},
            {"x": 2, "y": 3},
            {"x": 2, "y": 4},
        ]

    def test_grid_duplicates_error(self):
        with pytest.raises(ValueError):
            generate_multiverse_grid({"x": [1, 2], "y": [3, 3, 4]})

    def test_grid_sub_universes(self):
        assert generate_multiverse_grid({"x": [1, 2], "y": [[3, 4]]}) == [
            {"x": 1, "y": [3, 4]},
            {"x": 2, "y": [3, 4]},
        ]

    def test_edge_cases(self):
        # Test with empty dimensions
        with pytest.raises(ValueError):
            generate_multiverse_grid({})
        with pytest.raises(ValueError):
            generate_multiverse_grid({"x": "hello"})
        with pytest.raises(ValueError):
            generate_multiverse_grid({12: [1, 2, 3]})
        # Test with single dimension
        assert generate_multiverse_grid({"x": [1, 2, 3]}) == [
            {"x": 1},
            {"x": 2},
            {"x": 3},
        ]

        # Test with multiple dimensions with single value
        assert generate_multiverse_grid({"x": [1], "y": [2], "z": [3]}) == [
            {"x": 1, "y": 2, "z": 3}
        ]

    def test_apply_constraints(self):
        dimensions = {
            "fizz": ["A", "B", "C"],
            "buzz": ["always-allowed", "only-allowed-with-B", "allowed-except-for-C"],
        }

        constraints = {
            "buzz": [
                {
                    "value": "only-allowed-with-B",
                    "allowed_if": {"fizz": "B"},
                },
                {
                    "value": "allowed-except-for-C",
                    "forbidden_if": {"fizz": "C"},
                },
            ]
        }

        filtered_grid = generate_multiverse_grid(dimensions, constraints)

        expected_grid = [
            {"fizz": "A", "buzz": "always-allowed"},
            {"fizz": "A", "buzz": "allowed-except-for-C"},
            {"fizz": "B", "buzz": "always-allowed"},
            {"fizz": "B", "buzz": "only-allowed-with-B"},
            {"fizz": "B", "buzz": "allowed-except-for-C"},
            {"fizz": "C", "buzz": "always-allowed"},
        ]

        assert filtered_grid == expected_grid


class TestMultiverseAnalysis:
    def test_config_json(self):
        mv = MultiverseAnalysis(
            config=TEST_DIR / "scenarios" / "multiverse_simple_a.json", run_no=0
        )
        assert mv.dimensions == {
            "x": ["A", "B"],
            "y": ["A", "B"],
        }

    def test_config_toml(self):
        mv = MultiverseAnalysis(
            config=TEST_DIR / "scenarios" / "multiverse_simple_b.toml", run_no=0
        )
        assert mv.dimensions == {
            "x": ["B", "C"],
            "y": ["B", "C"],
        }

    def test_config_py(self):
        mv = MultiverseAnalysis(
            config=TEST_DIR / "scenarios" / "multiverse_simple_c.py", run_no=0
        )
        assert mv.dimensions == {
            "x": ["C", "D"],
            "y": ["C", "D"],
        }

    def test_noteboook_simple(self):
        output_dir = get_temp_dir("test_MultiverseAnalysis_noteboook_simple")
        mv = MultiverseAnalysis(
            dimensions={
                "x": ["A", "B"],
                "y": ["A", "B"],
            },
            universe=TEST_DIR / "scenarios" / "universe_simple.ipynb",
            output_dir=output_dir,
        )
        mv.examine_multiverse(n_jobs=1)

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv") == 4
        assert count_files(output_dir, "runs/1/universes/*.ipynb") == 4
        assert count_files(output_dir, "counter.txt") == 1

        # Check whether data aggregation works
        aggregated_data = mv.aggregate_data(save=False)
        assert not aggregated_data.empty
        assert "value" in aggregated_data.columns

        # Check whether missing universes remain
        missing_info = mv.check_missing_universes()
        assert len(missing_info["missing_universe_ids"]) == 0
        assert len(missing_info["extra_universe_ids"]) == 0

    def test_noteboook_simple_py(self):
        output_dir = get_temp_dir("test_MultiverseAnalysis_noteboook_simple_py")
        mv = MultiverseAnalysis(
            dimensions={
                "x": ["A", "B"],
                "y": ["A", "B"],
            },
            universe=TEST_DIR / "scenarios" / "universe_simple.py",
            output_dir=output_dir,
        )
        mv.examine_multiverse(n_jobs=1)

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv") == 4
        assert count_files(output_dir, "runs/1/universes/*.py") == 4
        assert count_files(output_dir, "counter.txt") == 1

        # Check whether data aggregation works
        aggregated_data = mv.aggregate_data(save=False)
        assert not aggregated_data.empty
        assert "value" in aggregated_data.columns

        # Check whether missing universes remain
        missing_info = mv.check_missing_universes()
        assert len(missing_info["missing_universe_ids"]) == 0
        assert len(missing_info["extra_universe_ids"]) == 0

    def test_noteboook_error(self, caplog):
        output_dir = get_temp_dir("test_MultiverseAnalysis_noteboook_error")
        mv = MultiverseAnalysis(
            dimensions={
                "x": ["A", "B"],
                "y": ["A", "B"],
            },
            universe=TEST_DIR / "scenarios" / "universe_error.ipynb",
            output_dir=output_dir,
        )
        mv.stop_on_error = False
        with caplog.at_level(logging.ERROR, logger="multiversum"):
            # Important: Logs are only captured correctly when *not* running in parallel
            mv.examine_multiverse(n_jobs=1)

        error_msg_count = 0
        for record in caplog.records:
            message = record.getMessage().lower()
            if "error in universe" in message:
                error_msg_count += 1
        assert error_msg_count == 2

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv") == 2
        assert count_files(output_dir, "runs/1/universes/*.ipynb") == 4
        assert count_files(output_dir, "runs/1/universes/E_*.ipynb") == 2, (
            "Notebooks with errors are highlighted"
        )
        assert count_files(output_dir, "counter.txt") == 1

        # Check whether missing universes remain
        with caplog.at_level(logging.WARNING):
            missing_info = mv.check_missing_universes()
        assert "Found 0 missing / 2 errored / 0 additional" in caplog.text
        assert len(missing_info["missing_universe_ids"]) == 0
        assert len(missing_info["extra_universe_ids"]) == 0

        # Check whether errors correctly show up in final data
        aggregated_data = mv.aggregate_data(save=False)
        assert aggregated_data.shape[0] == 4
        assert_series_equal(
            aggregated_data["mv_error_type"],
            pd.Series(
                [np.nan, np.nan, "ValueError", "ValueError"], name="mv_error_type"
            ),
        )

    def test_noteboook_timeout(self, caplog):
        output_dir = get_temp_dir("test_MultiverseAnalysis_noteboook_timeout")
        mv = MultiverseAnalysis(
            dimensions={
                "x": ["A", "B"],
                "y": ["A"],
            },
            universe=TEST_DIR / "scenarios" / "universe_slow.ipynb",
            output_dir=output_dir,
        )
        mv.cell_timeout = 1
        with pytest.raises(TimeoutError):
            mv.examine_multiverse()

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv") == 0
        assert count_files(output_dir, "runs/1/universes/*.ipynb") == 2
        assert count_files(output_dir, "counter.txt") == 1

        # Check whether missing universes remain
        with caplog.at_level(logging.WARNING):
            missing_info = mv.check_missing_universes()
        assert "Found 2 missing / 0 errored / 0 additional" in caplog.text
        assert len(missing_info["missing_universe_ids"]) == 2
        assert len(missing_info["extra_universe_ids"]) == 0

    def test_noteboook_timeout_without_stop(self):
        output_dir = get_temp_dir(
            "test_MultiverseAnalysis_noteboook_timeout_without_stop"
        )
        mv = MultiverseAnalysis(
            dimensions={
                "x": ["A", "B"],
                "y": ["A"],
            },
            universe=TEST_DIR / "scenarios" / "universe_slow.ipynb",
            output_dir=output_dir,
            cell_timeout=1,
            stop_on_error=False,
        )
        mv.examine_multiverse()

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv") == 0
        assert count_files(output_dir, "runs/1/universes/*.ipynb") == 2
        assert count_files(output_dir, "runs/1/universes/E_*.ipynb") == 2, (
            "Notebooks with errors are highlighted"
        )
        assert count_files(output_dir, "counter.txt") == 1

        # Check whether errors correctly show up in final data
        aggregated_data = mv.aggregate_data(save=False)
        assert aggregated_data.shape[0] == 2
        assert_series_equal(
            aggregated_data["mv_error_type"],
            pd.Series(["CellTimeoutError", "CellTimeoutError"], name="mv_error_type"),
        )

    def test_generate_universe_id(self):
        universe_id = generate_universe_id({"x": "A", "y": "B"})
        assert universe_id == "47899ae546a9854ebfe2de7396eff9fa"

    def test_generate_universe_id_order_invariance(self):
        assert generate_universe_id({"x": "A", "y": "B"}) == generate_universe_id(
            {"y": "B", "x": "A"}
        )

    def test_visit_universe(self):
        output_dir = get_temp_dir("test_MultiverseAnalysis_visit_universe")
        mv = MultiverseAnalysis(
            dimensions={
                "x": ["A", "B"],
                "y": ["A", "B"],
            },
            universe=TEST_DIR / "scenarios" / "universe_simple.ipynb",
            output_dir=output_dir,
        )
        mv.visit_universe({"x": "A", "y": "B"})
        assert count_files(output_dir, "runs/1/universes/*.ipynb") == 1

    def test_check_missing_universes_with_errors(self):
        output_dir = get_temp_dir(
            "test_MultiverseAnalysis_check_missing_universes_with_errors"
        )
        mv = MultiverseAnalysis(
            dimensions={
                "x": ["A", "B", "C"],
                "y": ["A", "B"],
            },
            universe=TEST_DIR / "scenarios" / "universe_error.ipynb",
            output_dir=output_dir,
            stop_on_error=False,
        )
        mv.examine_multiverse(n_jobs=1)

        missing_info = mv.check_missing_universes()

        # Test error universe IDs and objects
        assert len(missing_info["error_universe_ids"]) > 0
        assert len(missing_info["error_universes"]) > 0

        # Test error universes by type
        error_types = missing_info["error_universes_by_type"]
        assert "ValueError" in error_types
        assert len(error_types["ValueError"]) > 0

        # Verify each error universe has correct structure
        for error_universe in missing_info["error_universes"]:
            assert "x" in error_universe
            assert "y" in error_universe

    def test_check_missing_universes_without_errors(self):
        output_dir = get_temp_dir(
            "test_MultiverseAnalysis_check_missing_universes_without_errors"
        )
        mv = MultiverseAnalysis(
            dimensions={
                "x": ["A", "B"],
                "y": ["A", "B"],
            },
            universe=TEST_DIR / "scenarios" / "universe_simple.ipynb",
            output_dir=output_dir,
        )
        mv.examine_multiverse(n_jobs=1)

        missing_info = mv.check_missing_universes()

        # Verify no errors were found
        assert len(missing_info["error_universe_ids"]) == 0
        assert len(missing_info["error_universes"]) == 0
        assert len(missing_info["error_universes_by_type"]) == 0

        # Verify other fields are also empty (successful run)
        assert len(missing_info["missing_universe_ids"]) == 0
        assert len(missing_info["extra_universe_ids"]) == 0

    def test_check_missing_universes_multiple_error_types(self):
        output_dir = get_temp_dir(
            "test_MultiverseAnalysis_check_missing_universes_multiple_error_types"
        )
        mv = MultiverseAnalysis(
            dimensions={
                "x": ["A", "B"],
                "y": ["A"],
            },
            universe=TEST_DIR / "scenarios" / "universe_slow.ipynb",
            output_dir=output_dir,
            cell_timeout=1,
            stop_on_error=False,
        )
        mv.examine_multiverse(n_jobs=1)

        # Add a different type of error manually and re-run check
        # Note: This is a bit of an odd way of interfering with error handling,
        # so could more easily lead to tests failing in the future
        dims = mv.generate_minimal_grid()[0]
        u_id = generate_universe_id(dims)
        mv.save_error(u_id, dims, ValueError("Test error"))

        # Ensure error file is loaded by running check again
        missing_info = mv.check_missing_universes()

        print("TESTOUTPUT123")
        print(missing_info["error_universes_by_type"])

        # Should have both timeout and value errors
        error_types = missing_info["error_universes_by_type"]
        assert "CellTimeoutError" in error_types
        assert "ValueError" in error_types
        # Both original universes timeout, but we've overwritten one of them
        assert len(error_types["CellTimeoutError"]) == 1
        assert len(error_types["ValueError"]) == 1

        # Total errors should match sum of individual error types
        total_errors = sum(len(errors) for errors in error_types.values())
        assert len(missing_info["error_universes"]) == total_errors

    def test_aggregate_data_error_columns(self):
        output_dir = get_temp_dir("test_aggregate_data_error_columns")
        mv = MultiverseAnalysis(
            dimensions={
                "x": ["A", "B"],
                "y": ["A", "B"],
            },
            universe=TEST_DIR / "scenarios" / "universe_simple.ipynb",
            output_dir=output_dir,
        )
        mv.examine_multiverse(n_jobs=1)

        # Test with include_errors=True (should have error columns)
        data_with_errors = mv.aggregate_data(include_errors=True, save=False)
        assert "mv_error_type" in data_with_errors.columns
        assert "mv_error" in data_with_errors.columns
        assert data_with_errors["mv_error_type"].dtype == "string"
        assert data_with_errors["mv_error"].dtype == "string"

        # Test with include_errors=False (should not have error columns)
        data_without_errors = mv.aggregate_data(include_errors=False, save=False)
        assert "mv_error_type" not in data_without_errors.columns
        assert "mv_error" not in data_without_errors.columns


class TestConfig:
    def test_config(self):
        config = Config(
            dimensions={
                "x": ["A", "B"],
                "y": ["A", "B"],
            }
        )
        mv = MultiverseAnalysis(config=config)

        assert mv.generate_grid(save_format="none") == generate_multiverse_grid(
            {
                "x": ["A", "B"],
                "y": ["A", "B"],
            }
        )

    def test_save_grid_format(self):
        output_dir = get_temp_dir("test_save_grid_format")
        mv = MultiverseAnalysis(
            dimensions={
                "x": ["A", "B"],
                "y": ["A", "B"],
            },
            output_dir=output_dir,
        )

        # Test saving as JSON
        mv.generate_grid(save_format="json")
        assert count_files(output_dir, "multiverse_grid.json") == 1

        # Test saving as CSV
        mv.generate_grid(save_format="csv")
        assert count_files(output_dir, "multiverse_grid.csv") == 1

        # Test not saving
        mv.generate_grid(save_format="none")
        # No additional files should be created
        assert count_files(output_dir, "multiverse_grid.*") == 2


class TestUniverse:
    def test_add_universe_info(self):
        uv = Universe(settings={"dimensions": {"hello": "world"}})

        df = uv._add_universe_info(pd.DataFrame({"test_value": [42]}))
        # Drop execution time because it will always change
        df.drop(["mv_execution_time"], axis="columns", inplace=True)

        pd.testing.assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "mv_universe_id": ["no-universe-id-provided"],
                    "mv_run_no": 0,
                    "mv_dim_hello": "world",
                    "test_value": 42,
                }
            ),
        )

    def test_get_execution_time(self):
        uv = Universe(settings={"dimensions": {"hello": "world"}})
        execution_time = uv.get_execution_time()
        assert execution_time >= 0

    def test_save_data(self):
        output_dir = get_temp_dir("test_Universe_save_data")
        uv = Universe(
            settings={"dimensions": {"hello": "world"}, "output_dir": str(output_dir)}
        )
        data = pd.DataFrame({"test_value": [42]})
        uv.save_data(data)
        assert count_files(output_dir, "runs/0/data/*.csv") == 1

    def test_generate_sub_universes(self):
        uv = Universe(
            settings={"dimensions": {"x": ["A", "B"], "y": ["A", "B"], "z": "C"}}
        )
        sub_universes = uv.generate_sub_universes()
        assert len(sub_universes) == 4
        assert sub_universes == [
            {"x": "A", "y": "A", "z": "C"},
            {"x": "A", "y": "B", "z": "C"},
            {"x": "B", "y": "A", "z": "C"},
            {"x": "B", "y": "B", "z": "C"},
        ]

    def test_aggregate_data_no_files(self):
        output_dir = get_temp_dir("test_aggregate_data_no_files")
        mv = MultiverseAnalysis(
            dimensions={"x": ["A", "B"], "y": ["A", "B"]},
            output_dir=output_dir,
        )
        aggregated_data = mv.aggregate_data(save=False)
        assert aggregated_data.empty

    def test_manual_save_error(self):
        output_dir = get_temp_dir("test_save_error")
        mv = MultiverseAnalysis(
            dimensions={"x": ["A", "B"], "y": ["A", "B"]},
            output_dir=output_dir,
        )
        mv.save_error("test_universe", {}, Exception("Test exception"))
        error_file = output_dir / "runs/1/errors/e_1-test_universe.csv"
        assert error_file.is_file()
        error_data = pd.read_csv(error_file)
        assert error_data["mv_universe_id"].iloc[0] == "test_universe"
        assert error_data["mv_error_type"].iloc[0] == "Exception"
        assert error_data["mv_error"].iloc[0] == "Test exception"


class TestCLI:
    def test_simple(self):
        output_dir = get_temp_dir("test_CLI_simple")
        notebook = TEST_DIR / "scenarios" / "universe_simple.ipynb"
        config = TEST_DIR / "scenarios" / "multiverse_simple_a.json"

        # Run a test multiverse analysis via the CLI
        os.system(
            f"python -m multiversum --universe {notebook} --config {config} --output-dir {output_dir}"
        )

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv.gz") == 1
        assert count_files(output_dir, "runs/1/data/*.csv") == 4
        assert count_files(output_dir, "runs/1/universes/*.ipynb") == 4
        assert count_files(output_dir, "counter.txt") == 1
        assert count_files(output_dir, "multiverse_grid.json") == 1

    def test_multiverse_py_empty(self):
        output_dir = get_temp_dir("test_multiverse_py_empty")
        notebook = TEST_DIR / "scenarios" / "universe_simple.ipynb"

        # Run a test multiverse analysis via the CLI
        wd = os.getcwd()
        os.chdir(TEST_DIR / "scenarios")
        os.system(
            f"python -m multiversum --notebook {notebook} --output-dir {output_dir}"
        )
        os.chdir(wd)

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv.gz") == 0
        assert count_files(output_dir, "runs/1/data/*.csv") == 0
        assert count_files(output_dir, "runs/1/universes/*.ipynb") == 0
        assert count_files(output_dir, "counter.txt") == 0
        assert count_files(output_dir, "multiverse_grid.json") == 0


class TestHelpers:
    def test_add_universe_info_to_df_standard(self):
        data = pd.DataFrame({"test_value": [42]})
        data = add_universe_info_to_df(data, "test_universe", 0, {"hello": "world"})

        pd.testing.assert_frame_equal(
            data,
            pd.DataFrame(
                {
                    "mv_universe_id": ["test_universe"],
                    "mv_run_no": [0],
                    "mv_execution_time": [None],
                    "mv_dim_hello": ["world"],
                    "test_value": [42],
                }
            ),
        )

    def test_add_universe_info_to_df_empty(self):
        data = pd.DataFrame()
        data = add_universe_info_to_df(data, "test_universe", 0, {"hello": "world"})

        pd.testing.assert_frame_equal(
            data,
            pd.DataFrame(
                {
                    "mv_universe_id": ["test_universe"],
                    "mv_run_no": [0],
                    "mv_execution_time": [None],
                    "mv_dim_hello": ["world"],
                },
                index=["test_universe"],
            ),
        )

    def test_validate_dimensions_valid(self):
        dimensions = {"x": [1, 2], "y": [3, 4]}
        keys, values = validate_dimensions(dimensions)
        assert keys == ("x", "y")
        assert values == ([1, 2], [3, 4])

    def test_validate_dimensions_with_dict(self):
        dimensions = {"x": [1, 2], "y": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
        keys, values = validate_dimensions(dimensions)
        assert keys == tuple(dimensions.keys())
        assert values == tuple(dimensions.values())

    def test_validate_dimensions_with_nested_lists(self):
        dimensions = {"x": [1, 2], "y": [[3, 4], [5, 6]]}
        keys, values = validate_dimensions(dimensions)
        assert keys == ("x", "y")
        assert values == ([1, 2], [[3, 4], [5, 6]])

    def test_validate_dimensions_empty(self):
        with pytest.raises(ValueError, match="No \\(or empty\\) dimensions provided."):
            validate_dimensions({})

    def test_validate_dimensions_invalid_key_type(self):
        with pytest.raises(ValueError, match="All dimension names must be strings."):
            validate_dimensions({123: [1, 2]})

    def test_validate_dimensions_invalid_value_type(self):
        with pytest.raises(ValueError, match="All dimension values must be lists."):
            validate_dimensions({"x": "not_a_list"})

    def test_validate_dimensions_duplicates(self):
        with pytest.raises(
            ValueError, match="Dimensions must not contain duplicate values."
        ):
            validate_dimensions({"x": [1, 1, 2]})

    def test_generate_minimal_multiverse_grid_basic(self):
        dimensions = {"x": [1, 2], "y": [3, 4]}
        grid = generate_minimal_multiverse_grid(dimensions)
        assert len(grid) == 2
        assert grid == [
            {"x": 1, "y": 3},
            {"x": 2, "y": 4},
        ]

    def test_generate_minimal_multiverse_grid_uneven(self):
        dimensions = {"x": [1, 2, 3], "y": [4, 5]}
        grid = generate_minimal_multiverse_grid(dimensions)
        assert len(grid) == 3
        assert grid == [
            {"x": 1, "y": 4},
            {"x": 2, "y": 5},
            {"x": 3, "y": 4},  # Cycles back to first y value
        ]

    def test_generate_minimal_multiverse_grid_single_value(self):
        dimensions = {"x": [1], "y": [2]}
        grid = generate_minimal_multiverse_grid(dimensions)
        assert len(grid) == 1
        assert grid == [{"x": 1, "y": 2}]

    def test_generate_minimal_multiverse_grid_with_nested_lists(self):
        dimensions = {"x": [1, 2], "y": [[3, 4], [5, 6]]}
        grid = generate_minimal_multiverse_grid(dimensions)
        assert len(grid) == 2
        assert grid == [
            {"x": 1, "y": [3, 4]},
            {"x": 2, "y": [5, 6]},
        ]

    def test_generate_minimal_multiverse_grid_with_constraints(self):
        dimensions = {
            "fizz": ["A", "B", "C"],
            "buzz": ["always-allowed", "only-allowed-with-B", "allowed-except-for-C"],
        }

        constraints = {
            "buzz": [
                {
                    "value": "only-allowed-with-B",
                    "allowed_if": {"fizz": "B"},
                },
                {
                    "value": "allowed-except-for-C",
                    "forbidden_if": {"fizz": "C"},
                },
            ]
        }

        grid = generate_minimal_multiverse_grid(dimensions, constraints)

        # Check that all valid values appear at least once
        fizz_values = {universe["fizz"] for universe in grid}
        buzz_values = {universe["buzz"] for universe in grid}
        assert fizz_values == {"A", "B", "C"}
        assert buzz_values == {
            "always-allowed",
            "only-allowed-with-B",
            "allowed-except-for-C",
        }

        # Check that constraints are respected
        for universe in grid:
            if universe["buzz"] == "only-allowed-with-B":
                assert universe["fizz"] == "B"
            if universe["buzz"] == "allowed-except-for-C":
                assert universe["fizz"] != "C"

    def test_find_missing_values(self):
        dimensions = {
            "x": ["A", "B", "C"],
            "y": ["1", "2", "3"],
        }

        # Create a grid missing some values
        grid = [
            {"x": "A", "y": "1"},
            {"x": "B", "y": "2"},
        ]

        missing = find_missing_values(grid, dimensions)
        assert missing == {
            "x": {"C"},
            "y": {"3"},
        }

        # Test with constraints
        constraints = {
            "y": [
                {
                    "value": "3",
                    "forbidden_if": {"x": "C"},
                },
            ]
        }

        missing_with_constraints = find_missing_values(grid, dimensions, constraints)
        assert missing_with_constraints == {
            "x": {"C"},
            "y": {"3"},
        }

    def test_add_universe_info_to_df(self):
        data = pd.DataFrame({"test_value": [42]})
        universe_id = "test_universe"
        run_no = 1
        dimensions = {
            "dim1": "val1",
            "dim2": "val2",
            "dim3": {"key3": "val3"},
            "dim4": ["val4"],
        }
        execution_time = 123.456

        result = add_universe_info_to_df(
            data, universe_id, run_no, dimensions, execution_time
        )

        expected = pd.DataFrame(
            {
                "mv_universe_id": [universe_id],
                "mv_run_no": [run_no],
                "mv_execution_time": [execution_time],
                "mv_dim_dim1": ["val1"],
                "mv_dim_dim2": ["val2"],
                "mv_dim_dim3": ['{"key3": "val3"}'],
                "mv_dim_dim4": ['["val4"]'],
                "test_value": [42],
            }
        )

        pd.testing.assert_frame_equal(result, expected)
