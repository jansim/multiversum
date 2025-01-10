import logging
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

import pytest
from multiversum import (
    generate_multiverse_grid,
    generate_universe_id,
    MultiverseAnalysis,
    Universe,
)

from pathlib import Path
import shutil

import os

from multiversum.helpers import add_universe_info_to_df

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

    def test_edge_cases(self):
        # Test with empty dimensions
        with pytest.raises(ValueError):
            generate_multiverse_grid({})
        with pytest.raises(AssertionError):
            generate_multiverse_grid({"x": "hello"})
        with pytest.raises(AssertionError):
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


class TestMultiverseAnalysis:
    def test_config_json(self):
        mv = MultiverseAnalysis(
            config_file=TEST_DIR / "notebooks" / "simple_a.json", run_no=0
        )
        assert mv.dimensions == {
            "x": ["A", "B"],
            "y": ["A", "B"],
        }

    def test_config_toml(self):
        mv = MultiverseAnalysis(
            config_file=TEST_DIR / "notebooks" / "simple_b.toml", run_no=0
        )
        assert mv.dimensions == {
            "x": ["B", "C"],
            "y": ["B", "C"],
        }

    def test_noteboook_simple(self):
        output_dir = get_temp_dir("test_MultiverseAnalysis_noteboook_simple")
        mv = MultiverseAnalysis(
            {
                "x": ["A", "B"],
                "y": ["A", "B"],
            },
            notebook=TEST_DIR / "notebooks" / "simple.ipynb",
            output_dir=output_dir,
        )
        mv.examine_multiverse()

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv") == 4
        assert count_files(output_dir, "runs/1/notebooks/*.ipynb") == 4
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
            {
                "x": ["A", "B"],
                "y": ["A", "B"],
            },
            notebook=TEST_DIR / "notebooks" / "error.ipynb",
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
        assert count_files(output_dir, "runs/1/notebooks/*.ipynb") == 4
        assert count_files(output_dir, "counter.txt") == 1

        # Check whether missing universes remain
        with pytest.warns(UserWarning):
            missing_info = mv.check_missing_universes()
        assert len(missing_info["missing_universe_ids"]) == 2
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

    def test_noteboook_timeout(self):
        output_dir = get_temp_dir("test_MultiverseAnalysis_noteboook_timeout")
        mv = MultiverseAnalysis(
            {
                "x": ["A", "B"],
                "y": ["A"],
            },
            notebook=TEST_DIR / "notebooks" / "slow.ipynb",
            output_dir=output_dir,
        )
        mv.cell_timeout = 1
        with pytest.raises(TimeoutError):
            mv.examine_multiverse()

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv") == 0
        assert count_files(output_dir, "runs/1/notebooks/*.ipynb") == 2
        assert count_files(output_dir, "counter.txt") == 1

        # Check whether missing universes remain
        with pytest.warns(UserWarning):
            missing_info = mv.check_missing_universes()
        assert len(missing_info["missing_universe_ids"]) == 2
        assert len(missing_info["extra_universe_ids"]) == 0

    def test_noteboook_timeout_without_stop(self):
        output_dir = get_temp_dir(
            "test_MultiverseAnalysis_noteboook_timeout_without_stop"
        )
        mv = MultiverseAnalysis(
            {
                "x": ["A", "B"],
                "y": ["A"],
            },
            notebook=TEST_DIR / "notebooks" / "slow.ipynb",
            output_dir=output_dir,
            cell_timeout=1,
            stop_on_error=False,
        )
        mv.examine_multiverse()

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv") == 0
        assert count_files(output_dir, "runs/1/notebooks/*.ipynb") == 2
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
            {
                "x": ["A", "B"],
                "y": ["A", "B"],
            },
            notebook=TEST_DIR / "notebooks" / "simple.ipynb",
            output_dir=output_dir,
        )
        mv.visit_universe({"x": "A", "y": "B"})
        assert count_files(output_dir, "runs/1/notebooks/*.ipynb") == 1


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
        notebook = TEST_DIR / "notebooks" / "simple.ipynb"
        config = TEST_DIR / "notebooks" / "simple_a.json"

        # Run a test multiverse analysis via the CLI
        os.system(
            f"python -m multiversum --notebook {notebook} --config {config} --output-dir {output_dir}"
        )

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv.gz") == 1
        assert count_files(output_dir, "runs/1/data/*.csv") == 4
        assert count_files(output_dir, "runs/1/notebooks/*.ipynb") == 4
        assert count_files(output_dir, "counter.txt") == 1
        assert count_files(output_dir, "multiverse_grid.json") == 1

    def test_multiverse_py_empty(self):
        output_dir = get_temp_dir("test_multiverse_py_empty")
        notebook = TEST_DIR / "notebooks" / "simple.ipynb"

        # Run a test multiverse analysis via the CLI
        wd = os.getcwd()
        os.chdir(TEST_DIR / "notebooks")
        os.system(
            f"python -m multiversum --notebook {notebook} --output-dir {output_dir}"
        )
        os.chdir(wd)

        # Check whether all expected files are there
        assert count_files(output_dir, "runs/1/data/*.csv.gz") == 0
        assert count_files(output_dir, "runs/1/data/*.csv") == 0
        assert count_files(output_dir, "runs/1/notebooks/*.ipynb") == 0
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
