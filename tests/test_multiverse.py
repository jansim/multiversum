import pandas as pd
from multiversum import generate_multiverse_grid, MultiverseAnalysis, Universe

from pathlib import Path
import shutil

import os

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


def test_grid():
    assert generate_multiverse_grid({"x": [1, 2], "y": [3, 4]}) == [
        {"x": 1, "y": 3},
        {"x": 1, "y": 4},
        {"x": 2, "y": 3},
        {"x": 2, "y": 4},
    ]


def test_MultiverseAnalysis_config_json():
    mv = MultiverseAnalysis(
        config_file=TEST_DIR / "notebooks" / "simple_a.json", run_no=0
    )
    assert mv.dimensions == {
        "x": ["A", "B"],
        "y": ["A", "B"],
    }


def test_MultiverseAnalysis_config_toml():
    mv = MultiverseAnalysis(
        config_file=TEST_DIR / "notebooks" / "simple_b.toml", run_no=0
    )
    assert mv.dimensions == {
        "x": ["B", "C"],
        "y": ["B", "C"],
    }


def test_MultiverseAnalysis_noteboook_simple():
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


def test_CLI_simple():
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


def test_Universe_add_universe_info():
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


def test_generate_multiverse_grid_edge_cases():
    # Test with empty dimensions
    assert generate_multiverse_grid({}) == []

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


def test_MultiverseAnalysis_generate_grid():
    mv = MultiverseAnalysis(
        {
            "x": ["A", "B"],
            "y": ["A", "B"],
        },
        notebook=TEST_DIR / "notebooks" / "simple.ipynb",
        output_dir=get_temp_dir("test_MultiverseAnalysis_generate_grid"),
    )
    grid = mv.generate_grid(save=False)
    assert len(grid) == 4
    assert grid == [
        {"x": "A", "y": "A"},
        {"x": "A", "y": "B"},
        {"x": "B", "y": "A"},
        {"x": "B", "y": "B"},
    ]


def test_MultiverseAnalysis_aggregate_data():
    output_dir = get_temp_dir("test_MultiverseAnalysis_aggregate_data")
    mv = MultiverseAnalysis(
        {
            "x": ["A", "B"],
            "y": ["A", "B"],
        },
        notebook=TEST_DIR / "notebooks" / "simple.ipynb",
        output_dir=output_dir,
    )
    mv.examine_multiverse()
    aggregated_data = mv.aggregate_data(save=False)
    assert not aggregated_data.empty
    assert "value" in aggregated_data.columns


def test_MultiverseAnalysis_check_missing_universes():
    output_dir = get_temp_dir("test_MultiverseAnalysis_check_missing_universes")
    mv = MultiverseAnalysis(
        {
            "x": ["A", "B"],
            "y": ["A", "B"],
        },
        notebook=TEST_DIR / "notebooks" / "simple.ipynb",
        output_dir=output_dir,
    )
    mv.examine_multiverse()
    missing_info = mv.check_missing_universes()
    assert len(missing_info["missing_universe_ids"]) == 0
    assert len(missing_info["extra_universe_ids"]) == 0


def test_MultiverseAnalysis_generate_universe_id():
    mv = MultiverseAnalysis(
        {
            "x": ["A", "B"],
            "y": ["A", "B"],
        },
        notebook=TEST_DIR / "notebooks" / "simple.ipynb",
        output_dir=get_temp_dir("test_MultiverseAnalysis_generate_universe_id"),
    )
    universe_id = mv.generate_universe_id({"x": "A", "y": "B"})
    assert universe_id == "3d4f2c7e4e2e4e2e4e2e4e2e4e2e4e2e"


def test_MultiverseAnalysis_examine_multiverse():
    output_dir = get_temp_dir("test_MultiverseAnalysis_examine_multiverse")
    mv = MultiverseAnalysis(
        {
            "x": ["A", "B"],
            "y": ["A", "B"],
        },
        notebook=TEST_DIR / "notebooks" / "simple.ipynb",
        output_dir=output_dir,
    )
    mv.examine_multiverse()
    assert count_files(output_dir, "runs/1/data/*.csv") == 4
    assert count_files(output_dir, "runs/1/notebooks/*.ipynb") == 4


def test_MultiverseAnalysis_visit_universe():
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


def test_Universe_get_execution_time():
    uv = Universe(settings={"dimensions": {"hello": "world"}})
    execution_time = uv.get_execution_time()
    assert execution_time >= 0


def test_Universe_save_data():
    output_dir = get_temp_dir("test_Universe_save_data")
    uv = Universe(
        settings={"dimensions": {"hello": "world"}, "output_dir": str(output_dir)}
    )
    data = pd.DataFrame({"test_value": [42]})
    uv.save_data(data)
    assert count_files(output_dir, "runs/0/data/*.csv") == 1


def test_Universe_compute_sub_universe_metrics():
    uv = Universe(settings={"dimensions": {"hello": "world"}})
    y_pred_prob = pd.Series([0.1, 0.4, 0.6, 0.9])
    y_test = pd.Series([0, 0, 1, 1])
    org_test = pd.DataFrame({"majmin": [0, 0, 1, 1]})
    sub_universe = {"cutoff": "raw_0.5", "eval_fairness_grouping": "majority-minority"}
    fairness_dict, metric_frame = uv.compute_sub_universe_metrics(
        sub_universe, y_pred_prob, y_test, org_test
    )
    assert "equalized_odds_difference" in fairness_dict
    assert "accuracy" in metric_frame.overall


def test_Universe_visit_sub_universe():
    uv = Universe(settings={"dimensions": {"hello": "world"}})
    y_pred_prob = pd.Series([0.1, 0.4, 0.6, 0.9])
    y_test = pd.Series([0, 0, 1, 1])
    org_test = pd.DataFrame({"majmin": [0, 0, 1, 1]})
    sub_universe = {"cutoff": "raw_0.5", "eval_fairness_grouping": "majority-minority"}
    filter_data = lambda sub_universe, org_test: pd.Series([True, True, True, True])  # noqa: E731
    final_output = uv.visit_sub_universe(
        sub_universe, y_pred_prob, y_test, org_test, filter_data
    )
    assert "fair_main_equalized_odds_difference" in final_output.columns
    assert "perf_ovrl_accuracy" in final_output.columns


def test_Universe_generate_sub_universes():
    uv = Universe(settings={"dimensions": {"x": ["A", "B"], "y": ["A", "B"]}})
    sub_universes = uv.generate_sub_universes()
    assert len(sub_universes) == 4
    assert sub_universes == [
        {"x": "A", "y": "A"},
        {"x": "A", "y": "B"},
        {"x": "B", "y": "A"},
        {"x": "B", "y": "B"},
    ]


def test_Universe_compute_final_metrics():
    uv = Universe(settings={"dimensions": {"hello": "world"}})
    y_pred_prob = pd.Series([0.1, 0.4, 0.6, 0.9])
    y_test = pd.Series([0, 0, 1, 1])
    org_test = pd.DataFrame({"majmin": [0, 0, 1, 1]})
    filter_data = lambda sub_universe, org_test: pd.Series([True, True, True, True])  # noqa: E731
    final_output = uv.compute_final_metrics(
        y_pred_prob, y_test, org_test, filter_data, save=False
    )
    assert not final_output.empty
    assert "fair_main_equalized_odds_difference" in final_output.columns
    assert "perf_ovrl_accuracy" in final_output.columns
