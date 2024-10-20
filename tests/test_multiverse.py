from multiversum import generate_multiverse_grid, MultiverseAnalysis

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
    config = TEST_DIR / "notebooks" / "simple.json"

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
