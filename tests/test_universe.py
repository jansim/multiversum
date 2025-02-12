import random

import numpy as np
import pandas as pd
import pytest

from multiversum.universe import Universe, add_dict_to_df


class TestUniverse:
    def test_set_seeds(self):
        # Initialize universe
        Universe({"dimensions": []}, set_seed=True)

        assert random.random() == 0.8444218515250481
        assert np.random.randint(10000) == 2732

    def test_expand_dicts_false(self):
        # Test when expand_dicts is False (default)
        settings = {
            "dimensions": {
                "model": {"type": "random_forest", "n_estimators": 100},
                "data": "full",
            }
        }
        universe = Universe(settings, expand_dicts=False)
        assert universe.dimensions == settings["dimensions"]

    def test_expand_dicts_true(self):
        # Test when expand_dicts is True
        settings = {
            "dimensions": {
                "model": {"type": "random_forest", "n_estimators": 100},
                "data": "full",
            }
        }
        universe = Universe(settings, expand_dicts=True)
        expected_dimensions = {
            "type": "random_forest",
            "n_estimators": 100,
            "data": "full",
        }
        assert universe.dimensions == expected_dimensions

    def test_expand_dicts_nested(self):
        # Test with nested dictionaries
        settings = {
            "dimensions": {
                "model": {"params": {"type": "random_forest", "n_estimators": 100}},
                "data": "full",
            }
        }
        universe = Universe(settings, expand_dicts=True)
        expected_dimensions = {
            "params": {"type": "random_forest", "n_estimators": 100},
            "data": "full",
        }
        assert universe.dimensions == expected_dimensions


class TestHelpers:
    def test_add_dict_to_df_empty_df_and_dict(self):
        df = pd.DataFrame()
        dictionary = {}
        result_df = add_dict_to_df(df, dictionary)
        assert result_df.equals(df)

    def test_add_dict_to_df_empty_df_and_scalars(self):
        df = pd.DataFrame()
        dictionary = {"A": 1, "B": 2, "C": 3}
        result_df = add_dict_to_df(df, dictionary)
        expected_df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        assert result_df.equals(expected_df)

    def test_add_dict_to_df_non_empty_df_and_dict(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        dictionary = {"B": [4, 5, 6], "C": [7, 8, 9]}
        result_df = add_dict_to_df(df, dictionary)
        expected_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        assert result_df.equals(expected_df)

    def test_add_dict_to_df_with_index(self):
        df = pd.DataFrame({"A": [1]}, index=["gibberish"])
        dictionary = {"B": [2], "C": 3.0}
        result_df = add_dict_to_df(df, dictionary)
        expected_df = pd.DataFrame(
            {"A": [1], "B": [2], "C": [3.0]}, index=["gibberish"]
        )
        assert result_df.equals(expected_df)

    def test_add_dict_to_df_with_prefix(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        dictionary = {"B": [4, 5, 6]}
        result_df = add_dict_to_df(df, dictionary, prefix="prefix_")
        expected_df = pd.DataFrame({"A": [1, 2, 3], "prefix_B": [4, 5, 6]})
        assert result_df.equals(expected_df)

    def test_add_dict_to_df_mismatched_lengths(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        dictionary = {"B": [4, 5]}
        with pytest.raises(ValueError):
            add_dict_to_df(df, dictionary)


if __name__ == "__main__":
    pytest.main()
