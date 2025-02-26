import random
import warnings

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

    def test_seed_override(self):
        # Test that seed parameter overrides settings seed
        settings = {"dimensions": [], "seed": 123}

        # First check the seed from settings
        Universe(settings, set_seed=True)
        random_val1 = random.random()
        np_val1 = np.random.randint(10000)

        # Reset seeds for next test
        random.seed(0)
        np.random.seed(0)

        # Now check with seed override and capture warning
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered
            warnings.simplefilter("always")

            # Call the code that triggers the warning
            Universe(settings, set_seed=True, seed=456)

            # Verify warning was raised with correct message
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert (
                "Seed provided in constructor (456) is overriding seed from settings (123)"
                in str(w[0].message)
            )

        random_val2 = random.random()
        np_val2 = np.random.randint(10000)

        # Values should be different with different seeds
        assert random_val1 != random_val2
        assert np_val1 != np_val2

        # Verify the overridden seed is consistent
        random.seed(456)
        np.random.seed(456)
        assert random.random() == random_val2
        assert np.random.randint(10000) == np_val2

    def test_seed_dimension(self):
        # Test using a dimension as seed
        settings = {"dimensions": {"my_seed": 789, "other_dim": "value"}, "seed": 123}

        # First check the seed from settings
        Universe(settings, set_seed=True)
        random_val1 = random.random()

        # Reset seeds for next test
        random.seed(0)
        np.random.seed(0)

        # Now check with seed from dimension
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Provide dimension name as seed
            Universe(settings, set_seed=True, seed="my_seed")

            # Verify warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert (
                "Seed from dimension 'my_seed' (789) is overriding seed from settings (123)"
                in str(w[0].message)
            )

        random_val2 = random.random()

        # Values should be different with different seeds
        assert random_val1 != random_val2

        # Verify the dimension seed is consistent
        random.seed(789)
        assert random.random() == random_val2

    def test_seed_dimension_not_found(self):
        # Test using a non-existent dimension as seed
        settings = {"dimensions": {"existing_dim": "value"}, "seed": 123}

        # Check with non-existent dimension name
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Provide non-existent dimension name as seed
            Universe(settings, set_seed=True, seed="non_existent_dim")

            # Verify warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Dimension 'non_existent_dim' not found in dimensions" in str(
                w[0].message
            )

        # Should use settings seed instead
        random_val = random.random()

        # Reset and verify we get the same value with settings seed
        random.seed(123)
        assert random.random() == random_val

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
