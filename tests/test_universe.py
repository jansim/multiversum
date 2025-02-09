import pandas as pd
import pytest
import random
import numpy as np
from multiversum.universe import Universe, add_dict_to_df


class TestUniverse:
    def test_set_seeds(self):
        # Initialize universe
        Universe({"dimensions": []}, set_seed=True)

        assert random.random() == 0.8444218515250481
        assert np.random.randint(10000) == 2732


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
