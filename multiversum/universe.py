"""This module contains helpers for running the individual universes within
a multiverse analysis.
"""

from pathlib import Path
import random
import time
import pandas as pd
import json
import numpy as np
import warnings
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

from .helpers import add_universe_info_to_df
from .multiverse import generate_multiverse_grid


def predict_w_threshold(probabilities: np.array, threshold: float) -> np.array:
    """
    Create binary predictions from probabilities using a custom threshold.

    Args:
        probabilities: A numpy array containing the probabilities for each class.
        threshold: The threshold to use for the predictions.

    Returns:
        A numpy array containing the binary predictions.
    """
    # Expect 2 classes
    assert probabilities.shape[1] == 2

    # Check whether probability for second column (class 1) is gr. or equal to threshold
    return probabilities[:, 1] >= threshold


def add_dict_to_df(df: pd.DataFrame, dictionary: dict, prefix="") -> pd.DataFrame:
    """
    Add values from a dictionary as columns to a dataframe.

    Args:
        df: The dataframe to which the columns should be added.
        dictionary: The dictionary containing the values to be added.
        prefix: A prefix to be added to the column names. (optional)

    Returns:
        The dataframe with the added columns.
    """
    for key, value in dictionary.items():
        df[prefix + key] = value
    return df


def flatten_dict(d: dict, parent_key="", sep="_") -> dict:
    """
    Flatten a nested dictionary.

    Args:
        d: The dictionary to be flattened.
        parent_key: The parent key to be used for the flattened keys. (optional)
        sep: The separator to be used for the flattened keys. (optional)

    Returns:
        The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # Convert nested Pandas-Series to dict
        if isinstance(v, pd.Series):
            v = dict(v)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def list_wrap(value: Any) -> List[Any]:
    """
    Wrap a value in a List if it is not already a list.

    Args:
        value: Any sort of value.

    Returns:
        List[Any]: A list containing the value. If the value is already a list,
            the value is returned unchanged.
    """
    if isinstance(value, list):
        return value
    else:
        return [value]


class Universe:
    """
    A class to help with running the analysis of a single universe contained
    within a multiverse analysis.

    Attributes:
        run_no: The run number of the multiverse analysis.
        universe_id: The id of the universe.
        universe: The universe settings.
        output_dir: The directory to which the output should be written.
        metrics: A dictionary containing the metrics to be computed.
        fairness_metrics: A dictionary containing the fairness metrics to be
            computed.
        ts_start: The timestamp of the start of the analysis.
        ts_end: The timestamp of the end of the analysis.
    """

    metrics = None
    fairness_metrics = None
    ts_start = None
    ts_end = None

    def __init__(
        self,
        settings: Union[str, Dict[str, Any]],
        metrics: Optional[Dict[str, Callable]] = None,
        fairness_metrics: Optional[Dict[str, Callable]] = None,
        set_seed: bool = True,
    ) -> None:
        """
        Initialize the Universe class.

        The arguments should be passed in from the larger multiverse analysis.

        Args:
            settings: The settings for the universe analysis. This can usually
                just be passed along from the multiverse analysis. You only need
                to specify this yourself when developing / trying out an
                analysis. Possible keys in the dictionary are:
                - dimensions: The specified universe dimensions. This is the
                    only required information.
                - run_no: The run number of the multiverse analysis.
                - seed: The seed to use for analyses.
                - universe_id: The id of the universe.
                output_dir: The directory to which the output should be written.
            metrics: A dictionary containing the metrics to be computed.
                Pass an empty dictionary to not compute any.
            fairness_metrics: A dictionary containing the fairness metrics to be
                computed. (These are cumputed with awareness of groups.)
                Pass an empty dictionary to not compute any.
            set_seed: Whether to use the seed provided in the settings.
                Defaults to True. Please note, that this only sets the seed in
                the Python random module and numpy.
        """
        self.ts_start = time.time()

        # Extract settings
        parsed_settings = (
            json.loads(settings) if isinstance(settings, str) else settings
        )
        self.run_no = parsed_settings["run_no"] if "run_no" in parsed_settings else 0
        self.universe_id = (
            parsed_settings["universe_id"]
            if "universe_id" in parsed_settings
            else "no-universe-id-provided"
        )
        self.dimensions = parsed_settings["dimensions"]
        self.seed = parsed_settings["seed"] if "seed" in parsed_settings else 0
        self.output_dir = (
            Path(parsed_settings["output_dir"])
            if "output_dir" in parsed_settings
            else Path("./output")
        )

        self.metrics = metrics
        self.fairness_metrics = fairness_metrics

        if self.dimensions is None:
            warnings.warn("No dimensions specified for universe analysis.")

        if set_seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def get_execution_time(self) -> float:
        """
        Gets the execution time of the universe analysis.

        Returns:
            float: The execution time in seconds.
        """
        if self.ts_end is None:
            print("Stopping execution_time clock.")
            self.ts_end = time.time()
        return self.ts_end - self.ts_start

    def _add_universe_info(
        self, data: pd.DataFrame, overwrite_dimensions: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Add general universe / run info to the dataframe.

        Args:
            data: The dataframe to which the info should be added.
            overwrite_dimensions: A dictionary containing dimensions to overwrite. (optional)

        Returns:
            The dataframe with the added info.
        """
        return add_universe_info_to_df(
            data=data,
            universe_id=self.universe_id,
            run_no=self.run_no,
            dimensions=self.dimensions
            if overwrite_dimensions is None
            else overwrite_dimensions,
            execution_time=self.get_execution_time(),
        )

    def save_data(self, data: pd.DataFrame, add_info: bool = True) -> None:
        """
        Save the data to the appropriate file for this Universe.

        Args:
            data: The dataframe to be saved.
            add_info: Whether to add universe info to the dataframe. (optional)

        Returns:
            None
        """
        # Add universe data to the dataframe
        if add_info:
            data = self._add_universe_info(data=data)

        # Path management
        target_dir = self.output_dir / "runs" / str(self.run_no) / "data"
        # Make sure the directory exists
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"d_{str(self.run_no)}_{self.universe_id}.csv"
        filepath = target_dir / filename
        if filepath.exists():
            warnings.warn(f"File {filepath} already exists. Overwriting it.")
        # Write the file
        data.to_csv(filepath, index=False)

    def compute_sub_universe_metrics(
        self,
        sub_universe: Dict,
        y_pred_prob: pd.Series,
        y_test: pd.Series,
        org_test: pd.DataFrame,
    ) -> Tuple[dict, dict]:
        """
        Computes a set of metrics for a given sub-universe.

        Args:
            sub_universe: A dictionary containing the parameters for the
                sub-universe.
            y_pred_prob: A pandas series containing the predicted
                probabilities.
            y_test: A pandas series containing the true labels.
            org_test: A pandas dataframe containing the test data, including
                variables that were not used as features.

        Returns:
            A tuple containing two dics: explicit fairness metrics and
                performance metrics split by fairness groups.
        """
        # Determine cutoff for predictions
        cutoff_type, cutoff_value = sub_universe["cutoff"].split("_")
        cutoff_value = float(cutoff_value)

        if cutoff_type == "raw":
            threshold = cutoff_value
        elif cutoff_type == "quantile":
            probabilities_true = y_pred_prob[:, 1]
            threshold = np.quantile(probabilities_true, cutoff_value)

        fairness_grouping = sub_universe["eval_fairness_grouping"]
        if fairness_grouping == "majority-minority":
            fairness_group_column = "majmin"
        elif fairness_grouping == "race-all":
            fairness_group_column = "RAC1P"

        y_pred = predict_w_threshold(y_pred_prob, threshold)

        try:
            from fairlearn.metrics import MetricFrame
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                balanced_accuracy_score,
                f1_score,
            )
            from fairlearn.metrics import (
                false_positive_rate,
                false_negative_rate,
                selection_rate,
                count,
            )
            from fairlearn.metrics import (
                equalized_odds_difference,
                equalized_odds_ratio,
                demographic_parity_difference,
                demographic_parity_ratio,
            )

            metrics = (
                {
                    "accuracy": accuracy_score,
                    "balanced accuracy": balanced_accuracy_score,
                    "f1": f1_score,
                    "precision": precision_score,
                    "false positive rate": false_positive_rate,
                    "false negative rate": false_negative_rate,
                    "selection rate": selection_rate,
                    "count": count,
                }
                if self.metrics is None
                else self.metrics
            )

            fairness_metrics = (
                {
                    "equalized_odds_difference": equalized_odds_difference,
                    "equalized_odds_ratio": equalized_odds_ratio,
                    "demographic_parity_difference": demographic_parity_difference,
                    "demographic_parity_ratio": demographic_parity_ratio,
                }
                if self.fairness_metrics is None
                else self.fairness_metrics
            )

            # Compute fairness metrics
            fairness_dict = {
                name: metric(
                    y_true=y_test,
                    y_pred=y_pred,
                    sensitive_features=org_test[fairness_group_column],
                )
                for name, metric in fairness_metrics.items()
            }

            # Compute "normal" metrics (but split by fairness column)
            metric_frame = MetricFrame(
                metrics=metrics,
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=org_test[fairness_group_column],
            )

            return (fairness_dict, metric_frame)
        except ImportError:
            raise ImportError(
                "Packages fairlearn and scikit-learn are required for computing metrics."
            )

    def visit_sub_universe(
        self,
        sub_universe: Dict[str, Any],
        y_pred_prob: pd.Series,
        y_test: pd.Series,
        org_test: pd.Series,
        filter_data: Callable,
    ) -> pd.DataFrame:
        """
        Visit a sub-universe and compute the metrics for it.

        Sub-universes correspond to theoretically distinct universes of
        decisions, which can be computed without re-fitting a model. The
        distinction has only been made to improve performance by not having to
        compute these universes from scratch.

        Args:
            sub_universe: A dictionary containing the parameters for the
                sub-universe.
            y_pred_prob: A pandas series containing the predicted
                probabilities.
            y_test: A pandas series containing the true labels.
            org_test: A pandas dataframe containing the test data, including
                variables that were not used as features.
            filter_data: A function that filters data for each sub-universe.
                The function is called for each sub-universe with its
                respective settings and expected to return a pandas Series
                of booleans.

        Returns:
            A pandas dataframe containing the metrics for the sub-universe.
        """
        final_output = self._add_universe_info(
            data=pd.DataFrame(index=[self.universe_id]),
            overwrite_dimensions=sub_universe,
        )

        data_mask = filter_data(sub_universe=sub_universe, org_test=org_test)
        final_output["test_size_n"] = data_mask.sum()
        final_output["test_size_frac"] = data_mask.sum() / len(data_mask)

        # Compute metrics for majority-minority split
        fairness_dict, metric_frame = self.compute_sub_universe_metrics(
            sub_universe,
            y_pred_prob[data_mask],
            y_test[data_mask],
            org_test[data_mask],
        )

        # Add main fairness metrics to final_output
        final_output = add_dict_to_df(final_output, fairness_dict, prefix="fair_main_")
        final_output = add_dict_to_df(
            final_output, dict(metric_frame.overall), prefix="perf_ovrl_"
        )

        # Add group metrics to final output
        final_output = add_dict_to_df(
            final_output, flatten_dict(metric_frame.by_group), prefix="perf_grp_"
        )

        return final_output

    def generate_sub_universes(self) -> List[dict]:
        """
        Generate the sub-universes for the given universe settings.

        Returns:
            A list of dictionaries containing the sub-universes.
        """
        # Wrap all non-lists in the universe to make them work with generate_multiverse_grid
        universe_all_lists = {k: list_wrap(v) for k, v in self.dimensions.items()}

        # Within-Universe variation
        return generate_multiverse_grid(universe_all_lists)

    def compute_final_metrics(
        self,
        y_pred_prob: pd.Series,
        y_test: pd.Series,
        org_test: pd.Series,
        filter_data: Callable,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Generate the final output for the given universe settings.

        Args:
            y_pred_prob: A pandas series containing the predicted
                probabilities.
            y_test: A pandas series containing the true labels.
            org_test: A pandas dataframe containing the test data, including
                variables that were not used as features.
            filter_data: A function that filters data for each sub-universe.
                The function is called for each sub-universe with its
                respective settings and expected to return a pandas Series
                of booleans.
            save: Whether to save the output to a file. (optional)

        Returns:
            A pandas dataframe containing the final output.
        """
        # Within-Universe variation
        sub_universes = self.generate_sub_universes()

        final_outputs = list()
        for sub_universe in sub_universes:
            final_outputs.append(
                self.visit_sub_universe(
                    sub_universe=sub_universe,
                    y_pred_prob=y_pred_prob,
                    y_test=y_test,
                    org_test=org_test,
                    filter_data=filter_data,
                ).reset_index(drop=True)
            )
        final_output = pd.concat(final_outputs)

        # Write the final output file
        if save:
            self.save_data(final_output, add_info=False)

        return final_output
