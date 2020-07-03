#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""AUTHOR: KENNETH CHEN.

Methods for ensuring inputs to methods of ClassificationConfidenceIntervals are valid.

"""

from collections import Counter
from typing import List, Union


def check_init_params(  # noqa: C901
    sample_labels: List[Union[bool, int]],
    sample_predictions: List[Union[bool, int]],
    population_size: int,
    population_flagged_count: int,
    confidence_level: float,
    exact_precision: Union[None, float],
) -> None:
    """Ensure inputs to __init__ method of ClassificationConfidenceIntervals are valid.

    Args:
        sample_labels (list): Binary labels of datapoints in sample, with labels as boolean or
            binary in [0,1] or in [-1,1].
        sample_predictions (list): Binary labels of datapoints in sample flagged as positives
            by algorithm, with labels as boolean or binary in [0,1] or in [-1,1].
        population_size (int): Size of population.
        population_flagged_count, (int): Number of datapoints in population flagged as positives
            by algorithm.
        confidence_level (float): Confidence level, equal to area desired under curve.
        exact_precision (float): If provided, the actual population precision.

    """
    # sample labels
    if set(sample_labels) not in [set([True, False]), set([0, 1]), set([-1, 1])]:
        raise ValueError("sample_labels must be boolean or be binary in [0,1] or [-1,1].")

    # sample predictions
    if set(sample_predictions) not in [set([True, False]), set([0, 1]), set([-1, 1])]:
        raise ValueError("sample_predictions must be boolean or be binary in [0,1] or [-1,1].")

    # sample labels and sample predictions
    if len(sample_labels) != len(sample_predictions):
        raise ValueError(
            f"sample_labels of length {len(sample_labels)} and sample_predictions of "
            f"length {len(sample_predictions)} must be of the same length."
        )

    # check confusion matrix
    confusion_dict = Counter()  # type: ignore
    for truth, pred in zip(sample_labels, sample_predictions):
        confusion_dict[(truth, pred)] += 1
    if len(confusion_dict.values()) < 4:
        raise Exception("Your sample is degenerate with PPV or NPV as 0 or 1.")

    # population size
    if not isinstance(population_size, int) or not population_size > 0:
        raise Exception(f"population_size {population_size} must be a positive integer.")

    # population_flagged_count
    if not isinstance(population_flagged_count, int) or not population_flagged_count > 0:
        raise Exception(
            f"population_flagged_count {population_flagged_count} must be a non-negative integer."
        )

    # sample length and population size
    if len(sample_labels) > population_size:
        raise ValueError(
            f"population_size {population_size} must be a positive integer greater than "
            f"or equal to your sample size of {len(sample_labels)}."
        )

    # population size and population_flagged_count
    if not 0 <= population_flagged_count <= population_size:
        raise ValueError(
            f"population_flagged_count {population_flagged_count} must be a non-negative integer "
            f"less than your population_size of {population_size}."
        )

    # confidence_level
    if not isinstance(confidence_level, float) or not 0 < confidence_level < 1:
        raise Exception(
            f"confidence_level {confidence_level} must be of type float, "
            f"between 0 and 1 exclusive."
        )

    # exact_precision
    if exact_precision is not None:
        if not isinstance(exact_precision, float) or not 0 <= exact_precision <= 1:
            raise Exception(
                f"exact_precision {exact_precision} must be of type float, "
                f"between 0 and 1 inclusive."
            )


def check_get_cis_params(n_iters: int, plot_filename: str) -> None:
    """Ensure inputs to get_cis method of ClassificationConfidenceIntervals are valid.

    Args:
        n_iters (int): Number of iterations to simulate posterior models.
        plot_filename (str): If not empty, save plots using filename as relative path.

    """

    # n_iters
    if not isinstance(n_iters, int):
        raise TypeError(f"n_iters {n_iters} must be of type integer.")
    if not n_iters > 0:
        raise ValueError(f"n_iters {n_iters} must be greater than 0.")

    # plots
    if not isinstance(plot_filename, str):
        raise TypeError(f"plot_filename {plot_filename} must be of type str.")
