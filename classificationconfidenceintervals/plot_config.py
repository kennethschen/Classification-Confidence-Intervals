#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""AUTHOR: KENNETH CHEN.

Default plot parameters for ClassificationConfidenceIntervals.get_plots method.

"""

lq = 0.000001
CONFIG = {
    "n_subplots": 4,
    "fontsize": 8,
    "transparency": 0.7,
    "lq": lq,
    "uq": 1 - lq,
    "n_bins": 100,
    "rounding_digits": 2,
    "hspace": 0.25,
}

COLORS = {
    "tnorm_ci": "red",
    "poisson_ci": "orange",
    "lrt_ci": "gold",
    "score_ci": "purple",
    "posterior_ci": "blue",
    "simulated_ci": "green",
}
