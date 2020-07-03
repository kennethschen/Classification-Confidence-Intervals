#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""AUTHOR: KENNETH CHEN.

Module containing class for constructing positive rate, precision, negative predictive value, and
recall confidence intervals based on samples of a population.

"""

from collections import Counter
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats as st
import seaborn as sns

from .check_inputs import check_get_cis_params, check_init_params
from .confidence_intervals_data_class import CIDataClass
from .plot_config import COLORS, CONFIG
from .read_only_properties_decorator import read_only_properties


@read_only_properties(
    "N",
    "N_flagged",
    "alpha",
    "exact_precision",
    "n",
    "n_flagged",
    "n_positives",
    "n_predicted_negatives",
    "n_true_negatives",
    "n_true_positives",
    "negative_predictive_dist_poisson",
    "negative_predictive_dist_posterior",
    "negative_predictive_dist_tnorm",
    "positives_dist_poisson",
    "positives_dist_posterior",
    "positives_dist_tnorm",
    "precision_dist_poisson",
    "precision_dist_posterior",
    "precision_dist_tnorm",
    "sample_labels",
    "sample_predictions",
)
class ClassificationConfidenceIntervals:
    """Class for determining confidence intervals for classification metrics."""

    def __init__(
        self,
        sample_labels: List[Union[bool, int]],
        sample_predictions: List[Union[bool, int]],
        population_size: int,
        population_flagged_count: int,
        confidence_level: float,
        exact_precision: Union[None, float] = None,
    ) -> None:
        """Initialization.

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
        check_init_params(
            sample_labels,
            sample_predictions,
            population_size,
            population_flagged_count,
            confidence_level,
            exact_precision,
        )

        self.get_sample_counts(sample_labels, sample_predictions)

        self.N = population_size
        self.N_flagged = population_flagged_count
        self.alpha = confidence_level
        self.exact_precision = exact_precision

        self.get_sample_distributions()

    def get_sample_counts(
        self, sample_labels: List[Union[bool, int]], sample_predictions: List[Union[bool, int]]
    ) -> None:
        """Get sample counts and successes for positives, precision, and negative predictive value.

        Args:
            sample_labels (list): Binary labels of datapoints in sample, with labels as boolean or
                binary in [0,1] or in [-1,1].
            sample_predictions (list): Binary labels of datapoints in sample flagged as positives
                by algorithm, with labels as boolean or binary in [0,1] or in [-1,1].

        """
        sample_labels = np.array(sample_labels)
        sample_predictions = np.array(sample_predictions)

        # pylint: disable=W0201
        self.n = len(sample_labels)
        self.n_pos = sum(sample_labels)

        self.n_flagged = sum(sample_predictions)
        self.n_true_pos = sum([1 == x == y for x, y in zip(sample_labels, sample_predictions)])

        self.n_pred_neg = self.n - self.n_flagged
        self.n_true_neg = self.n_pred_neg - (self.n_pos - self.n_true_pos)
        # pylint: disable=W0201

    def get_sample_distributions(self) -> None:
        """Get parametric distributions for positives, precision, and negative predictive value."""
        # pylint: disable=W0201
        # positive rate distributions based on binomial distribution approximations
        pos_phat = self.n_pos / self.n
        pos_sigmahat = np.sqrt(pos_phat * (1 - pos_phat) / self.n)
        self.pos_dist_tnorm = st.truncnorm(
            a=(pos_phat - 1) / pos_sigmahat,
            b=pos_phat / pos_sigmahat,
            loc=pos_phat,
            scale=pos_sigmahat,
        )
        self.pos_dist_poisson = st.poisson(self.n_pos)

        # precision distributions based on binomial distribution approximations
        ppv_phat = self.n_true_pos / self.n_flagged
        ppv_sigmahat = np.sqrt(ppv_phat * (1 - ppv_phat) / self.n_flagged)
        self.precision_dist_tnorm = st.truncnorm(
            a=(ppv_phat - 1) / ppv_sigmahat,
            b=ppv_phat / ppv_sigmahat,
            loc=ppv_phat,
            scale=ppv_sigmahat,
        )
        self.ppv_dist_poisson = st.poisson(self.n_true_pos)

        # negative predictive value distributions based on binomial distribution approximations
        npv_phat = self.n_true_neg / self.n_pred_neg
        npv_sigmahat = np.sqrt(npv_phat * (1 - npv_phat) / self.n_pred_neg)
        self.npv_dist_tnorm = st.truncnorm(
            a=(npv_phat - 1) / npv_sigmahat,
            b=npv_phat / npv_sigmahat,
            loc=npv_phat,
            scale=npv_sigmahat,
        )
        self.npv_dist_poisson = st.poisson(self.n_true_neg)

        # posterior distributions with flat priors
        self.pos_dist_posterior = st.beta(a=0.5 + self.n_pos, b=0.5 + self.n - self.n_pos)
        self.ppv_dist_posterior = st.beta(
            a=0.5 + self.n_true_pos, b=0.5 + self.n_flagged - self.n_true_pos
        )
        self.npv_dist_posterior = st.beta(
            a=0.5 + self.n_true_neg, b=0.5 + self.n_pred_neg - self.n_true_neg,
        )
        # pylint: disable=W0201

    def get_cis(
        self, n_iters: int = 1000000, plot_filename: str = ""
    ) -> Tuple[
        CIDataClass, CIDataClass, CIDataClass, CIDataClass,
    ]:
        """Get confidence intervals.

        Args:
            n_iters (int): Number of iterations to simulate posterior models.
            plot_filename (str): If not empty, save plots using filename as relative path.

        Returns:
            pos_rate_cis (CIDataClass): Confidence intervals for pos rate based on multiple methods.
            ppv_cis (CIDataClass): Confidence intervals for precision based on multiple methods.
            npv_cis (CIDataClass): Confidence intervals for NPV based on multiple methods.
            recall_cis (CIDataClass): Confidence intervals for recall based on multiple methods.

        """
        # check get_cis params
        check_get_cis_params(n_iters, plot_filename)

        # MC simulation
        self.run_simulations(n_iters)

        # get cis
        pos_rate_cis = self.get_pos_rate_cis()

        if not self.exact_precision:
            ppv_cis = self.get_ppv_cis()
        else:
            exact_ppv_ci: Tuple[float, float] = tuple([self.exact_precision] * 2)  # type: ignore
            ppv_cis = CIDataClass(
                exact_ppv_ci, exact_ppv_ci, exact_ppv_ci, exact_ppv_ci, exact_ppv_ci, exact_ppv_ci
            )

        npv_cis = self.get_npv_cis()

        recall_cis = self.get_recall_cis(ppv_cis, npv_cis)

        # get plots
        if plot_filename:
            self.get_plots(plot_filename, pos_rate_cis, ppv_cis, npv_cis, recall_cis)

        return pos_rate_cis, ppv_cis, npv_cis, recall_cis

    def run_simulations(self, n_iters: int) -> None:
        """Run MC simulations and create simulated recall distribution.

        Args:
            n_iters (int): Number of iterations to simulate.

        """
        # pylint: disable=W0201
        # simulated distributions based on posteriors
        self.pos_hats = self.pos_dist_posterior.rvs(size=n_iters, random_state=10)
        self.ppv_hats = self.ppv_dist_posterior.rvs(size=n_iters, random_state=10)
        self.npv_hats = self.npv_dist_posterior.rvs(size=n_iters, random_state=10)

        # simulated distribution for recall
        self.tpr_hats = (self.N_flagged * self.ppv_hats) / (
            self.N_flagged * self.ppv_hats + (self.N - self.N_flagged) * (1 - self.npv_hats)
        )
        # pylint: disable=W0201

    def get_pos_rate_cis(self) -> CIDataClass:
        """Get confidence intervals for probability of being a positive.

        Returns:
            pos_rate_cis (CIDataClass): Confidence intervals for pos rate based on multiple methods.

        """
        return CIDataClass(
            self.pos_dist_tnorm.interval(self.alpha),
            self.binomial_poisson_approx_ci(self.n, self.pos_dist_poisson, self.alpha),
            self.binomial_likelihood_ratio_test_ci(self.n, self.n_pos, self.alpha),
            self.binomial_score_test_ci(self.n, self.n_pos, self.alpha),
            self.pos_dist_posterior.interval(self.alpha),
            self.binomial_simulated_ci(self.pos_hats, self.alpha),
        )

    def get_ppv_cis(self) -> CIDataClass:
        """Get confidence intervals for precision.

        Returns:
            ppv_cis (CIDataClass): Confidence intervals for precision based on multiple methods.

        """
        return CIDataClass(
            self.precision_dist_tnorm.interval(self.alpha),
            self.binomial_poisson_approx_ci(self.n_flagged, self.ppv_dist_poisson, self.alpha),
            self.binomial_likelihood_ratio_test_ci(self.n_flagged, self.n_true_pos, self.alpha),
            self.binomial_score_test_ci(self.n_flagged, self.n_true_pos, self.alpha),
            self.ppv_dist_posterior.interval(self.alpha),
            self.binomial_simulated_ci(self.ppv_hats, self.alpha),
        )

    def get_npv_cis(self) -> CIDataClass:
        """Get confidence intervals for negative predictive value.

        Returns:
            npv_cis (CIDataClass): Confidence intervals for NPV based on multiple methods.

        """
        return CIDataClass(
            self.npv_dist_tnorm.interval(self.alpha),
            self.binomial_poisson_approx_ci(self.n_pred_neg, self.npv_dist_poisson, self.alpha),
            self.binomial_likelihood_ratio_test_ci(self.n_pred_neg, self.n_true_neg, self.alpha),
            self.binomial_score_test_ci(self.n_pred_neg, self.n_true_neg, self.alpha),
            self.npv_dist_posterior.interval(self.alpha),
            self.binomial_simulated_ci(self.npv_hats, self.alpha),
        )

    def get_recall_cis(self, ppv_cis: CIDataClass, npv_cis: CIDataClass,) -> CIDataClass:
        """Get confidence intervals for recall.

        Args:
            ppv_cis (CIDataClass): Confidence intervals for precision based on multiple methods.
            npv_cis (CIDataClass): Confidence intervals for NPV based on multiple methods.

        Returns:
            recall_cis (CIDataClass): Confidence intervals for recall based on multiple methods.

        """
        recall_cis: Dict[str, Tuple[float, float]] = {}

        # distributional cis
        for key in CIDataClass.keys():
            a_min = min(ppv_cis.get(key)) * self.N_flagged
            a_max = max(ppv_cis.get(key)) * self.N_flagged
            b_min = min(1 - np.array(npv_cis.get(key))) * (self.N - self.N_flagged)
            b_max = max(1 - np.array(npv_cis.get(key))) * (self.N - self.N_flagged)
            ci = tuple([a_min / (a_min + b_max), a_max / (a_max + b_min)])
            recall_cis[key] = ci  # type: ignore

        # simulated cis
        recall_cis["simulated_ci"] = self.binomial_simulated_ci(self.tpr_hats, self.alpha)

        return CIDataClass(
            recall_cis["tnorm_ci"],
            recall_cis["poisson_ci"],
            recall_cis["lrt_ci"],
            recall_cis["score_ci"],
            recall_cis["posterior_ci"],
            recall_cis["simulated_ci"],
        )

    def get_plots(
        self,
        plot_filename: str,
        pos_rate_cis: CIDataClass,
        ppv_cis: CIDataClass,
        npv_cis: CIDataClass,
        recall_cis: CIDataClass,
    ) -> None:
        """Save plots using filename as relative path.

        Args:
            plot_filename (str): Save plots using filename as relative path.
            pos_rate_cis (CIDataClass): Confidence intervals for pos rate based on multiple methods.
            ppv_cis (CIDataClass): Confidence intervals for precision based on multiple methods.
            npv_cis (CIDataClass): Confidence intervals for NPV based on multiple methods.
            recall_cis (CIDataClass): Confidence intervals for recall based on multiple methods.

        """

        # set up plot
        sns.set(style="darkgrid")
        fig, axs = plt.subplots(CONFIG["n_subplots"], sharex=True, figsize=(10, 10))
        fig.suptitle(f"Metrics for {100*self.alpha}% Confidence Level")
        plt.xticks(np.arange(0, 1.05, 0.05), rotation=30)
        plt.xlim((0, 1))

        # ci whitespaces
        n_whitespaces = max([len(key) for key in CIDataClass.keys()])

        # plot distributions
        lq, uq, n_bins = CONFIG["lq"], CONFIG["uq"], CONFIG["n_bins"]
        for label, color, distributions in zip(
            ("Truncated-Normal", "Poisson", "Beta-Posterior"),
            (COLORS["tnorm_ci"], COLORS["poisson_ci"], COLORS["posterior_ci"]),
            (
                (self.pos_dist_tnorm, self.precision_dist_tnorm, self.npv_dist_tnorm,),
                (self.pos_dist_poisson, self.ppv_dist_poisson, self.npv_dist_poisson,),
                (self.pos_dist_posterior, self.ppv_dist_posterior, self.npv_dist_posterior,),
            ),
        ):
            for i, dist in enumerate(distributions):
                if label == "Poisson":
                    xs = np.arange(dist.ppf(lq), dist.ppf(uq))
                    ys = dist.pmf(xs)
                    xs /= (self.n, self.n_flagged, self.n - self.n_flagged)[i]
                else:
                    xs = np.linspace(dist.ppf(lq), dist.ppf(uq), n_bins,)
                    ys = dist.pdf(xs)
                    ys /= sum(ys)
                axs[i].plot(xs, ys, color=color, alpha=CONFIG["transparency"], label=label)

        # simulated distributions
        for i, hats in enumerate((self.pos_hats, self.ppv_hats, self.npv_hats, self.tpr_hats)):
            counts = Counter(np.round(hats, CONFIG["rounding_digits"]))  # type: ignore
            proportions = sorted(counts.keys())
            densities = np.array([counts[p] for p in proportions]) / sum(counts.values())
            axs[i].plot(
                proportions,
                densities,
                color=COLORS["simulated_ci"],
                alpha=CONFIG["transparency"],
                label="MC Simulations",
            )

        # confidence intervals
        for key in CIDataClass.keys():
            for i, ci in enumerate(
                (pos_rate_cis.get(key), ppv_cis.get(key), npv_cis.get(key), recall_cis.get(key),)
            ):
                axs[i].vlines(
                    x=ci,
                    ymin=0,
                    ymax=max(axs[i].get_ylim()),
                    linestyles="--",
                    colors=COLORS[key],
                    alpha=CONFIG["transparency"],
                    label=f"{key}: {' ' * (n_whitespaces - len(key))}" f"{np.round(ci, 4)}",
                )

        # legend and title
        titles = ["Positive Rate", "Precision", "NPV", "Recall"]
        for i in range(int(CONFIG["n_subplots"])):
            axs[i].legend(fontsize=CONFIG["fontsize"])
            axs[i].set_title(titles[i])
            axs[i].set(xlabel="Probability", ylabel="Density")
            axs[i].label_outer()
        plt.subplots_adjust(hspace=CONFIG["hspace"])
        plt.savefig(plot_filename)

    @staticmethod
    def binomial_poisson_approx_ci(n: int, dist: st.poisson, alpha: float) -> Tuple[float, float]:
        """Invoke poisson approximation to binomial to get confidence intervals.

        Args:
            n (int): Total sample size from which poisson distribution was fitted.
            dist (st.poisson): The fitted poisson distribution.
            alpha (float): Confidence level, equal to area desired under curve.

        Returns:
            ci (list): Confidence interval based on poisson approximation.

        """
        ci: Tuple[float, float] = tuple(np.array(dist.interval(alpha)) / n)  # type: ignore
        return ci

    @staticmethod
    def binomial_simulated_ci(hats: List[float], alpha: float) -> Tuple[float, float]:
        """Draw quantiles from simulations to get confidence intervals.

        Args:
            hats (list): List of simulated estimates for parameter.
            alpha (float): Confidence level, equal to area desired under curve.

        Returns:
            ci (list): Confidence interval based on drawing quantiles from simulations.

        """
        area = 100 * (1 - alpha) / 2
        ci: Tuple[float, float] = tuple(np.percentile(hats, [area, 100 - area]))  # type: ignore
        return ci

    @staticmethod
    def binomial_likelihood_ratio_test_ci(n: int, y: int, alpha: float) -> Tuple[float, float]:
        """Invert binomial LRT test with 1 DF to get confidence intervals.

        Args:
            n (int): Sample size.
            y (int): Number of successes in the sample.
            alpha (float): Confidence level, equal to area desired under curve.

        Returns:
            ci (list): Confidence interval based on inverting a binomial LRT test.

        """
        phat = y / n
        threshold = st.chi2.ppf(alpha, df=1)
        valid_p0s = []

        for p0 in np.arange(0.0001, 1, 0.0001):
            ll_ratio_test_stat = 2 * (
                y * np.log(phat / p0) + (n - y) * np.log((1 - phat) / (1 - p0))
            )
            if ll_ratio_test_stat <= threshold:
                valid_p0s.append(p0)

        ci: Tuple[float, float] = (valid_p0s[0], valid_p0s[-1])
        return ci

    @staticmethod
    def binomial_score_test_ci(n: int, y: int, alpha: float) -> Tuple[float, float]:
        """Invert binomial score test with 1 DF to get confidence intervals.

        Args:
            n (int): Sample size.
            y (int): Number of successes in the sample.
            alpha (float): Confidence level, equal to area desired under curve.

        Returns:
            ci (list): Confidence interval based on inverting a binomial score test.

        """
        phat = y / n
        threshold = st.chi2.ppf(alpha, df=1)
        valid_p0s = []

        for p0 in np.arange(0.0001, 1, 0.0001):
            score_test_stat = ((phat - p0) / np.sqrt(p0 * (1 - p0) / n)) ** 2
            if score_test_stat <= threshold:
                valid_p0s.append(p0)

        ci: Tuple[float, float] = (valid_p0s[0], valid_p0s[-1])
        return ci

    def __repr__(self) -> str:
        """Repr default implementation."""
        return f"{self.__class__.__name__}({self.__dict__!r})"

    def __str__(self) -> str:
        """Str default implementation."""
        return repr(self)

    def __del__(self) -> None:
        """Deconstructor default implementation."""
        return None
