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
from .plot_config import COLORS, CONFIG, MODEL_TO_LABEL
from .read_only_properties_decorator import read_only_properties


@read_only_properties(
    "N",
    "N_flagged",
    "alpha",
    "exact_precision",
    "n",
    "n_flagged",
    "n_pos",
    "n_pred_neg",
    "n_true_neg",
    "n_true_pos",
    "npv_dist_binom",
    "npv_dist_poisson",
    "npv_dist_posterior",
    "npv_dist_tnorm",
    "pos_dist_binom",
    "pos_dist_poisson",
    "pos_dist_posterior",
    "pos_dist_tnorm",
    "ppv_dist_binom",
    "ppv_dist_poisson",
    "ppv_dist_posterior",
    "ppv_dist_tnorm",
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
            population_flagged_count (int): Number of datapoints in population flagged as positives
                by algorithm.
            confidence_level (float): Confidence level, equal to area desired under curve.
            exact_precision (float): If provided, the actual population precision.

        Attributes:
            N (int): Population size.
            N_flagged (int): Population flagged count.
            alpha (float): Confidence level.
            exact_precision (float): If provided, the actual population precision.
            n (int): Sample size.
            n_flagged (int): Sample flagged count.
            n_pos (int): Sample positives count.
            n_pred_neg (int): Sample predicted negatives count.
            n_true_neg (int): Sample true negatives count.
            n_true_pos (int): Sample true positives count.
            npv_dist_binom (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted binomial distribution for negative predictive value.
            npv_dist_poisson (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted poisson distribution for negative predictive value.
            npv_dist_posterior (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted posterior distribution for negative predictive value.
            npv_dist_tnorm (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted truncated normal distribution for negative predictive value.
            pos_dist_binom (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted binomial distribution for positive rate.
            pos_dist_poisson (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted poisson distribution for positive rate.
            pos_dist_posterior (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted posterior distribution for positive rate.
            pos_dist_tnorm (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted truncated normal distribution for positive rate.
            ppv_dist_binom (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted binomial distribution for precision.
            ppv_dist_poisson (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted poisson distribution for precision.
            ppv_dist_posterior (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted posterior distribution for precision.
            ppv_dist_tnorm (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted truncated normal distribution for precision.

        """
        check_init_params(
            sample_labels,
            sample_predictions,
            population_size,
            population_flagged_count,
            confidence_level,
            exact_precision,
        )

        self.N = population_size
        self.N_flagged = population_flagged_count
        self.alpha = confidence_level
        self.exact_precision = exact_precision

        sample_labels = np.array(sample_labels)
        sample_predictions = np.array(sample_predictions)

        self.n = len(sample_labels)
        self.n_pos = sum(sample_labels)

        self.n_flagged = sum(sample_predictions)
        self.n_true_pos = sum([1 == x == y for x, y in zip(sample_labels, sample_predictions)])

        self.n_pred_neg = self.n - self.n_flagged
        self.n_true_neg = self.n_pred_neg - (self.n_pos - self.n_true_pos)

        (
            self.pos_dist_tnorm,
            self.pos_dist_poisson,
            self.pos_dist_binom,
            self.pos_dist_posterior,
        ) = self.get_parametric_distributions(self.n_pos, self.n)
        (
            self.ppv_dist_tnorm,
            self.ppv_dist_poisson,
            self.ppv_dist_binom,
            self.ppv_dist_posterior,
        ) = self.get_parametric_distributions(self.n_true_pos, self.n_flagged)
        (
            self.npv_dist_tnorm,
            self.npv_dist_poisson,
            self.npv_dist_binom,
            self.npv_dist_posterior,
        ) = self.get_parametric_distributions(self.n_true_neg, self.n_pred_neg)

    @staticmethod
    def get_parametric_distributions(
        successes: int, size: int
    ) -> Tuple[
        st._distn_infrastructure.rv_frozen,  # pylint: disable=W0212
        st._distn_infrastructure.rv_frozen,  # pylint: disable=W0212
        st._distn_infrastructure.rv_frozen,  # pylint: disable=W0212
        st._distn_infrastructure.rv_frozen,  # pylint: disable=W0212
    ]:
        """Get parametric distributions for given metric.

        Args:
            successes (int): Number of successes in relevant sample.
            size (int): Size of relevant sample.

        Returns:
            metric_dist_tnorm (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted truncated normal distribution for metric.
            metric_dist_poisson (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted poisson distribution for metric.
            metric_dist_posterior (scipy.stats._distn_infrastructure.rv_frozen):
                Fitted posterior distribution for metirc.

        """

        phat = successes / size
        sigmahat = np.sqrt(phat * (1 - phat) / size)
        metric_dist_tnorm = st.truncnorm(
            a=(phat - 1) / sigmahat,
            b=phat / sigmahat,
            loc=phat,
            scale=sigmahat,
        )

        metric_dist_poisson = st.poisson(successes)

        metric_dist_binom = st.binom(n=size, p=phat)

        metric_dist_posterior = st.beta(a=0.5 + successes, b=0.5 + size - successes)

        return metric_dist_tnorm, metric_dist_poisson, metric_dist_binom, metric_dist_posterior

    def get_cis(
        self, n_iters: int = 1000000, plot_filename: str = ""
    ) -> Tuple[CIDataClass, CIDataClass, CIDataClass, CIDataClass]:
        """Get confidence intervals.

        Args:
            n_iters (int): Number of iterations to simulate posterior models.
            plot_filename (str): If not empty, save plots using filename as relative path.

        Returns:
            pos_rate_cis (CIDataClass): Confidence intervals for pos rate based on multiple models.
            ppv_cis (CIDataClass): Confidence intervals for precision based on multiple models.
            npv_cis (CIDataClass): Confidence intervals for NPV based on multiple models.
            recall_cis (CIDataClass): Confidence intervals for recall based on multiple models.

        """
        # check get_cis params
        check_get_cis_params(n_iters, plot_filename)

        # get cis
        pos_rate_cis = self.get_pos_rate_cis()

        if not self.exact_precision:
            ppv_cis = self.get_ppv_cis()
        else:
            exact_ppv_ci: Tuple[float, float] = tuple([self.exact_precision] * 2)  # type: ignore
            ppv_cis = CIDataClass(
                exact_ppv_ci, exact_ppv_ci, exact_ppv_ci, exact_ppv_ci, exact_ppv_ci
            )

        npv_cis = self.get_npv_cis()

        recall_cis = self.get_recall_cis(n_iters)

        # get plots
        if plot_filename:
            self.get_plots(plot_filename, pos_rate_cis, ppv_cis, npv_cis, recall_cis)

        return pos_rate_cis, ppv_cis, npv_cis, recall_cis

    def get_pos_rate_cis(self) -> CIDataClass:
        """Get confidence intervals for positive rate.

        Returns:
            pos_rate_cis (CIDataClass): Confidence intervals for pos rate based on multiple models.

        """
        return CIDataClass(
            self.pos_dist_tnorm.interval(self.alpha),
            self.binomial_poisson_approx_ci(self.n, self.pos_dist_poisson, self.alpha),
            self.binomial_likelihood_ratio_test_ci(self.n, self.n_pos, self.alpha),
            self.binomial_score_test_ci(self.n, self.n_pos, self.alpha),
            self.pos_dist_posterior.interval(self.alpha),
        )

    def get_ppv_cis(self) -> CIDataClass:
        """Get confidence intervals for precision.

        Returns:
            ppv_cis (CIDataClass): Confidence intervals for precision based on multiple models.

        """
        return CIDataClass(
            self.ppv_dist_tnorm.interval(self.alpha),
            self.binomial_poisson_approx_ci(self.n_flagged, self.ppv_dist_poisson, self.alpha),
            self.binomial_likelihood_ratio_test_ci(self.n_flagged, self.n_true_pos, self.alpha),
            self.binomial_score_test_ci(self.n_flagged, self.n_true_pos, self.alpha),
            self.ppv_dist_posterior.interval(self.alpha),
        )

    def get_npv_cis(self) -> CIDataClass:
        """Get confidence intervals for negative predictive value.

        Returns:
            npv_cis (CIDataClass): Confidence intervals for NPV based on multiple models.

        """
        return CIDataClass(
            self.npv_dist_tnorm.interval(self.alpha),
            self.binomial_poisson_approx_ci(self.n_pred_neg, self.npv_dist_poisson, self.alpha),
            self.binomial_likelihood_ratio_test_ci(self.n_pred_neg, self.n_true_neg, self.alpha),
            self.binomial_score_test_ci(self.n_pred_neg, self.n_true_neg, self.alpha),
            self.npv_dist_posterior.interval(self.alpha),
        )

    def get_recall_cis(self, n_iters: int) -> CIDataClass:
        """Get confidence intervals for recall using Monte Carlo simulations.

        Args:
            n_iters (int): Number of iterations to simulate posterior models.

        Returns:
            recall_cis (CIDataClass): Confidence intervals for recall based on multiple models.

        """
        # pylint: disable=W0201
        self.tpr_hats_tnorm = self.run_simulations(
            self.ppv_dist_tnorm, self.npv_dist_tnorm, n_iters, False
        )
        self.tpr_hats_poisson = self.run_simulations(
            self.ppv_dist_poisson, self.npv_dist_poisson, n_iters, True
        )
        self.tpr_hats_lrt = self.tpr_hats_score = self.run_simulations(
            self.ppv_dist_binom, self.npv_dist_binom, n_iters, True
        )
        self.tpr_hats_posterior = self.run_simulations(
            self.ppv_dist_posterior, self.npv_dist_posterior, n_iters, False
        )
        # pylint: disable=W0201

        return CIDataClass(
            self.simulated_ci(self.tpr_hats_tnorm, self.alpha),
            self.simulated_ci(self.tpr_hats_poisson, self.alpha),
            self.simulated_ci(self.tpr_hats_lrt, self.alpha),
            self.simulated_ci(self.tpr_hats_score, self.alpha),
            self.simulated_ci(self.tpr_hats_posterior, self.alpha),
        )

    def run_simulations(
        self,
        ppv_model: st._distn_infrastructure.rv_frozen,  # pylint: disable=W0212
        npv_model: st._distn_infrastructure.rv_frozen,  # pylint: disable=W0212
        n_iters: int,
        counts_correction: bool,
    ) -> np.array:
        """Run MC simulation and create simulated recall distribution.

        Args:
            ppv_model (st._distn_infrastructure.rv_frozen): Fitted precision model.
            npv_model (st._distn_infrastructure.rv_frozen): Fitted NPV model.
            n_iters (int): Number of iterations to simulate.
            counts_correction (bool): If distribution models counts instead of proportion parameter, set to True.

        Returns:
            tpr_hats (list): List of simulated recall values.

        """
        # simulated distributions based on posteriors
        ppv_hats = ppv_model.rvs(size=n_iters, random_state=10)
        npv_hats = npv_model.rvs(size=n_iters, random_state=10)

        # simulated distribution for recall
        if counts_correction:
            ppv_hats = ppv_hats / self.n_flagged
            npv_hats = npv_hats / self.n_pred_neg
        return (self.N_flagged * ppv_hats) / (
            self.N_flagged * ppv_hats + (self.N - self.N_flagged) * (1 - npv_hats)
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
            pos_rate_cis (CIDataClass): Confidence intervals for pos rate based on multiple models.
            ppv_cis (CIDataClass): Confidence intervals for precision based on multiple models.
            npv_cis (CIDataClass): Confidence intervals for NPV based on multiple models.
            recall_cis (CIDataClass): Confidence intervals for recall based on multiple models.

        """

        # set up plot
        sns.set(style="darkgrid")
        fig, axs = plt.subplots(CONFIG["n_subplots"], sharex=True, figsize=(10, 10))
        fig.suptitle(f"Metrics for {100*self.alpha}% Confidence Level")
        plt.xticks(np.arange(0, 1.05, 0.05), rotation=30)
        plt.xlim((0, 1))
        recall_idx = 3

        # ci whitespaces
        n_whitespaces = max([len(label) for label in MODEL_TO_LABEL.values()])

        # plot exact distributions
        lq, uq, n_bins = CONFIG["lq"], CONFIG["uq"], CONFIG["n_bins"]
        for model in ["tnorm", "poisson", "posterior"]:
            distributions = [
                getattr(self, f"pos_dist_{model}"),
                getattr(self, f"ppv_dist_{model}"),
                getattr(self, f"npv_dist_{model}"),
            ]
            sizes = (self.n, self.n_flagged, self.n - self.n_flagged)
            for i, dist in enumerate(distributions):
                if model in ["binom", "poisson"]:
                    xs = np.arange(dist.ppf(lq), dist.ppf(uq))
                    ys = dist.pmf(xs)
                    xs /= sizes[i]
                else:
                    xs = np.linspace(
                        dist.ppf(lq),
                        dist.ppf(uq),
                        n_bins,
                    )
                    ys = dist.pdf(xs)
                    ys /= sum(ys)
                axs[i].plot(xs, ys, color=COLORS[model], alpha=CONFIG["transparency"])

        # plot simulated recall distributions
        for model in ["tnorm", "poisson", "posterior"]:
            counts: Dict[int, int] = Counter(
                np.round(getattr(self, f"tpr_hats_{model}"), CONFIG["rounding_digits"])
            )
            proportions = sorted(counts.keys())
            densities = np.array([counts[p] for p in proportions]) / sum(counts.values())
            axs[recall_idx].plot(
                proportions,
                densities,
                color=COLORS[model],
                alpha=CONFIG["transparency"],
            )

        # plot confidence intervals
        for key in CIDataClass.keys():
            model = key.split("_ci")[0]
            for i, ci in enumerate(
                (
                    pos_rate_cis.get(key),
                    ppv_cis.get(key),
                    npv_cis.get(key),
                    recall_cis.get(key),
                )
            ):
                label = MODEL_TO_LABEL[model]
                axs[i].vlines(
                    x=ci,
                    ymin=0,
                    ymax=max(axs[i].get_ylim()),
                    linestyles="--",
                    colors=COLORS[model],
                    alpha=CONFIG["transparency"],
                    label=f"{label}: {' ' * (n_whitespaces - len(label))}{np.round(ci, 4)}",
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
        """Invoke poisson approximation to binomial to get confidence intervals for proportion parameter.

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

    @staticmethod
    def simulated_ci(hats: List[float], alpha: float) -> Tuple[float, float]:
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

    def __repr__(self) -> str:
        """Repr default implementation."""
        return f"{self.__class__.__name__}({self.__dict__!r})"

    def __str__(self) -> str:
        """Str default implementation."""
        return repr(self)

    def __del__(self) -> None:
        """Deconstructor default implementation."""
        return None
