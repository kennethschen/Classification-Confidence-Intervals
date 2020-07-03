#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""AUTHOR: KENNETH CHEN.

Module containing confidence intervals data class.

"""

from dataclasses import dataclass
from typing import Set, Tuple

from .read_only_properties_decorator import read_only_properties


@read_only_properties(
    "tnorm_ci", "poisson_ci", "lrt_ci", "score_ci", "posterior_ci", "simulated_ci"
)
@dataclass
class CIDataClass:
    """Confidence Intervals Data Class."""

    tnorm_ci: Tuple[float, float]
    poisson_ci: Tuple[float, float]
    lrt_ci: Tuple[float, float]
    score_ci: Tuple[float, float]
    posterior_ci: Tuple[float, float]
    simulated_ci: Tuple[float, float]

    def get(self, key: str) -> Tuple[float, float]:
        """Getter method.

        Args:
            key (str): String name of desired ci.

        Returns:
            ci (tuple): Confidence interval.

        """
        return self.__getattribute__(key)  # type: ignore

    @classmethod
    def keys(cls) -> Set[str]:
        """Return keys.

        Args:
            cls (Generic): Class

        Returns:
            keys (set): Set of names of instance variables.

        """
        return set(cls.__dataclass_fields__.keys())  # type: ignore # pylint: disable=E1101
