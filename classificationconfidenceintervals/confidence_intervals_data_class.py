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
    """Confidence Intervals Data Class.

    Attributes:
        tnorm_ci (tuple): Truncated normal confidence interval.
        poisson_ci (tuple): Poisson confidence interval.
        lrt_ci (tuple): Inverted binomial likelihood ratio test confidence interval.
        score_ci (tuple): Inverted binomial score test confidence interval.
        posterior_ci (tuple): Beta-Binomial posterior confidence interval.

    """

    tnorm_ci: Tuple[float, float]
    poisson_ci: Tuple[float, float]
    lrt_ci: Tuple[float, float]
    score_ci: Tuple[float, float]
    posterior_ci: Tuple[float, float]

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
            keys (list): Set of names of instance variables.

        """
        return cls.__dataclass_fields__.keys()  # type: ignore # pylint: disable=E1101

    def __repr__(self) -> str:
        """Repr default implementation."""
        return f"{self.__class__.__name__}({self.__dict__!r})"

    def __str__(self) -> str:
        """Str default implementation."""
        return repr(self)

    def __del__(self) -> None:
        """Deconstructor default implementation."""
        return None
