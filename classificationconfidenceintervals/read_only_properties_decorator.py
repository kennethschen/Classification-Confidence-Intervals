#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""AUTHOR: KENNETH CHEN.

Module containing decorator around class with overwritten __setattr__ to make decorator names
instance variables read-only.

"""

from __future__ import annotations

from typing import Any


def read_only_properties(*attrs: str) -> "CustomClass":  # type: ignore # pylint: disable=E0602 # noqa: F821, E501
    """Read only properties decorator.

    Args:
        *attrs (*args): Comma-separated arguments to make read-only in class.

    Returns:
        CustomClass (CustomClass): Class with overwritten __setattr__.

    """

    def class_rebuilder(cls) -> "CustomClass":  # type: ignore # noqa: F821
        """Rebuild class.

        Args:
            cls (class): Class.

        Returns:
            CustomClass (CustomClass): Class with overwritten __setattr__.

        """

        class CustomClass(cls):  # type: ignore
            """Overwritten class."""

            def __setattr__(self, name: str, value: Any) -> None:
                """Overwrite __setattr__ to force decorator names to be read-only.

                Args:
                    name (str): Name of the instance variable.
                    value (any): Value of the instance variable.

                """

                if name in attrs and name in self.__dict__:
                    raise AttributeError("Can't modify {}".format(name))

                super().__setattr__(name, value)

        return CustomClass

    return class_rebuilder
