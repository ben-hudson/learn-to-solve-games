"""Variational-inequality family registry: name -> family class."""

from l2s_games.envs.base import VariationalInequality, VariationalInequalityFamily, bind

__all__ = [
    "VariationalInequality",
    "VariationalInequalityFamily",
    "bind",
]
