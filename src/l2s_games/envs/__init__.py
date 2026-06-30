"""Variational-inequality family registry: name -> family class."""

from l2s_games.envs.base import VariationalInequality, VariationalInequalityFamily, bind
from l2s_games.envs.matrix import MatrixGame
from l2s_games.envs.rps import RockPaperScissors2D, SymmetricZeroSumGame
from l2s_games.envs.toy import RotationalFieldGame
from l2s_games.envs.traffic import MarkovTrafficEquilibrium

GAMES = {
    "toy": RotationalFieldGame,
    "rps": RockPaperScissors2D,
    "symmetric": SymmetricZeroSumGame,
    "traffic": MarkovTrafficEquilibrium,
}


def make_game(name, **kwargs):
    return GAMES[name](**kwargs)


__all__ = [
    "VariationalInequality",
    "VariationalInequalityFamily",
    "bind",
    "MatrixGame",
    "RotationalFieldGame",
    "SymmetricZeroSumGame",
    "MarkovTrafficEquilibrium",
    "GAMES",
    "make_game",
]
