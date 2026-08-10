"""Variational-inequality family registry: name -> family class."""

from l2s_games.envs.base import VariationalInequality, VariationalInequalityFamily, bind
from l2s_games.envs.gamut import GamutGame
from l2s_games.envs.matrix import MatrixGame
from l2s_games.envs.rps import RockPaperScissors2D, SymmetricZeroSumGame
from l2s_games.envs.toy import RotationalFieldGame
from l2s_games.envs.traffic import MarkovTrafficEquilibrium
from l2s_games.envs.pume_traffic import PUMEMarkovTrafficEquilibrium
from l2s_games.envs.asym_pume_traffic import AsymmetricPUMEMarkovTrafficEquilibrium

GAMES = {
    "toy": RotationalFieldGame,
    "rps": RockPaperScissors2D,
    "symmetric": SymmetricZeroSumGame,
    "gamut": GamutGame,
    "traffic": MarkovTrafficEquilibrium,
    "pume_traffic": PUMEMarkovTrafficEquilibrium,
    "asym_pume_traffic": AsymmetricPUMEMarkovTrafficEquilibrium,
}


def make_game(name, **kwargs):
    return GAMES[name](**kwargs)


__all__ = [
    "VariationalInequality",
    "VariationalInequalityFamily",
    "bind",
    "GamutGame",
    "MatrixGame",
    "RotationalFieldGame",
    "SymmetricZeroSumGame",
    "MarkovTrafficEquilibrium",
    "PUMEMarkovTrafficEquilibrium",
    "AsymmetricPUMEMarkovTrafficEquilibrium",
    "GAMES",
    "make_game",
]
