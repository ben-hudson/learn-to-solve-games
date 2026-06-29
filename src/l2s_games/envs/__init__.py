"""Game registry: name -> game class."""

from l2s_games.envs.base import Game, ParamSpec
from l2s_games.envs.matrix import MatrixGame
from l2s_games.envs.rps import RockPaperScissors2D, SymmetricZeroSumGame
from l2s_games.envs.toy import RotationalFieldGame

GAMES = {
    "toy": RotationalFieldGame,
    "rps": RockPaperScissors2D,
    "symmetric": SymmetricZeroSumGame,
}


def make_game(name, **kwargs):
    return GAMES[name](**kwargs)


__all__ = [
    "Game",
    "ParamSpec",
    "MatrixGame",
    "SymmetricZeroSumGame",
    "RotationalFieldGame",
    "GAMES",
    "make_game",
]
