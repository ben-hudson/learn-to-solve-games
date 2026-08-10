"""GAMUT invocation: one ``java -jar gamut.jar`` call per game instance, parsed into payoff tensors.

GAMUT (http://gamut.stanford.edu) is the standard suite of normal-form game generators. It ships as a
single jar and emits one game per invocation; this module owns the subprocess call and the parser for its
default ``SimpleOutput`` format, and nothing else -- the VI family that consumes the payoffs lives in
``envs/gamut.py`` (keeping this module import-light so it never cycles through ``envs``).

Reproducibility: GAMUT takes a ``-random_seed``, and the repo rule is that no API grows a seed argument --
``lightning.seed_everything`` at the call site is the only entry point. So ``generate_payoffs`` draws each
instance's GAMUT seed from the **global torch RNG**, which makes a generation run reproducible under the
script's ``--seed`` exactly like the torch-sampled families.

Setup (nothing is bundled): install a Java runtime (``brew install --cask temurin``), download ``gamut.jar``
from the GAMUT website, and point ``$GAMUT_JAR`` at it.
"""

import os
import subprocess
import tempfile

import torch


def default_jar():
    """``gamut.jar`` location: ``$GAMUT_JAR``, else the working directory."""
    return os.environ.get("GAMUT_JAR", "gamut.jar")


def generate_payoffs(gamut_class, n_actions, jar_path, min_payoff=-1.0, max_payoff=1.0, options=()):
    """One GAMUT game: ``(A, B)`` payoff matrices, each ``[n_actions, n_actions]`` in real units.

    ``A[i, j]`` is player 1's payoff and ``B[i, j]`` player 2's when they play actions ``i`` and ``j``.
    ``options`` carries the class-specific flags (``-actions n n`` for RandomZeroSum; ``-players 2
    -actions n n [-r r]`` for RandomGame/CovariantGame; nothing for the fixed 2x2 classes), assembled once
    by the generation script and persisted on the dataset's base graph. Payoffs are normalized into
    ``[min_payoff, max_payoff]`` by GAMUT itself, which is what makes the range a family-level constant
    the conditioning can rely on.
    """
    seed = int(torch.randint(1, 2**31 - 1, ()))
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "game.out")
        command = [
            "java", "-jar", str(jar_path), "-g", gamut_class, *options,
            "-random_seed", str(seed),
            "-normalize", "-min_payoff", str(min_payoff), "-max_payoff", str(max_payoff),
            "-f", out,
        ]  # fmt: skip
        subprocess.run(command, check=True, capture_output=True, text=True)
        with open(out) as handle:
            text = handle.read()
    return parse_simple_output(text, n_actions)


def parse_simple_output(text, n_actions):
    """GAMUT's default ``SimpleOutput`` -> ``(A, B)`` payoff tensors, each ``[n_actions, n_actions]``.

    The grammar (see the GAMUT user guide): ``#``-prefixed comment lines, then one line per outcome,
    ``[a1 a2] : [ p1 p2 ]`` with **1-indexed** actions and one payoff per player.
    """
    payoffs = torch.zeros(2, n_actions, n_actions)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        actions, values = line.split(":")
        row, column = (int(action) - 1 for action in actions.strip(" \t[]").split())
        for player, payoff in enumerate(float(value) for value in values.strip(" \t[]").split()):
            payoffs[player, row, column] = payoff
    return payoffs[0], payoffs[1]
