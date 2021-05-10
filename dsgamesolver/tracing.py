# from typing import Tuple, Union

import numpy as np

from dsgamesolver.sgame import SGame, SGameHomotopy
# from dsgamesolver.homcont import HomCont

ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# %% parent class for log tracing homotopy


class Tracing(SGameHomotopy):
    def __init__(self, game: SGame, priors="centroid", etas=None, nu=1.0):
        super().__init__(game)

        if priors == "random":
            priors = np.empty(self.game.num_actions_total, dtype=np.float64)
            idx = 0
            for s in range(self.game.num_states):
                for p in range(self.game.num_players):
                    sigma = np.random.exponential(
                        scale=1, size=self.game.nums_actions[s, p]
                    )
                    sigma = sigma / sigma.sum()
                    etas[idx : idx + self.game.nums_actions[s, p]] = sigma
                    idx += self.game.nums_actions[s, p]
            self.priors = priors
        elif priors == "centroid":
            priors = np.empty(self.game.num_actions_total, dtype=np.float64)
            idx = 0
            for s in range(self.game.num_states):
                for p in range(self.game.num_players):
                    etas[idx : idx + self.game.nums_actions[s, p]] = (
                        1 / self.game.nums_actions[s, p]
                    )
                    idx += self.game.nums_actions[s, p]
            self.priors = priors
        else:
            # TODO: how are priors specified / should they be checked?
            self.priors = priors

        self.nu = nu

        if etas is None:
            # TODO: format of etas?
            self.etas = np.ones((self.game.num_states, self.game.num_players, self.game.num_actions_max),
                                dtype=np.float64)
        else:
            # TODO: how are etas specified / should they be checked?
            self.etas = etas

    def find_y0(self):
        # TODO
        y0 = None
        self.y0 = y0


# %% Numpy implementation of Tracing


class Tracing_np(Tracing):
    """Tracing homotopy: Numpy implementation"""

    def __init__(self, game: SGame) -> None:
        """prepares the following:
            - H_mask, J_mask
            - T_H, T_J
            - einsum_eqs
        """
        super().__init__(game)

    def H(self, y: np.ndarray) -> np.ndarray:
        # TODO
        return super().H(y)

    def J(self, y: np.ndarray) -> np.ndarray:
        # TODO
        return super().J(y)


# %% Cython implementation of Tracing


class Tracing_ct(Tracing):
    """Tracing homotopy: Cython implementation"""

    def __init__(self, game: SGame) -> None:
        super().__init__(game)

    def H(self, y: np.ndarray) -> np.ndarray:
        # TODO
        return super().H(y)

    def J(self, y: np.ndarray) -> np.ndarray:
        # TODO
        return super().J(y)
