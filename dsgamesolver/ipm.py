"""Interior point method homotopy."""

# TODO: check user-provided initial_strategies and weights?
# TODO: adjust tracking parameters with "scale" of game
# TODO: think about Cython import
# TODO: add Numpy implementation?


from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike

from dsgamesolver.sgame import SGame, SGameHomotopy
from dsgamesolver.homcont import HomCont


# %% parent class for interior point method homotopy


class IPM(SGameHomotopy):
    """Interior point method homotopy: base class"""

    def __init__(self, game: SGame, initial_strategies: Union[str, ArrayLike] = "centroid",
                 weights: Optional[ArrayLike] = None) -> None:
        super().__init__(game)

        # TODO: adjust parameters with scale of payoff matrix:

        self.tracking_parameters['normal'] = {
            'x_tol': 1e-7,
            't_tol': 1e-7,
            'H_tol': 1e-7,
            'ds0': 0.1,
            'ds_infl': 1.2,
            'ds_defl': 0.5,
            'ds_min': 1e-9,
            'ds_max': 10,
            'corr_steps_max': 20,
            'corr_dist_max': 0.5,
            'corr_ratio_max': 0.9,
            'detJ_change_max': 0.6,
            'bifurc_angle_min': 175,
        }

        self.tracking_parameters['robust'] = {
            'x_tol': 1e-7,
            't_tol': 1e-7,
            'H_tol': 1e-8,
            'ds0': 0.1,
            'ds_infl': 1.1,
            'ds_defl': 0.5,
            'ds_min': 1e-9,
            'ds_max': 10,
            'corr_steps_max': 30,
            'corr_dist_max': 0.3,
            'corr_ratio_max': 0.7,
            'detJ_change_max': 0.7,
            'bifurc_angle_min': 175,
        }

        if initial_strategies == "centroid":
            self.sigma_0 = np.zeros((self.game.num_states, self.game.num_players, self.game.num_actions_max))
            for s in range(self.game.num_states):
                for p in range(self.game.num_players):
                    self.sigma_0[s, p, 0:self.game.nums_actions[s, p]] = 1 / self.game.nums_actions[s, p]
        elif initial_strategies == "random":
            self.sigma_0 = np.zeros((self.game.num_states, self.game.num_players, self.game.num_actions_max))
            for s in range(self.game.num_states):
                for p in range(self.game.num_players):
                    sigma = np.random.exponential(scale=1, size=self.game.nums_actions[s, p])
                    self.sigma_0[s, p, 0:self.game.nums_actions[s, p]] = sigma / sigma.sum()
        else:
            # TODO: document how initial strategies should be specified / should they be checked?
            self.sigma_0 = np.array(initial_strategies)

        self.sigma_0_flat = self.game.flatten_strategies(self.sigma_0)

        if weights is None:
            self.nu = np.ones((self.game.num_states, self.game.num_players, self.game.num_actions_max))
        else:
            # TODO: document how weights should be specified / should they be checked?
            self.nu = weights

    def initialize(self) -> None:
        self.y0 = self.find_y0()
        # Note: homotopy parameter t goes from 1 to 0
        self.solver = HomCont(self.H, self.y0, self.J, t_target=0.0,
                              parameters=self.tracking_parameters['robust'],
                              x_transformer=self.x_transformer, store_path=True)

    def find_y0(self) -> np.ndarray:
        V = np.ones((self.game.num_states, self.game.num_players))
        return self.sigma_V_t_to_y(self.sigma_0, V, 1.0)

    def x_transformer(self, y: np.ndarray) -> np.ndarray:
        x = y.copy()
        z, t = x[0:self.game.num_actions_total], x[-1]
        x[0:self.game.num_actions_total] = 0.25 * (z + np.sqrt(z**2 + 4*t*np.sqrt(self.sigma_0_flat)))**2
        return x

    def sigma_V_t_to_y(self, sigma: np.ndarray, V: np.ndarray, t: float) -> np.ndarray:
        sigma_flat = self.game.flatten_strategies(sigma)
        z_flat = (sigma_flat - t * np.sqrt(self.sigma_0_flat)) / np.sqrt(sigma_flat)
        V_flat = self.game.flatten_values(V)
        return np.concatenate([z_flat, V_flat, [t]])

    def y_to_sigma_V_t(self, y: np.ndarray, zeros: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
        sigma_V_t_flat = self.x_transformer(y)
        sigma = self.game.unflatten_strategies(sigma_V_t_flat[0:self.game.num_actions_total], zeros=zeros)
        V = self.game.unflatten_values(sigma_V_t_flat[self.game.num_actions_total:-1])
        t = sigma_V_t_flat[-1]
        return sigma, V, t


# %% Cython implementation of IPM


class IPM_ct(IPM):
    """Interior point method homotopy: Cython implementation"""

    def __init__(self, game: SGame, initial_strategies: Union[str, ArrayLike] = "centroid",
                 weights: Optional[ArrayLike] = None) -> None:
        super().__init__(game, initial_strategies, weights)

        # only import Cython module on class instantiation
        try:
            import pyximport
            pyximport.install(build_dir='./dsgamesolver/__build__/', build_in_temp=False, language_level=3,
                              setup_args={'include_dirs': [np.get_include()]})
            import dsgamesolver.ipm_ct as ipm_ct

        except ImportError:
            raise ImportError("Cython implementation of IPM homotopy could not be imported. ",
                              "Make sure your system has the relevant C compilers installed. ",
                              "For Windows, check https://wiki.python.org/moin/WindowsCompilers ",
                              "to find the right Microsoft Visual C++ compiler for your Python version. ",
                              "Standalone compilers are sufficient, there is no need to install Visual Studio. ",
                              "For Linux, make sure the Python package gxx_linux-64 is installed in your environment.")

        self.ipm_ct = ipm_ct

    def H(self, y: np.ndarray) -> np.ndarray:
        return self.ipm_ct.H(y, self.game.payoffs, self.game.transitions, self.sigma_0, self.nu,
                             self.game.num_states, self.game.num_players, self.game.nums_actions,
                             self.game.num_actions_max, self.game.num_actions_total)

    def J(self, y: np.ndarray) -> np.ndarray:
        return self.ipm_ct.J(y, self.game.payoffs, self.game.transitions, self.sigma_0, self.nu,
                             self.game.num_states, self.game.num_players, self.game.nums_actions,
                             self.game.num_actions_max, self.game.num_actions_total)


# %% testing


if __name__ == '__main__':

    from tests.random_game import create_random_game
    game = SGame(*create_random_game())

    # cython
    ipm_ct = IPM_ct(game)
    ipm_ct.initialize()
    ipm_ct.solver.solve()

    y0 = ipm_ct.y0
    """
    %timeit ipm_ct.H(y0)
    %timeit ipm_ct.J(y0)
    """
