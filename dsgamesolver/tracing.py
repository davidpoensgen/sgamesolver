"""(Logarithmic stochastic) tracing homotopy."""

# TODO: check user-provided priors and etas?
# TODO: maybe write custom optimization for find_y0 to avoid scipy import

# TODO: play with einsum_path
# TODO: adjust tracking parameters with "scale" of game
# TODO: think about Cython import

import warnings
from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from dsgamesolver.sgame import SGame, SGameHomotopy
from dsgamesolver.homcont import HomCont

ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# %% parent class for Tracing homotopy


class Tracing(SGameHomotopy):
    """Tracing homotopy: base class"""

    def __init__(self, game: SGame, priors: Union[str, ArrayLike] = "centroid", weights: Optional[ArrayLike] = None):
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
            'corr_ratio_max': 0.5,
            'detJ_change_max': 0.6,  # TODO: change format. was: 1.5
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
            'corr_ratio_max': 0.3,
            'detJ_change_max': 0.7,  # TODO: change format. was: 1.3
            'bifurc_angle_min': 175,
        }

        # rho
        if priors == "centroid":
            self.priors = np.zeros((self.game.num_states, self.game.num_players, self.game.num_actions_max))
            for s in range(self.game.num_states):
                for p in range(self.game.num_players):
                    self.priors[s, p, 0:self.game.nums_actions[s, p]] = 1 / self.game.nums_actions[s, p]
        elif priors == "random":
            self.priors = np.zeros((self.game.num_states, self.game.num_players, self.game.num_actions_max))
            for s in range(self.game.num_states):
                for p in range(self.game.num_players):
                    sigma = np.random.exponential(scale=1, size=self.game.nums_actions[s, p])
                    self.priors[s, p, 0:self.game.nums_actions[s, p]] = sigma / sigma.sum()
        else:
            # TODO: document how priors should be specified / should they be checked?
            self.priors = np.array(priors)

        # nu
        if weights is None:
            self.weights = np.ones((self.game.num_states, self.game.num_players, self.game.num_actions_max))
        else:
            # TODO: document how weights should be specified / should they be checked?
            self.weights = weights

        # eta_0
        self.scale = 1.0

        # prepare payoffs and transition given other players follow prior
        num_s, num_p, num_a_max = self.game.num_states, self.game.num_players, self.game.num_actions_max
        rho_p_list = [self.priors[:, p, :] for p in range(num_p)]
        einsum_eqs_u_rho = ['s' + ABC[0:num_p] + ',s' + ',s'.join(ABC[p_] for p_ in range(num_p) if p_ != p)
                            + '->s' + ABC[p] for p in range(num_p)]
        einsum_eqs_phi_rho = ['s' + ABC[0:num_p] + 't,s' + ',s'.join(ABC[p_] for p_ in range(num_p) if p_ != p)
                              + '->s' + ABC[p] + 't' for p in range(num_p)]

        if num_p > 1:
            self.u_rho = np.empty((num_s, num_p, num_a_max))
            self.phi_rho = np.empty((num_s, num_p, num_a_max, num_s))
            for p in range(num_p):
                self.u_rho[:, p] = np.einsum(einsum_eqs_u_rho[p],
                                             self.game.payoffs[:, p], *(rho_p_list[:p]+rho_p_list[(p+1):]))
                self.phi_rho[:, p] = np.einsum(einsum_eqs_phi_rho[p],
                                               self.game.transitions[:, p], *(rho_p_list[:p]+rho_p_list[(p+1):]))
        else:
            self.u_rho = self.game.payoffs
            self.phi_rho = self.game.transitions

    def initialize(self) -> None:
        self.y0 = self.find_y0()
        self.solver = HomCont(self.H, self.y0, self.J, t_target=1.0,
                              parameters=self.tracking_parameters['normal'],
                              x_transformer=self.x_transformer, store_path=True)

    def find_y0(self, tol: Union[float, int] = 1e-12, max_iter: int = 10000) -> np.ndarray:
        """Value function iteration."""

        num_s, num_p, nums_a = self.game.num_states, self.game.num_players, self.game.nums_actions

        sigma_old = self.game.centroid_strategy()
        V_old = self.game.get_values(sigma_old)
        sigma = sigma_old.copy()
        V = V_old.copy()

        for k in range(max_iter):
            c = self.u_rho + np.einsum('spaS,Sp->spa', self.phi_rho, V_old)

            # computation for all agents separately
            for s in range(num_s):
                for p in range(num_p):

                    # solve system of multi-linear equations
                    # - by first combining all equations to one big equation in sigma1
                    # - and then solving the equation by Brent's method
                    c_max_idx = np.argmax(c[s, p, 0:nums_a[s, p]])

                    def f(sigma1):
                        return - 1 + ((self.weights[s, p, 0:nums_a[s, p]] / self.weights[s, p, c_max_idx])
                                      / ((c[s, p, c_max_idx] - c[s, p, 0:nums_a[s, p]])
                                         / (self.scale * self.weights[s, p, c_max_idx]) + 1/sigma1)).sum()

                    sigma1 = brentq(f, 1e-24, 1)

                    sigma[s, p, 0:nums_a[s, p]] = ((self.weights[s, p, 0:nums_a[s, p]] / self.weights[s, p, c_max_idx])
                                                   / ((c[s, p, c_max_idx] - c[s, p, 0:nums_a[s, p]])
                                                      / (self.scale * self.weights[s, p, c_max_idx]) + 1/sigma1))
                    V[s, p] = (c[s, p, c_max_idx] + ((self.scale * self.weights[s, p, c_max_idx]) / sigma1)
                               + self.scale * ((self.weights[s, p, 0:nums_a[s, p]]
                                                * (np.log(sigma[s, p, 0:nums_a[s, p]]) - 1)).sum()))

            if np.max(np.abs(V - V_old)) < tol and np.allclose(sigma, sigma_old, rtol=0, atol=tol, equal_nan=True):
                break
            else:
                V_old = V.copy()
                sigma_old = sigma.copy()

        if k >= max_iter-1:
            warnings.warn('Value function iteration has not converged.')

        return self.sigma_V_t_to_y(sigma, V, 0.0)

    def x_transformer(self, y: np.ndarray) -> np.ndarray:
        """Reverts logarithmization of strategies in vector y:
        Transformed values are needed to check whether sigmas have converged.
        """
        x = y.copy()
        x[0:self.game.num_actions_total] = np.exp(x[0:self.game.num_actions_total])
        return x


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

        # only import Cython module on class instantiation
        try:
            import pyximport
            pyximport.install(build_dir='./dsgamesolver/__build__/', build_in_temp=False, language_level=3,
                              setup_args={'include_dirs': [np.get_include()]})
            import dsgamesolver.tracing_ct as tracing_ct

        except ImportError:
            raise ImportError("Cython implementation of Tracing homotopy could not be imported. ",
                              "Make sure your system has the relevant C compilers installed. ",
                              "For Windows, check https://wiki.python.org/moin/WindowsCompilers ",
                              "to find the right Microsoft Visual C++ compiler for your Python version. ",
                              "Standalone compilers are sufficient, there is no need to install Visual Studio. ",
                              "For Linux, make sure the Python package gxx_linux-64 is installed in your environment.")

        self.tracing_ct = tracing_ct

    def H(self, y: np.ndarray) -> np.ndarray:
        return self.tracing_ct.H(y, self.game.payoffs, self.game.transitions,
                                 self.priors, self.weights, self.scale, self.u_rho, self.phi_rho,
                                 self.game.num_states, self.game.num_players, self.game.nums_actions,
                                 self.game.num_actions_max, self.game.num_actions_total)

    def J(self, y: np.ndarray) -> np.ndarray:
        return self.tracing_ct.J(y, self.game.payoffs, self.game.transitions,
                                 self.priors, self.weights, self.scale, self.u_rho, self.phi_rho,
                                 self.game.num_states, self.game.num_players, self.game.nums_actions,
                                 self.game.num_actions_max, self.game.num_actions_total)


# %% testing


if __name__ == '__main__':

    from tests.random_game import create_random_game
    game = SGame(*create_random_game())

    # numpy
    tracing_np = Tracing_np(game)
    y0 = tracing_np.find_y0()
    """
    %timeit tracing_np.H(y0)
    %timeit tracing_np.J(y0)
    """

    # cython
    tracing_ct = Tracing_ct(game)
    print(tracing_ct.H(tracing_ct.find_y0()))
    """
    %timeit tracing_ct.H(y0)
    %timeit tracing_ct.J(y0)
    """
