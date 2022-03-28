"""Interior point method homotopy."""

# TODO: @Steffen, please re-write to remove x-transformer; write a distance function instead if desired

# TODO: check user-provided initial_strategies and weights?
# TODO: adjust tracking parameters with "scale" of game
# TODO: think about Cython import
# TODO: add Numpy implementation?


from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike

from sgamesolver.sgame import SGame, SGameHomotopy
from sgamesolver.homcont import HomCont


ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


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
            'ds_max': 1000,
            'corr_steps_max': 20,
            'corr_dist_max': 0.5,
            'corr_ratio_max': 0.9,
            'detJ_change_max': 1.5,
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
            'detJ_change_max': 1.3,
            'bifurc_angle_min': 175,
        }
        # TODO: use SGame methods
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

    def solver_setup(self) -> None:
        self.y0 = self.find_y0()
        # Note: homotopy parameter t goes from 1 to 0
        self.solver = HomCont(self.H, self.y0, self.J, t_target=0.0,
                              parameters=self.tracking_parameters['normal'],
                              x_transformer=self.x_transformer)

    def find_y0(self) -> np.ndarray:
        V = np.ones((self.game.num_states, self.game.num_players))
        return self.sigma_V_t_to_y(self.sigma_0, V, 1.0)

    def x_transformer(self, y: np.ndarray) -> np.ndarray:
        x = y.copy()
        z, t = x[0:self.game.num_actions_total], x[-1]
        x[0:self.game.num_actions_total] = 0.25 * (z + (z**2 + 4*t*self.sigma_0_flat**0.5)**0.5)**2
        return x

    def sigma_V_t_to_y(self, sigma: np.ndarray, V: np.ndarray, t: float) -> np.ndarray:
        sigma_flat = self.game.flatten_strategies(sigma)
        z_flat = (sigma_flat - t * self.sigma_0_flat**0.5) / sigma_flat**0.5
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
            import sgamesolver.homotopy._ipm_ct as ipm_ct

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


# %% Sympy implementation of IPM


class IPM_sp(IPM):
    """Interior point method homotopy: Sympy implementation"""

    def __init__(self, game: SGame, initial_strategies: Union[str, ArrayLike] = "centroid",
                 weights: Optional[ArrayLike] = None) -> None:
        super().__init__(game, initial_strategies, weights)

        # only import Sympy module on class instantiation
        try:
            import sympy as sp

        except ModuleNotFoundError:
            raise ModuleNotFoundError("Sympy implementation of IPM homotopy requires package 'sympy'.")

        sp.init_printing()

        num_s, num_p, nums_a = self.game.num_states, self.game.num_players, self.game.nums_actions
        num_a_max, num_a_tot = self.game.num_actions_max, self.game.num_actions_total

        # symbols
        y = sp.symarray('y', num_a_tot + num_s*num_p + 1)

        # strategies, values and homotopy parameter

        z = np.zeros((num_s, num_p, num_a_max), dtype='object')
        flat_index = 0
        for s in range(num_s):
            for p in range(num_p):
                for a in range(nums_a[s, p]):
                    z[s, p, a] = y[flat_index]
                    flat_index += 1

        V = np.zeros((num_s, num_p), dtype='object')
        flat_index = num_a_tot
        for s in range(num_s):
            for p in range(num_p):
                V[s, p] = y[flat_index]
                flat_index += 1

        t = y[-1]

        # transformations
        sigma = 0.25*(z + (z**2 + 4*t*self.sigma_0**0.5)**0.5)**2
        lambda_ = 0.25*(-z + (z**2 + 4*t*self.sigma_0**0.5)**0.5)**2

        # payoffs including continuation values

        # u_tilde[spA]
        u_tilde = np.zeros(self.game.payoffs.shape, dtype='object')
        u_tilde += self.game.payoffs
        for index, _ in np.ndenumerate(self.game.payoffs):
            for to_state in range(num_s):
                u_tilde[index] += (self.game.transitions[(*index, to_state)] * V[to_state]).sum()

        # Eu_tilde_a[spa]
        Eu_tilde_a = np.zeros((num_s, num_p, num_a_max), dtype='object')
        for index, util in np.ndenumerate(u_tilde):
            temp_prob = 1
            for other in range(num_p):
                if other == index[1]:
                    continue
                temp_prob *= sigma[index[0], other, index[other+2]]
            Eu_tilde_a[index[0], index[1], index[index[1]+2]] += temp_prob * util

        # homotopy function
        H_list = []
        for s in range(num_s):
            for p in range(num_p):
                for a in range(nums_a[s, p]):
                    H_list.append((1-t)*Eu_tilde_a[s, p, a] + lambda_[s, p, a] - V[s, p] - t*(1-t)*self.nu[s, p, a])
        for s in range(num_s):
            for p in range(num_p):
                H_list.append(sigma[s, p, 0:nums_a[s, p]].sum() - 1)
        H_sym = sp.Matrix(H_list)

        # Jacobian matrix
        J_sym = H_sym.jacobian(y)

        # lambdify
        self.H_num = sp.lambdify(y, H_sym, modules=['numpy'])
        self.J_num = sp.lambdify(y, J_sym, modules=['numpy'])

    def H(self, y: np.ndarray) -> np.ndarray:
        return self.H_num(*tuple(np.array(y)))[:, 0]

    def J(self, y: np.ndarray) -> np.ndarray:
        return self.J_num(*tuple(np.array(y)))

