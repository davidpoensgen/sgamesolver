"""Interior point method homotopy."""

# TODO: @Steffen, please re-write to remove x-transformer;
# TODO: write a distance function instead if desired (don't think that makes sense for IPM though?)

from typing import Union, Optional, Tuple
from warnings import warn
import numpy as np

from sgamesolver.sgame import SGame, SGameHomotopy
from sgamesolver.homcont import HomContSolver

try:
    import sgamesolver.homotopy._ipm_ct as _ipm_ct

    ct = True
except ImportError:
    ct = False

ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def IPM(game: SGame, initial_strategies: Union[str, np.ndarray] = 'centroid',
        weights: Optional[np.ndarray] = None, implementation='auto'):
    """Interior point method (IPM) homotopy for stochastic games."""
    if implementation == 'cython' or (implementation == 'auto' and ct):
        return IPM_ct(game, initial_strategies, weights)
    else:
        if implementation == 'auto' and not ct:
            warn('Defaulting to sympy+numpy implementation of IPM, because cython version is not installed. This '
                 'version is substantially slower. For help setting up the cython version, please consult the manual.')
        return IPM_sp(game, initial_strategies, weights)


# %% parent class for interior point method homotopy


class IPM_base(SGameHomotopy):
    """Interior point method homotopy: base class"""

    def __init__(self, game: SGame, initial_strategies: Union[str, np.ndarray] = "centroid",
                 weights: Optional[np.ndarray] = None) -> None:
        super().__init__(game)
        # legacy version of transition arrays: needed until homotopy functions are updated
        self.game._make_transitions()

        self.tracking_parameters['normal'] = {
            'convergence_tol': 1e-7,
            'corrector_tol': 1e-7,
            'ds_initial': 0.1,
            'ds_inflation_factor': 1.2,
            'ds_deflation_factor': 0.5,
            'ds_min': 1e-9,
            'ds_max': 1000,
            'corrector_steps_max': 20,
            'corrector_distance_max': 0.5,
            'corrector_ratio_max': 0.9,
            'detJ_change_max': 1.5,
            'bifurcation_angle_min': 175,
        }

        self.tracking_parameters['robust'] = {
            'convergence_tol': 1e-7,
            'corrector_tol': 1e-8,
            'ds_initial': 0.1,
            'ds_inflation_factor': 1.1,
            'ds_deflation_factor': 0.5,
            'ds_min': 1e-9,
            'ds_max': 10,
            'corrector_steps_max': 30,
            'corrector_distance_max': 0.3,
            'corrector_ratio_max': 0.7,
            'detJ_change_max': 1.3,
            'bifurcation_angle_min': 175,
        }

        if initial_strategies == "centroid":
            self.sigma_0 = self.game.centroid_strategy(zeros=True)
        elif initial_strategies == "random":
            self.sigma_0 = self.game.random_strategy(zeros=True)
        else:
            self.sigma_0 = np.array(initial_strategies)

        self.sigma_0_flat = self.game.flatten_strategies(self.sigma_0)

        if weights is None:
            self.nu = np.ones((self.game.num_states, self.game.num_players, self.game.num_actions_max))
        else:
            self.nu = weights

    def solver_setup(self) -> None:
        self.y0 = self.find_y0()
        # Note: homotopy parameter t goes from 1 to 0
        self.solver = HomContSolver(self.H, self.y0, self.J, t_target=0.0,
                                    parameters=self.tracking_parameters['normal'])

    def find_y0(self) -> np.ndarray:
        V = np.ones((self.game.num_states, self.game.num_players))
        return self.sigma_V_t_to_y(self.sigma_0, V, 1.0)

    def x_transformer(self, y: np.ndarray) -> np.ndarray:
        x = y.copy()
        z, t = x[0:self.game.num_actions_total], x[-1]
        x[0:self.game.num_actions_total] = 0.25 * (z + (z ** 2 + 4 * t * self.sigma_0_flat ** 0.5) ** 0.5) ** 2
        return x

    def sigma_V_t_to_y(self, sigma: np.ndarray, V: np.ndarray, t: float) -> np.ndarray:
        sigma_flat = self.game.flatten_strategies(sigma)
        z_flat = (sigma_flat - t * self.sigma_0_flat ** 0.5) / sigma_flat ** 0.5
        V_flat = self.game.flatten_values(V)
        return np.concatenate([z_flat, V_flat, [t]])

    def y_to_sigma_V_t(self, y: np.ndarray, zeros: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        sigma_V_t_flat = self.x_transformer(y)
        sigma = self.game.unflatten_strategies(sigma_V_t_flat[0:self.game.num_actions_total], zeros=zeros)
        V = self.game.unflatten_values(sigma_V_t_flat[self.game.num_actions_total:-1])
        t = sigma_V_t_flat[-1]
        return sigma, V, t


# %% Cython implementation of IPM


class IPM_ct(IPM_base):
    """Interior point method homotopy: Cython implementation"""

    def __init__(self, game: SGame, initial_strategies: Union[str, np.ndarray] = "centroid",
                 weights: Optional[np.ndarray] = None) -> None:
        super().__init__(game, initial_strategies, weights)

    def H(self, y: np.ndarray) -> np.ndarray:
        return _ipm_ct.H(y, self.game.payoffs, self.game.transitions, self.sigma_0, self.nu,
                         self.game.num_states, self.game.num_players, self.game.nums_actions,
                         self.game.num_actions_max, self.game.num_actions_total)

    def J(self, y: np.ndarray) -> np.ndarray:
        return _ipm_ct.J(y, self.game.payoffs, self.game.transitions, self.sigma_0, self.nu,
                         self.game.num_states, self.game.num_players, self.game.nums_actions,
                         self.game.num_actions_max, self.game.num_actions_total)


# %% Sympy implementation of IPM


class IPM_sp(IPM_base):
    """Interior point method homotopy: Sympy implementation"""

    def __init__(self, game: SGame, initial_strategies: Union[str, np.ndarray] = "centroid",
                 weights: Optional[np.ndarray] = None) -> None:
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
        y = sp.symarray('y', num_a_tot + num_s * num_p + 1)

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
        sigma = 0.25 * (z + (z ** 2 + 4 * t * self.sigma_0 ** 0.5) ** 0.5) ** 2
        lambda_ = 0.25 * (-z + (z ** 2 + 4 * t * self.sigma_0 ** 0.5) ** 0.5) ** 2

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
                temp_prob *= sigma[index[0], other, index[other + 2]]
            Eu_tilde_a[index[0], index[1], index[index[1] + 2]] += temp_prob * util

        # homotopy function
        H_list = []
        for s in range(num_s):
            for p in range(num_p):
                for a in range(nums_a[s, p]):
                    H_list.append(
                        (1 - t) * Eu_tilde_a[s, p, a] + lambda_[s, p, a] - V[s, p] - t * (1 - t) * self.nu[s, p, a])
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
