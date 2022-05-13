"""(Logarithmic stochastic) tracing homotopy."""

# TODO: check user-provided priors and weights?
# TODO: maybe write custom optimization for find_y0 to avoid scipy import

import warnings
from typing import Union, Optional

import numpy as np
from scipy.optimize import brentq

from sgamesolver.sgame import SGame, LogStratHomotopy
from sgamesolver.homcont import HomCont

try:
    import sgamesolver.homotopy._logtracing_ct as _logtracing_ct

    ct = True
except ImportError:
    ct = False

ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def LogTracing(game: SGame, rho: Union[str, np.ndarray] = "centroid", nu: Optional[np.ndarray] = None,
               eta: float = 1.0, eta_fix: bool = False, implementation='auto', **kwargs):
    """Tracing homotopy for stochastic games."""
    if implementation == 'cython' or (implementation == 'auto' and ct):
        return LogTracing_ct(game, rho, nu, eta, eta_fix, **kwargs)
    else:
        if implementation == 'auto' and not ct:
            print('Defaulting to numpy implementation of LogTracing, because cython version is not installed. Numpy '
                  'may be substantially slower. For help setting up the cython version, please consult the manual.')
        return LogTracing_np(game, rho, nu, eta, eta_fix)


class LogTracing_base(LogStratHomotopy):
    """Tracing homotopy: base class"""

    def __init__(self, game: SGame, rho: Union[str, np.ndarray] = "centroid",
                 nu: Optional[np.ndarray] = None, eta: float = 1.0, eta_fix: bool = False) -> None:
        super().__init__(game)

        self.tracking_parameters['normal'] = {
            'convergence_tol': 1e-7,
            'corrector_tol': 1e-7,
            'ds_initial': 0.1,
            'ds_inflation_factor': 1.2,
            'ds_deflation_factor': 0.5,
            'ds_min': 1e-9,
            'ds_max': 100,
            'corrector_steps_max': 20,
            'corrector_distance_max': 0.5,
            'corrector_ratio_max': 0.5,
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
            'corrector_ratio_max': 0.3,
            'detJ_change_max': 1.3,
            'bifurcation_angle_min': 175,
        }

        if rho == "centroid":
            self.rho = self.game.centroid_strategy(zeros=True)
        elif rho == "random":
            self.rho = self.game.random_strategy(zeros=True)
        else:
            self.rho = np.array(rho, dtype=np.float64)

        if nu is None:
            self.nu = np.ones((self.game.num_states, self.game.num_players, self.game.num_actions_max))
        else:
            self.nu = np.array(nu, dtype=np.float64)

        self.eta = eta
        self.eta_fix = eta_fix

        # prepare payoffs and transition given other players follow prior
        num_s, num_p, num_a_max = self.game.num_states, self.game.num_players, self.game.num_actions_max
        rho_p_list = [self.rho[:, p, :] for p in range(num_p)]
        self.einsum_eqs = {
            'u_a': ['s' + ABC[0:num_p] + ',s' + ',s'.join(ABC[p_] for p_ in range(num_p) if p_ != p)
                    + '->s' + ABC[p] for p in range(num_p)],
            'phi_a': ['s' + ABC[0:num_p] + 't,s' + ',s'.join(ABC[p_] for p_ in range(num_p) if p_ != p)
                      + '->s' + ABC[p] + 't' for p in range(num_p)],
        }
        if num_p > 1:
            self.u_rho = np.zeros((num_s, num_p, num_a_max))
            self.phi_rho = np.zeros((num_s, num_p, num_a_max, num_s))
            for p in range(num_p):
                self.u_rho[:, p] = np.einsum(self.einsum_eqs['u_a'][p],
                                             self.game.payoffs[:, p], *(rho_p_list[:p] + rho_p_list[(p + 1):]))
                self.phi_rho[:, p] = np.einsum(self.einsum_eqs['phi_a'][p], self.game.phi,
                                               *(rho_p_list[:p] + rho_p_list[(p + 1):])) * self.game.discount_factors[p]
        else:
            self.u_rho = self.game.payoffs
            self.phi_rho = np.expand_dims(self.game.phi, 1) * self.game.discount_factors[0]

    def solver_setup(self) -> None:
        self.y0 = self.find_y0()
        self.solver = HomCont(self.H, self.y0, self.J, t_target=1.0,
                              parameters=self.tracking_parameters['normal'],
                              distance_function=self.sigma_distance)

    def find_y0(self, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
        """Value function iteration."""

        num_s, num_p, nums_a = self.game.num_states, self.game.num_players, self.game.nums_actions

        sigma_old = self.game.centroid_strategy()
        V_old = self.game.get_values(sigma_old)
        sigma = sigma_old.copy()
        V = V_old.copy()

        for _ in range(max_iter):
            c = self.u_rho + np.einsum('spaS,Sp->spa', self.phi_rho, V_old)

            # computation for all agents separately
            for s in range(num_s):
                for p in range(num_p):
                    # solve system of multi-linear equations
                    # - by first combining all equations to one big equation in sigma1
                    # - and then solving the equation by Brent's method
                    c_max_idx = np.argmax(c[s, p, 0:nums_a[s, p]])

                    numerator = self.nu[s, p, 0:nums_a[s, p]] / self.nu[s, p, c_max_idx]
                    denominator = (c[s, p, c_max_idx] - c[s, p, 0:nums_a[s, p]]) / (self.eta * self.nu[s, p, c_max_idx])

                    def f(sigma1):
                        return (numerator / (denominator + 1/sigma1)).sum() - 1

                    sigma1 = brentq(f, 1e-12, 1, xtol=1e-4, rtol=1e-4)

                    sigma[s, p, 0:nums_a[s, p]] = numerator / (denominator + 1 / sigma1)
                    V[s, p] = (c[s, p, c_max_idx] + ((self.eta * self.nu[s, p, c_max_idx]) / sigma1)
                               + self.eta * ((self.nu[s, p, 0:nums_a[s, p]]
                                              * (np.log(sigma[s, p, 0:nums_a[s, p]]) - 1)).sum()))

            if np.max(np.abs(V - V_old)) < tol and np.allclose(sigma, sigma_old, rtol=0, atol=tol, equal_nan=True):
                break
            else:
                V_old = V.copy()
                sigma_old = sigma.copy()
        else:  # loop ended without break
            warnings.warn('Value function iteration has not converged during computation of the starting point.')

        return self.sigma_V_t_to_y(sigma, V, 0.0)


class LogTracing_ct(LogTracing_base):
    """Tracing homotopy: Cython implementation"""

    def __init__(self, game: SGame, rho: Union[str, np.ndarray] = "centroid",
                 nu: Optional[np.ndarray] = None, eta: float = 1.0, eta_fix: bool = False, **kwargs):
        super().__init__(game, rho, nu, eta, eta_fix)
        self.cache = _logtracing_ct.TracingCache()
        self.parallel = False
        if 'parallel' in kwargs:
            self.parallel = kwargs['parallel']
        if 'cache' in kwargs and not kwargs['cache']:
            self.cache = None

    def H(self, y: np.ndarray) -> np.ndarray:
        return _logtracing_ct.H(y, self.game.u_ravel, self.game.phi_ravel, self.game.discount_factors,
                                self.rho, self.nu, self.eta, self.u_rho, self.phi_rho,
                                self.game.nums_actions, self.eta_fix, self.parallel, self.cache)

    def J(self, y: np.ndarray) -> np.ndarray:
        return _logtracing_ct.J(y, self.game.u_ravel, self.game.phi_ravel, self.game.discount_factors,
                                self.rho, self.nu, self.eta, self.u_rho, self.phi_rho,
                                self.game.nums_actions, self.eta_fix, self.parallel, self.cache)


class LogTracing_np(LogTracing_base):
    """Tracing homotopy: Numpy implementation"""

    def __init__(self, game: SGame, rho: Union[str, np.ndarray] = "centroid",
                 nu: Optional[np.ndarray] = None, eta: float = 1.0, eta_fix: bool = False, **kwargs):
        """prepares the following:
            - H_mask, J_mask
            - T_H, T_J
            - einsum_eqs
        """
        super().__init__(game, rho, nu, eta, eta_fix)
        # legacy version of transition arrays: needed until homotopy functions are updated
        self.game._make_transitions()

        num_s, num_p, nums_a = self.game.num_states, self.game.num_players, self.game.nums_actions
        num_a_max = self.game.num_actions_max

        # indices to mask H and J according to nums_a
        H_mask = []
        flat_index = 0
        for s in range(num_s):
            for p in range(num_p):
                for _ in range(nums_a[s, p]):
                    H_mask.append(flat_index)
                    flat_index += 1
                flat_index += num_a_max - nums_a[s, p]
        for s in range(num_s):
            for p in range(num_p):
                H_mask.append(flat_index)
                flat_index += 1
        self.H_mask = np.array(H_mask, dtype=np.int64)

        self.J_mask = tuple(
            np.meshgrid(
                H_mask,
                np.append(H_mask, [num_s * num_p * num_a_max + num_s * num_p]),
                indexing="ij",
                sparse=True,
            )
        )

        # tensors to assemble J
        T_J_0 = np.zeros((num_s, num_p, num_a_max, num_s, num_p, num_a_max))
        for s in range(num_s):
            for p in range(num_p):
                for a in range(nums_a[s, p]):
                    T_J_0[s, p, a, s, p, a] = 1

        T_J_1 = np.zeros((num_s, num_p, num_a_max, num_s, num_p, num_a_max))
        for s in range(num_s):
            for p in range(num_p):
                for a in range(nums_a[s, p]):
                    for b in range(nums_a[s, p]):
                        if b != a:
                            T_J_1[s, p, a, s, p, b] = 1

        T_J_2 = np.zeros((num_s, num_p, num_a_max, num_s, num_p, num_a_max))
        for s in range(num_s):
            for p in range(num_p):
                for q in range(num_p):
                    if q != p:
                        T_J_2[s, p, :, s, q, :] = 1

        T_J_3 = np.zeros((num_s, num_p, num_a_max, num_s, num_p))
        for p in range(num_p):
            T_J_3[:, p, :, :, p] = 1

        T_J_4 = np.zeros((num_s, num_p, num_a_max, num_s, num_p))
        for s in range(num_s):
            for p in range(num_p):
                T_J_4[s, p, :, s, p] = 1

        T_J_5 = np.zeros((num_s, num_p, num_s, num_p, num_a_max))
        for s in range(num_s):
            for p in range(num_p):
                T_J_5[s, p, s, p, :] = 1

        self.T_J = {0: T_J_0,
                    1: T_J_1,
                    2: T_J_2,
                    3: T_J_3,
                    4: T_J_4,
                    5: T_J_5}

        # equations to be used by einsum
        self.einsum_eqs['u_ab'] = [
            ['s' + ABC[0:num_p] + ',s'.join([''] + [ABC[p_] for p_ in range(num_p) if p_ not in [p, q]])
             + '->s' + ABC[p] + (ABC[q] if q != p else '') for q in range(num_p)] for p in range(num_p)
        ]

    def H(self, y: np.ndarray) -> np.ndarray:
        """Homotopy function."""

        num_s, num_p = self.game.num_states, self.game.num_players
        num_a_max, num_a_tot = self.game.num_actions_max, self.game.num_actions_total

        sigma, V, t = self.y_to_sigma_V_t(y, zeros=True)

        # eta_fix == False is a version of the logarithmic tracing procedure that sets η(t) = (1-t)*η_0
        if self.eta_fix:
            eta = self.eta
        else:
            eta = (1 - t) * self.eta

        # building blocks of H

        beta_with_nan = self.game.unflatten_strategies(y[:num_a_tot])
        sigma_inv = np.nan_to_num(np.exp(-beta_with_nan))
        beta = np.nan_to_num(beta_with_nan, nan=1.0)

        sigma_p_list = [sigma[:, p, :] for p in range(num_p)]

        if num_p > 1:
            u_sigma = np.empty((num_s, num_p, num_a_max))
            phi_sigma = np.empty((num_s, num_p, num_a_max, num_s))
            for p in range(num_p):
                u_sigma[:, p] = np.einsum(self.einsum_eqs['u_a'][p], self.game.payoffs[:, p],
                                          *(sigma_p_list[:p] + sigma_p_list[(p + 1):]))
                phi_sigma[:, p] = np.einsum(self.einsum_eqs['phi_a'][p], self.game.transitions[:, p],
                                            *(sigma_p_list[:p] + sigma_p_list[(p + 1):]))
        else:
            u_sigma = self.game.payoffs
            phi_sigma = self.game.transitions

        u_bar = t * u_sigma + (1 - t) * self.u_rho
        phi_bar = t * phi_sigma + (1 - t) * self.phi_rho

        Eu_tilde_a = u_bar + np.einsum('spat,tp->spa', phi_bar, V)

        # assemble H

        spa = num_s * num_p * num_a_max
        sp = num_s * num_p
        H = np.empty(spa + sp)

        # H_val
        H[0:spa] = (Eu_tilde_a - np.repeat(V[:, :, np.newaxis], num_a_max, axis=2) + (1 - t) * eta
                    * (self.nu * sigma_inv
                       + np.repeat((self.nu * (beta - 1)).sum(axis=2)[:, :, np.newaxis], num_a_max, axis=2))
                    ).reshape(spa)
        # H_strat
        H[spa: spa + sp] = (np.sum(sigma, axis=2) - np.ones((num_s, num_p))).reshape(sp)

        return H[self.H_mask]

    def J(self, y: np.ndarray) -> np.ndarray:
        """Jacobian matrix of homotopy function."""

        num_s, num_p = self.game.num_states, self.game.num_players
        num_a_max, num_a_tot = self.game.num_actions_max, self.game.num_actions_total

        sigma, V, t = self.y_to_sigma_V_t(y, zeros=True)

        # eta_fix == False is a version of the logarithmic tracing procedure that sets η(t) = (1-t)*η_0
        # For J, not only eta itself is adjusted, but also a factor in the column containing dH/dt:
        # eta fixed: d/dt (1-t)η = -η
        # eta varies in t: d/dt (1-t)η = d/dt (1-t)^2 η_0 = -2(1-t)η_0 = -2η
        if self.eta_fix:
            eta = self.eta
            eta_col_factor = 1.0
        else:
            eta = (1 - t) * self.eta
            eta_col_factor = 2.0

        # building blocks of J

        beta_with_nan = self.game.unflatten_strategies(y[:num_a_tot])
        sigma_inv = np.nan_to_num(np.exp(-beta_with_nan))
        beta = np.nan_to_num(beta_with_nan, nan=1.0)

        sigma_p_list = [sigma[:, p, :] for p in range(num_p)]

        u_tilde = self.game.payoffs + np.einsum('sp...S,Sp->sp...', self.game.transitions, V)

        if num_p > 1:
            u_sigma = np.empty((num_s, num_p, num_a_max))
            phi_sigma = np.empty((num_s, num_p, num_a_max, num_s))
            Eu_tilde_ab = np.empty((num_s, num_p, num_p, num_a_max, num_a_max))
            for p in range(num_p):
                u_sigma[:, p] = np.einsum(self.einsum_eqs['u_a'][p], self.game.payoffs[:, p],
                                          *(sigma_p_list[:p] + sigma_p_list[(p + 1):]))
                phi_sigma[:, p] = np.einsum(self.einsum_eqs['phi_a'][p], self.game.transitions[:, p],
                                            *(sigma_p_list[:p] + sigma_p_list[(p + 1):]))
                for q in range(num_p):
                    Eu_tilde_pq = np.einsum(self.einsum_eqs['u_ab'][p][q], u_tilde[:, p],
                                            *[sigma_p_list[p_] for p_ in range(num_p) if p_ not in [p, q]])
                    if q == p:
                        Eu_tilde_pq = np.repeat(np.expand_dims(Eu_tilde_pq, axis=-1), num_a_max, axis=-1)
                    Eu_tilde_ab[:, p, q] = Eu_tilde_pq
        else:
            u_sigma = self.game.payoffs
            phi_sigma = self.game.transitions
            Eu_tilde_ab = np.repeat(u_tilde[:, :, np.newaxis, :, np.newaxis], num_a_max, axis=4)

        u_hat = u_sigma - self.u_rho
        phi_hat = phi_sigma - self.phi_rho

        phi_bar = t * phi_sigma + (1 - t) * self.phi_rho

        # assemble J

        spa = num_s * num_p * num_a_max
        sp = num_s * num_p
        J = np.zeros((spa + sp, spa + sp + 1))

        # dH_val_dbeta
        J[0:spa, 0:spa] = (
                (1 - t) * eta * np.einsum('spaSPA,SPA->spaSPA', self.T_J[0], self.nu * (1 - sigma_inv))
                + (1 - t) * eta * np.einsum('spaSPA,spA->spaSPA', self.T_J[1], self.nu)
                + t * np.einsum('spaSPA,sPA,spPaA->spaSPA', self.T_J[2], sigma, Eu_tilde_ab)
        ).reshape((spa, spa))
        # dH_val_dV
        J[0:spa, spa: spa + sp] = (np.einsum('spaSP,spaS->spaSP', self.T_J[3], phi_bar) - self.T_J[4]).reshape(
            (spa, sp))
        # dH_val_dt
        J[0:spa, spa + sp] = (u_hat + np.einsum('spaS,Sp->spa', phi_hat, V) - eta_col_factor * eta
                              * (self.nu * sigma_inv + np.repeat((self.nu * (beta - 1)).sum(axis=2)[:, :, np.newaxis],
                                                                 num_a_max, axis=2))
                              ).reshape(spa)
        # dH_strat_dbeta
        J[spa: spa + sp, 0:spa] = np.einsum('spSPA,SPA->spSPA', self.T_J[5], sigma).reshape((sp, spa))
        # dH_strat_dV = 0
        # dH_strat_dt = 0

        return J[self.J_mask]
