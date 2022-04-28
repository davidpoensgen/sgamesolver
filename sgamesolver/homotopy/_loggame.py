"""Logarithmic game homotopy."""

from typing import Optional

import numpy as np

from sgamesolver.sgame import SGame, LogStratHomotopy
from sgamesolver.homcont import HomCont

try:
    import sgamesolver.homotopy._loggame_ct as _loggame_ct
    ct = True
except ImportError:
    ct = False


ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def LogGame(game: SGame, nu: Optional[np.ndarray] = None, implementation='auto'):
    """LogGame homotopy for stochastic games."""
    if implementation == 'cython' or (implementation == 'auto' and ct):
        return LogGame_ct(game, nu)
    else:
        if implementation == 'auto' and not ct:
            print('Defaulting to numpy implementation of LogGame, because cython version is not installed. Numpy '
                  'may be substantially slower. For help setting up the cython version, please consult the manual.')
        return LogGame_np(game, nu)


class LogGame_base(LogStratHomotopy):
    """Logarithmic game homotopy: base class"""

    def __init__(self, game: SGame, nu: Optional[np.ndarray] = None) -> None:
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
            'ds_max': 10,
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

        if nu is None:
            self.nu = np.ones((self.game.num_states, self.game.num_players, self.game.num_actions_max))
        else:
            self.nu = nu

    def solver_setup(self) -> None:
        self.y0 = self.find_y0()
        self.solver = HomCont(self.H, self.y0, self.J, t_target=1.0,
                              parameters=self.tracking_parameters['normal'],
                              distance_function=self.sigma_distance)

    def find_y0(self) -> np.ndarray:
        sigma = self.game.weighted_centroid_strategy(self.nu)

        # compute values including log penalty terms
        V = np.empty((self.game.num_states, self.game.num_players))
        for s in range(self.game.num_states):
            for p in range(self.game.num_players):
                V[s, p] = (self.nu[s, p, 0] / sigma[s, p, 0]
                           + np.sum(self.nu[s, p, 0:self.game.nums_actions[s, p]] * (np.log(sigma[s, p, 0]) - 1)))

        return self.sigma_V_t_to_y(sigma, V, 0.0)


class LogGame_ct(LogGame_base):
    """Logarithmic game homotopy: Cython implementation"""

    def H(self, y: np.ndarray) -> np.ndarray:
        return _loggame_ct.H(y, self.game.payoffs, self.game.transitions, self.nu,
                             self.game.num_states, self.game.num_players, self.game.nums_actions,
                             self.game.num_actions_max, self.game.num_actions_total)

    def J(self, y: np.ndarray) -> np.ndarray:
        return _loggame_ct.J(y, self.game.payoffs, self.game.transitions, self.nu,
                             self.game.num_states, self.game.num_players, self.game.nums_actions,
                             self.game.num_actions_max, self.game.num_actions_total)


class LogGame_np(LogGame_base):
    """Logarithmic game homotopy: Numpy implementation"""

    def __init__(self, game: SGame, nu: Optional[np.ndarray] = None) -> None:
        """prepares the following:
            - H_mask, J_mask
            - T_H, T_J
            - einsum_eqs
        """
        super().__init__(game, nu)

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
        self.einsum_eqs = {
            'u_a': ['s' + ABC[0:num_p] + ',s' + ',s'.join(ABC[p_] for p_ in range(num_p) if p_ != p)
                    + '->s' + ABC[p] for p in range(num_p)],
            'phi_a': ['s' + ABC[0:num_p] + 't,s' + ',s'.join(ABC[p_] for p_ in range(num_p) if p_ != p)
                      + '->s' + ABC[p] + 't' for p in range(num_p)],
            'u_ab': [['s' + ABC[0:num_p] + ',s'.join(['']+[ABC[p_] for p_ in range(num_p) if p_ not in [p, q]])
                      + '->s' + ABC[p] + (ABC[q] if q != p else '') for q in range(num_p)] for p in range(num_p)]
        }

    def H(self, y: np.ndarray) -> np.ndarray:
        """Homotopy function."""

        num_s, num_p = self.game.num_states, self.game.num_players
        num_a_max, num_a_tot = self.game.num_actions_max, self.game.num_actions_total

        sigma, V, t = self.y_to_sigma_V_t(y, zeros=True)

        # building blocks of H

        beta_with_nan = self.game.unflatten_strategies(y[:num_a_tot])
        sigma_inv = np.nan_to_num(np.exp(-beta_with_nan))
        beta = np.nan_to_num(beta_with_nan, nan=1.0)

        sigma_p_list = [sigma[:, p, :] for p in range(num_p)]

        u_tilde = self.game.payoffs + np.einsum('sp...S,Sp->sp...', self.game.transitions, V)

        # TODO: sigma_prod_with_p ?

        if num_p > 1:
            Eu_tilde_a = np.empty((num_s, num_p, num_a_max))
            for p in range(num_p):
                Eu_tilde_a[:, p] = np.einsum(self.einsum_eqs['u_a'][p], u_tilde[:, p],
                                             *(sigma_p_list[:p] + sigma_p_list[(p + 1):]))
        else:
            Eu_tilde_a = u_tilde

        # assemble H

        spa = num_s * num_p * num_a_max
        sp = num_s * num_p
        H = np.zeros(spa+sp)

        # H_val
        H[0:spa] = (- np.repeat(V[:, :, np.newaxis], num_a_max, axis=2) + t * Eu_tilde_a
                    + (1-t) * (self.nu * sigma_inv
                               + np.repeat((self.nu*(beta-1)).sum(axis=2)[:, :, np.newaxis], num_a_max, axis=2))
                    ).reshape(spa)
        # H_strat
        H[spa : spa+sp] = (np.sum(sigma, axis=2) - np.ones((num_s, num_p))).reshape(sp)

        return H[self.H_mask]

    def J(self, y: np.ndarray) -> np.ndarray:
        """Jacobian matrix of homotopy function."""

        num_s, num_p = self.game.num_states, self.game.num_players
        num_a_max, num_a_tot = self.game.num_actions_max, self.game.num_actions_total

        sigma, V, t = self.y_to_sigma_V_t(y, zeros=True)

        # building blocks of J

        beta_with_nan = self.game.unflatten_strategies(y[:num_a_tot])
        sigma_inv = np.nan_to_num(np.exp(-beta_with_nan))
        beta = np.nan_to_num(beta_with_nan, nan=1.0)

        sigma_p_list = [sigma[:, p, :] for p in range(num_p)]

        u_tilde = self.game.payoffs + np.einsum('sp...S,Sp->sp...', self.game.transitions, V)

        if num_p > 1:
            Eu_tilde_a = np.empty((num_s, num_p, num_a_max))
            phi_a = np.empty((num_s, num_p, num_a_max, num_s))
            Eu_tilde_ab = np.empty((num_s, num_p, num_p, num_a_max, num_a_max))
            for p in range(num_p):
                Eu_tilde_a[:, p] = np.einsum(self.einsum_eqs['u_a'][p], u_tilde[:, p],
                                             *(sigma_p_list[:p] + sigma_p_list[(p + 1):]))
                phi_a[:, p] = np.einsum(self.einsum_eqs['phi_a'][p], self.game.transitions[:, p],
                                        *(sigma_p_list[:p] + sigma_p_list[(p + 1):]))
                for q in range(num_p):
                    # TODO: can loop be made more efficient?
                    Eu_tilde_pq = np.einsum(self.einsum_eqs['u_ab'][p][q], u_tilde[:, p],
                                            *[sigma_p_list[p_] for p_ in range(num_p) if p_ not in [p, q]])
                    if q == p:
                        Eu_tilde_pq = np.repeat(np.expand_dims(Eu_tilde_pq, axis=-1), num_a_max, axis=-1)
                    Eu_tilde_ab[:, p, q] = Eu_tilde_pq
        else:
            Eu_tilde_a = u_tilde
            phi_a = self.game.transitions
            Eu_tilde_ab = np.repeat(u_tilde[:, :, np.newaxis, :, np.newaxis], num_a_max, axis=4)

        # assemble J

        spa = num_s * num_p * num_a_max
        sp = num_s * num_p
        J = np.zeros((spa+sp, spa+sp+1))

        # dH_val_dbeta
        J[0:spa, 0:spa] = ((1-t) * np.einsum('spaSPA,SPA->spaSPA', self.T_J[0], self.nu*(1-sigma_inv))
                           + (1-t) * np.einsum('spaSPA,spA->spaSPA', self.T_J[1], self.nu)
                           + t * np.einsum('spaSPA,sPA,spPaA->spaSPA', self.T_J[2], sigma, Eu_tilde_ab)
                           ).reshape((spa, spa))
        # dH_val_dV
        J[0:spa, spa : spa+sp] = (t * np.einsum('spaSP,spaS->spaSP', self.T_J[3], phi_a) - self.T_J[4]
                                  ).reshape((spa, sp))
        # dH_val_dt
        J[0:spa, spa+sp] = (Eu_tilde_a
                            - (self.nu*sigma_inv + np.repeat((self.nu*(beta-1)).sum(axis=2)[:, :, np.newaxis],
                                                             num_a_max, axis=2))
                            ).reshape(spa)
        # dH_strat_dbeta
        J[spa : spa+sp, 0:spa] = np.einsum('spSPA,SPA->spSPA', self.T_J[5], sigma).reshape((sp, spa))
        # dH_strat_dV = 0
        # dH_strat_dt = 0

        return J[self.J_mask]
