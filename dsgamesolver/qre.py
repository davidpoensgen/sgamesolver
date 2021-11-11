"""(Markov logit) quantal response equilibrium (QRE) homotopy."""

# TODO: play with einsum_path
# seems to slow things down considerably, which is strange (there is an open github issue on it)
# can be tested with option dev=True in H and J
# remove dev option once development is finished

# TODO: adjust tracking parameters with "scale" of game

# TODO: think about Cython import
# Cython compilation complains about depreciated Numpy API.
# Could be suppressed in general setup.py with 'define_macros': [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')].
# Seemingly cannot be suppressed with pxyimport.
# Stick to pxyimport?
# Pre-compile anyway?

# TODO: play with numba boost on numpy implementation


import numpy as np

from dsgamesolver.sgame import SGame, SGameHomotopy
from dsgamesolver.homcont import HomCont

ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# %% parent class for QRE homotopy


class QRE(SGameHomotopy):
    """QRE homotopy: base class"""

    def __init__(self, game: SGame) -> None:
        super().__init__(game)

        # TODO: adjust parameters with scale of payoff matrix:

        self.tracking_parameters['normal'] = {
            'x_tol': 1e-7,
            't_tol': 1e-7,
            'H_tol': 1e-7,
            'ds0': 0.01,
            'ds_infl': 1.2,
            'ds_defl': 0.5,
            'ds_min': 1e-9,
            'ds_max': 1000,
            'corr_steps_max': 20,
            'corr_dist_max': 0.3,
            'corr_ratio_max': 0.3,
            'detJ_change_max': 1.3,
            'bifurc_angle_min': 175,
        }

        self.tracking_parameters['robust'] = {
            'x_tol': 1e-7,
            't_tol': 1e-7,
            'H_tol': 1e-8,
            'ds0': 0.01,
            'ds_infl': 1.1,
            'ds_defl': 0.5,
            'ds_min': 1e-9,
            'ds_max': 1000,
            'corr_steps_max': 30,
            'corr_dist_max': 0.1,
            'corr_ratio_max': 0.1,
            'detJ_change_max': 1.1,
            'bifurc_angle_min': 175,
        }

    def initialize(self, target_lambda: float = np.inf) -> None:
        self.y0 = self.find_y0()
        self.solver = HomCont(self.H, self.y0, self.J, t_target=target_lambda,
                              parameters=self.tracking_parameters['normal'],
                              x_transformer=self.x_transformer, store_path=True)

    def find_y0(self) -> np.ndarray:
        sigma = self.game.centroid_strategy()
        V = self.game.get_values(sigma)
        return self.sigma_V_t_to_y(sigma, V, 0.0)


# %% Numpy implementation of QRE


class QRE_np(QRE):
    """QRE homotopy: Numpy implementation"""

    def __init__(self, game: SGame) -> None:  # sourcery no-metrics
        """Prepares the following:
            - H_mask, J_mask
            - T_H, T_J
            - einsum_eqs
        """
        super().__init__(game)

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

        # tensors to assemble H
        T_H_0 = np.zeros((num_s, num_p, num_a_max))
        for s in range(num_s):
            for p in range(num_p):
                T_H_0[s, p, 0] = 1

        T_H_1 = np.zeros((num_s, num_p, num_a_max, num_s, num_p, num_a_max))
        for s in range(num_s):
            for p in range(num_p):
                T_H_1[s, p, 0, s, p, :] = -1

        T_H_2 = np.zeros((num_s, num_p, num_a_max, num_s, num_p, num_a_max))
        for s in range(num_s):
            for p in range(num_p):
                for a in range(1, nums_a[s, p]):
                    T_H_2[s, p, a, s, p, a] = -1
                    T_H_2[s, p, a, s, p, 0] = 1

        T_H_3 = np.zeros((num_s, num_p, num_s, num_p))
        for s in range(num_s):
            for p in range(num_p):
                T_H_3[s, p, s, p] = -1

        self.T_H = {0: T_H_0, 1: T_H_1, 2: T_H_2, 3: T_H_3}

        # tensors to assemble J
        T_J_qre_temp = np.zeros((num_s, num_p, num_a_max, num_s, num_p, num_a_max))
        for s in range(num_s):
            for p in range(num_p):
                for a in range(nums_a[s, p]):
                    T_J_qre_temp[s, p, a, s, p, a] = 1

        T_J_0 = np.einsum('spatqb,tqbSPA->spaSPA', T_H_2, T_J_qre_temp)

        T_J_1 = np.einsum('spatqb,tqbSPA->spaSPA', T_H_1, T_J_qre_temp)

        T_J_qre_temp = np.zeros((num_s, num_p, num_s, num_p))
        for s in range(num_s):
            for p in range(num_p):
                T_J_qre_temp[s, p, s, p] = 1

        T_J_3 = np.einsum('sp...t,tpSP->sp...SP', self.game.transitions, T_J_qre_temp)

        T_J_5 = np.einsum('sptq,tqSP->spSP', T_H_3, T_J_qre_temp)

        T_J_2 = np.zeros((num_s, num_p, *[num_a_max] * (num_p - 1), num_s, num_p, num_a_max))
        for s in range(num_s):
            for p in range(num_p):
                a_profiles_without_p = list(np.ndindex(tuple(nums_a[s, :p]) + tuple(nums_a[s, (p + 1):])))
                for A in a_profiles_without_p:
                    for p_ in range(num_p):
                        if p_ != p:
                            a_ = A[p_] if p_ < p else A[p_ - 1]
                            T_J_2[(s, p) + A + (s, p_, a_)] = 1

        T_J_4 = np.zeros((num_s, num_p, *[num_a_max] * num_p, num_s, num_p, num_a_max))
        for s in range(num_s):
            for p in range(num_p):
                a_profiles = list(np.ndindex(tuple(nums_a[s, :])))
                for A in a_profiles:
                    for p_ in range(num_p):
                        T_J_4[(s, p) + A + (s, p_, A[p_])] = 1

        self.T_J = {0: T_J_0,
                    1: T_J_1,
                    2: T_J_2,
                    3: T_J_3,
                    4: T_J_4,
                    5: T_J_5}

        # equations to be used by einsum
        self.einsum_eqs = {
            'sigma_prod': 's' + ',s'.join(ABC[0:num_p]) + '->s' + ABC[0:num_p],
            'sigma_prod_with_p': ['s' + ',s'.join(ABC[0:num_p]) + '->s' + ABC[0:num_p] for p in range(num_p)],
            'Eu_tilde_a_H': ['s' + ABC[0:num_p] + ',s' + ',s'.join(ABC[p_] for p_ in range(num_p) if p_ != p)
                             + '->s' + ABC[p] for p in range(num_p)],
            'Eu_tilde_a_J': ['s' + ABC[0:num_p] + ',s' + ABC[0:num_p] + '->s' + ABC[p] for p in range(num_p)],
            'dEu_tilde_a_dbeta': ['s' + ABC[0:num_p] + ',s' + ''.join(ABC[p_] for p_ in range(num_p) if p_ != p)
                                  + 'tqb->s' + ABC[p] + 'tqb' for p in range(num_p)],
            'dEu_tilde_a_dV': ['s' + ABC[0:num_p] + 'tp,s' + ABC[0:num_p] + '->s' + ABC[p] + 'tp'
                               for p in range(num_p)],
            'dEu_tilde_dbeta': 'sp' + ABC[0:num_p] + ',sp' + ABC[0:num_p] + 'tqb->sptqb',
        }

        # optimal paths to be used by einsum
        # TODO: seems to slow computation down considerably, remove once development is completed
        sigma = self.game.centroid_strategy()
        V = self.game.get_values(sigma)
        sigma_p_list = [sigma[:, p, :] for p in range(num_p)]
        u_tilde = self.game.payoffs + np.einsum('sp...S,Sp->sp...', self.game.transitions, V)
        self.einsum_paths = {
            'u_tilde': np.einsum_path('sp...S,Sp->sp...', self.game.transitions, V, optimize=('optimal'))[0],
            'Eu_tilde_a': [np.einsum_path(self.einsum_eqs['Eu_tilde_a_H'][p], u_tilde[:, p],
                                          *(sigma_p_list[:p] + sigma_p_list[(p + 1):]), optimize=('optimal'))[0]
                           for p in range(num_p)],
        }

    def H(self, y: np.ndarray, dev: bool = False) -> np.ndarray:
        """Homotopy function."""

        num_s, num_p = self.game.num_states, self.game.num_players
        num_a_max, num_a_tot = self.game.num_actions_max, self.game.num_actions_total

        beta = self.game.unflatten_strategies(y[:num_a_tot], zeros=True)
        sigma, V, lambda_ = self.y_to_sigma_V_t(y, zeros=True)

        # building blocks of H

        sigma_p_list = [sigma[:, p, :] for p in range(num_p)]
        if dev:
            u_tilde = self.game.payoffs + np.einsum('sp...S,Sp->sp...', self.game.transitions, V,
                                                    optimize=False)
        else:
            u_tilde = self.game.payoffs + np.einsum('sp...S,Sp->sp...', self.game.transitions, V)

        if num_p > 1:
            Eu_tilde_a = np.empty((num_s, num_p, num_a_max))
            for p in range(num_p):
                if dev:
                    Eu_tilde_a[:, p] = np.einsum(self.einsum_eqs['Eu_tilde_a_H'][p], u_tilde[:, p],
                                                 *(sigma_p_list[:p] + sigma_p_list[(p + 1):]),
                                                 optimize=self.einsum_paths['Eu_tilde_a'][p])
                else:
                    Eu_tilde_a[:, p] = np.einsum(self.einsum_eqs['Eu_tilde_a_H'][p], u_tilde[:, p],
                                                 *(sigma_p_list[:p] + sigma_p_list[(p + 1):]))
        else:
            Eu_tilde_a = u_tilde

        Eu_tilde = np.einsum('spa,spa->sp', sigma, Eu_tilde_a)

        # assemble H

        spa = num_s * num_p * num_a_max
        sp = num_s * num_p
        H = np.zeros(spa+sp)

        # H_strat
        H[0:spa] = (self.T_H[0] + np.einsum('spaSPA,SPA->spa', self.T_H[1], sigma)
                    + np.einsum('spaSPA,SPA->spa', self.T_H[2], beta - lambda_ * Eu_tilde_a)
                    ).reshape(spa)
        # H_val
        H[spa : spa+sp] = np.einsum('spSP,SP->sp', self.T_H[3], V - Eu_tilde).reshape(sp)

        return H[self.H_mask]

    def J(self, y: np.ndarray, dev: bool = False) -> np.ndarray:
        """Jacobian matrix of homotopy function."""

        num_s, num_p, num_a_max = self.game.num_states, self.game.num_players, self.game.num_actions_max

        sigma, V, lambda_ = self.y_to_sigma_V_t(y, zeros=True)

        # building blocks of J

        sigma_p_list = [sigma[:, p, :] for p in range(num_p)]
        u_tilde = self.game.payoffs + np.einsum('sp...S,Sp->sp...', self.game.transitions, V)

        sigma_prod = np.einsum(self.einsum_eqs['sigma_prod'], *sigma_p_list)

        sigma_prod_with_p = np.empty((num_s, num_p, *[num_a_max] * num_p))
        for p in range(num_p):
            sigma_p_list_with_p = sigma_p_list[:p] + [np.ones_like(sigma[:, p, :])] + sigma_p_list[(p + 1):]
            sigma_prod_with_p[:, p] = np.einsum(self.einsum_eqs['sigma_prod_with_p'][p], *sigma_p_list_with_p)

        if num_p > 1:
            Eu_tilde_a = np.empty((num_s, num_p, num_a_max))
            dEu_tilde_a_dbeta = np.empty((num_s, num_p, num_a_max, num_s, num_p, num_a_max))
            dEu_tilde_a_dV = np.empty((num_s, num_p, num_a_max, num_s, num_p))
            for p in range(num_p):
                Eu_tilde_a[:, p] = np.einsum(self.einsum_eqs['Eu_tilde_a_J'][p], u_tilde[:, p], sigma_prod_with_p[:, p])
                dEu_tilde_a_dV[:, p] = np.einsum(self.einsum_eqs['dEu_tilde_a_dV'][p], self.T_J[3][:, p],
                                                 sigma_prod_with_p[:, p])
                T_temp = np.einsum('s...,s...->s...', u_tilde[:, p], sigma_prod_with_p[:, p])
                dEu_tilde_a_dbeta[:, p] = np.einsum(self.einsum_eqs['dEu_tilde_a_dbeta'][p], T_temp, self.T_J[2][:, p])

        else:
            Eu_tilde_a = u_tilde
            dEu_tilde_a_dbeta = np.zeros((num_s, num_p, num_a_max, num_s, num_p, num_a_max))
            dEu_tilde_a_dV = self.T_J[3]

        T_temp = np.einsum("sp...,s...->sp...", u_tilde, sigma_prod)
        dEu_tilde_dbeta = np.einsum(self.einsum_eqs['dEu_tilde_dbeta'], T_temp, self.T_J[4])

        dEu_tilde_dV = np.einsum('spa,spaSP->spSP', sigma, dEu_tilde_a_dV)

        # assemble J

        spa = num_s * num_p * num_a_max
        sp = num_s * num_p
        J = np.zeros((spa+sp, spa+sp+1))

        # dH_strat_dbeta
        J[0:spa, 0:spa] = (self.T_J[0] + np.einsum('spaSPA,SPA->spaSPA', self.T_J[1], sigma) + lambda_ *
                           np.einsum('spatqb,tqbSPA->spaSPA', -self.T_H[2], dEu_tilde_a_dbeta)
                           ).reshape((spa, spa))
        # dH_strat_dV
        J[0:spa, spa : spa+sp] = (lambda_ * np.einsum('spatqb,tqbSP->spaSP', -self.T_H[2], dEu_tilde_a_dV)
                                  ).reshape((spa, sp))
        # dH_strat_dlambda
        J[0:spa, spa+sp] = np.einsum('spatqb,tqb->spa', -self.T_H[2], Eu_tilde_a).reshape(spa)
        # dH_val_dbeta
        J[spa : spa+sp, 0:spa] = np.einsum('sptq,tqSPA->spSPA', -self.T_H[3], dEu_tilde_dbeta).reshape((sp, spa))
        # dH_val_dV
        J[spa : spa+sp, spa : spa+sp] = (self.T_J[5] + np.einsum('sptq,tqSP->spSP', -self.T_H[3], dEu_tilde_dV)
                                         ).reshape((sp, sp))
        # dH_val_dlambda = 0

        return J[self.J_mask]


# %% Cython implementation of QRE


class QRE_ct(QRE):
    """QRE homotopy: Cython implementation"""

    def __init__(self, game: SGame) -> None:
        super().__init__(game)

        # only import Cython module on class instantiation
        try:
            import pyximport
            pyximport.install(build_dir='./dsgamesolver/__build__/', build_in_temp=False, language_level=3,
                              setup_args={'include_dirs': [np.get_include()]})
            import dsgamesolver.qre_ct as qre_ct

        except ImportError:
            raise ImportError("Cython implementation of QRE homotopy could not be imported. ",
                              "Make sure your system has the relevant C compilers installed. ",
                              "For Windows, check https://wiki.python.org/moin/WindowsCompilers ",
                              "to find the right Microsoft Visual C++ compiler for your Python version. ",
                              "Standalone compilers are sufficient, there is no need to install Visual Studio. ",
                              "For Linux, make sure the Python package gxx_linux-64 is installed in your environment.")

        self.qre_ct = qre_ct

    def H(self, y: np.ndarray) -> np.ndarray:
        return self.qre_ct.H(y, self.game.payoffs, self.game.transitions, self.game.num_states, self.game.num_players,
                             self.game.nums_actions, self.game.num_actions_max, self.game.num_actions_total)

    def J(self, y: np.ndarray) -> np.ndarray:
        return self.qre_ct.J(y, self.game.payoffs, self.game.transitions, self.game.num_states, self.game.num_players,
                             self.game.nums_actions, self.game.num_actions_max, self.game.num_actions_total)


# %% experimental: Numpy implementation of QRE with Numba boost
# some trouble with custom classes...


# from numba import njit


# class QRE_nb(QRE_np):
#     """QRE homotopy: Numpy implementation with Numba boost."""

#     # def __init__(self, game: SGame) -> None:
#     #     super().__init__(game)

#     @njit
#     def H(self, y: np.ndarray) -> np.ndarray:
#         return super().H(y)

#     @njit
#     def J(self, y: np.ndarray, old: bool = True) -> np.ndarray:
#         return super().J(y, old)


# %% testing


if __name__ == '__main__':

    from tests.random_game import create_random_game
    game = SGame(*create_random_game())

    # numpy
    qre_np = QRE_np(game)
    qre_np.initialize()
    qre_np.solver.solve()

    y0 = qre_np.find_y0()
    """
    %timeit qre_np.H(y0)
    %timeit qre_np.J(y0)
    """

    # cython
    qre_ct = QRE_ct(game)
    qre_ct.initialize()
    qre_ct.solver.solve()

    """
    %timeit qre_ct.H(y0)
    %timeit qre_ct.J(y0)
    """

    # numba
    # qre_nb = QRE_nb(game)
    # qre_nb.initialize()
    # qre_nb.solver.solve()
    # """
    # %timeit qre_nb.H(y0)
    # %timeit qre_nb.J(y0)
    # """
