# TODO: is there a canonical way in which strategies, values are ordered in y?
# TODO: in which the equations are ordered in H?
# TODO (if so, this should be documented; and could be used for symmetry helpers etc.)
"""CHANGELOG (to be deleted after approval)


QRE:
----

ABC as constant outside of class.

Added type hints.



QRE_np:
-------



QRE_ct:
-------


"""

from typing import Tuple, Union

import numpy as np
from numba.experimental import jitclass

from dsgamesolver.sgame import SGame, SGameHomotopy
from dsgamesolver.homcont import HomCont

ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# %% general QRE class


class QRE(SGameHomotopy):
    """QRE homotopy: base class"""

    def __init__(self, game: SGame) -> None:
        super().__init__(game)

        # TODO: adjust parameters with scale of payoff matrix:

        self.tracking_parameters["normal"] = {
            "x_tol": 1e-7,
            "t_tol": 1e-7,
            "H_tol": 1e-7,
            "ds0": 0.01,
            "ds_infl": 1.2,
            "ds_defl": 0.5,
            "ds_min": 1e-9,
            "ds_max": 1000,
            "corr_steps_max": 20,
            "corr_dist_max": 0.3,
            "corr_ratio_max": 0.3,
            "detJ_change_max": 0.7,  # TODO: change format. was: 1.3
            "bifurc_angle_min": 175,
        }

        self.tracking_parameters["robust"] = {
            "x_tol": 1e-7,
            "t_tol": 1e-7,
            "H_tol": 1e-8,
            "ds0": 0.01,
            "ds_infl": 1.1,
            "ds_defl": 0.5,
            "ds_min": 1e-9,
            "ds_max": 1000,
            "corr_steps_max": 30,
            "corr_dist_max": 0.1,
            "corr_ratio_max": 0.1,
            "detJ_change_max": 0.7,  # TODO: change format. was: 1.1
            "bifurc_angle_min": 175,
        }

    def initialize(self, target_lambda: float = np.inf) -> None:
        self.y0 = self.find_y0()
        self.solver = HomCont(self.H, self.y0, self.J, t_target=target_lambda,
                              parameters=self.tracking_parameters["normal"],
                              x_transformer=self.x_transformer, store_path=True)
        # TODO: delete normalization
        # self.solver = HomCont(self.H, self.y0, self.J, parameters=self.tracking_parameters["normal"],
        #                       x_transformer=self.x_transformer, store_path=True)
        # self.solver.t_target = target_lambda * (self.game.payoff_max-self.game.payoff_min)

    def find_y0(self) -> np.ndarray:
        sigma = self.game.centroid_strategy()
        V = self.game.get_values(sigma)
        t = 0.0
        return self.sigma_V_t_to_y(sigma, V, t)

    def x_transformer(self, y: np.ndarray) -> np.ndarray:
        """Reverts logarithmization of strategies in vector y:
        Transformed values are needed to check whether sigmas have converged.
        """
        out = y.copy()
        out[0 : self.game.num_actions_total] = np.exp(out[0 : self.game.num_actions_total])
        return out

    def sigma_V_t_to_y(self, sigma: np.ndarray, V: np.ndarray, t: Union[float, int]) -> np.ndarray:
        beta_flat = np.log(self.game.flatten_strategies(sigma))
        V_flat = self.game.flatten_values(V)
        # TODO: delete normalization
        # V = self.game.flatten_values(self.game.get_values(sigma, normalized=True))
        return np.concatenate([beta_flat, V_flat, [t]])

    def y_to_sigma_V_t(self, y: np.ndarray, zeros: bool = False) -> Tuple[np.ndarray, np.ndarray, Union[float, int]]:
        sigma_V_t_flat = self.x_transformer(y)

        sigma = self.game.unflatten_strategies(sigma_V_t_flat[0:self.game.num_actions_total], zeros=zeros)
        V = self.game.unflatten_values(sigma_V_t_flat[self.game.num_actions_total:-1])
        # V = self.game.get_values(sigma)  # if values in y are transformed
        t = sigma_V_t_flat[-1]

        return sigma, V, t


# %% Numpy implementation of QRE


class QRE_np(QRE):
    """QRE homotopy: Numpy implementation"""

    def __init__(self, game: SGame) -> None:
        """prepares the following:
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
                for a in range(nums_a[s, p]):
                    H_mask.append(flat_index)
                    flat_index += 1
                flat_index += num_a_max - nums_a[s, p]
        for s in range(num_s):
            for p in range(num_p):
                H_mask.append(flat_index)
                flat_index += 1
        H_mask = np.array(H_mask, dtype=np.int64)

        J_mask = tuple(
            np.meshgrid(
                H_mask,
                np.append(H_mask, [num_s*num_p*num_a_max + num_s*num_p]),
                indexing="ij",
                sparse=True,
            )
        )

        self.H_mask = H_mask
        self.J_mask = J_mask

        # tensors to assemble H_qre
        T_H_qre_0 = np.zeros(shape=(num_s, num_p, num_a_max), dtype=np.float64)
        for s in range(num_s):
            for p in range(num_p):
                T_H_qre_0[s, p, 0] = 1

        T_H_qre_1 = np.zeros(shape=(num_s, num_p, num_a_max, num_s, num_p, num_a_max), dtype=np.float64)
        for s in range(num_s):
            for p in range(num_p):
                T_H_qre_1[s, p, 0, s, p, :] = -1

        T_H_qre_2 = np.zeros(shape=(num_s, num_p, num_a_max, num_s, num_p, num_a_max), dtype=np.float64)
        for s in range(num_s):
            for p in range(num_p):
                for a in range(1, nums_a[s, p]):
                    T_H_qre_2[s, p, a, s, p, a] = -1
                    T_H_qre_2[s, p, a, s, p, 0] = 1

        T_H_qre_3 = np.zeros(shape=(num_s, num_p, num_s, num_p), dtype=np.float64)
        for s in range(num_s):
            for p in range(num_p):
                T_H_qre_3[s, p, s, p] = -1

        self.T_H = {0: T_H_qre_0, 1: T_H_qre_1, 2: T_H_qre_2, 3: T_H_qre_3}

        # tensors to assemble J_qre
        T_J_qre_temp = np.zeros(shape=(num_s, num_p, num_a_max, num_s, num_p, num_a_max), dtype=np.float64)
        for s in range(num_s):
            for p in range(num_p):
                for a in range(nums_a[s, p]):
                    T_J_qre_temp[s, p, a, s, p, a] = 1

        T_J_qre_0 = np.einsum("spatqb,tqbSPA->spaSPA", T_H_qre_2, T_J_qre_temp)

        T_J_qre_1 = np.einsum("spatqb,tqbSPA->spaSPA", T_H_qre_1, T_J_qre_temp)

        T_J_qre_temp = np.zeros(shape=(num_s, num_p, num_s, num_p), dtype=np.float64)
        for s in range(num_s):
            for p in range(num_p):
                T_J_qre_temp[s, p, s, p] = 1

        T_J_qre_3 = np.einsum("sp...t,tpSP->sp...SP", self.game.transitions, T_J_qre_temp)

        T_J_qre_5 = np.einsum("sptq,tqSP->spSP", T_H_qre_3, T_J_qre_temp)

        T_J_qre_2 = np.zeros(shape=(num_s, num_p, *[num_a_max] * (num_p - 1), num_s, num_p, num_a_max),
                             dtype=np.float64)
        for s in range(num_s):
            for p in range(num_p):
                a_profiles_without_p = list(np.ndindex(tuple(nums_a[s, :p]) + tuple(nums_a[s, (p + 1) :])))
                for A in a_profiles_without_p:
                    for p_ in range(num_p):
                        if p_ != p:
                            a_ = A[p_] if p_ < p else A[p_ - 1]
                            T_J_qre_2[(s, p) + A + (s, p_, a_)] = 1

        T_J_qre_4 = np.zeros(shape=(num_s, num_p, *[num_a_max] * num_p, num_s, num_p, num_a_max), dtype=np.float64)
        for s in range(num_s):
            for p in range(num_p):
                a_profiles = list(np.ndindex(tuple(nums_a[s, :])))
                for A in a_profiles:
                    for p_ in range(num_p):
                        T_J_qre_4[(s, p) + A + (s, p_, A[p_])] = 1

        self.T_J = {
            0: T_J_qre_0,
            1: T_J_qre_1,
            2: T_J_qre_2,
            3: T_J_qre_3,
            4: T_J_qre_4,
            5: T_J_qre_5,
        }

        # equations to be used by einsum
        self.einsum_eqs = {
            "sigma_prod": "s" + ",s".join(ABC[0:num_p]) + "->s" + ABC[0:num_p],
            "sigma_prod_with_p": ["s" + ",s".join(ABC[0:num_p]) + "->s" + ABC[0:num_p] for p in range(num_p)],
            "Eu_tilde_a_H": ["s" + ABC[0:num_p] + ",s" + ",s".join([ABC[p_] for p_ in range(num_p) if p_ != p])
                             + "->s" + ABC[p] for p in range(num_p)],
            "Eu_tilde_a_J": ["s" + ABC[0:num_p] + ",s" + ABC[0:num_p] + "->s" + ABC[p] for p in range(num_p)],
            "dEu_tilde_a_dbeta": ["s" + ABC[0:num_p] + ",s" + "".join([ABC[p_] for p_ in range(num_p) if p_ != p])
                                  + "tqb->s" + ABC[p] + "tqb" for p in range(num_p)],
            "dEu_tilde_a_dV": ["s" + ABC[0:num_p] + "tp,s" + ABC[0:num_p] + "->s" + ABC[p] + "tp"
                               for p in range(num_p)],
            "dEu_tilde_dbeta": "sp" + ABC[0:num_p] + ",sp" + ABC[0:num_p] + "tqb->sptqb",
        }

    def H(self, y: np.ndarray) -> np.ndarray:
        """Homotopy function."""

        num_s, num_p = self.game.num_states, self.game.num_players
        num_a_max, num_a_tot = self.game.num_actions_max, self.game.num_actions_total

        # extract log-strategies beta, state values V and homotopy parameter gamma from y
        beta = self.game.unflatten_strategies(y[:num_a_tot], zeros=True)
        sigma, V, gamma = self.y_to_sigma_V_t(y, zeros=True)

        # generate building blocks of H

        sigma_p_list = [sigma[:, p, :] for p in range(num_p)]
        # TODO: delete normalization
        # u_tilde = self.game.payoffs_normalized + np.einsum("sp...S,Sp->sp...", self.game.transitions, V)
        u_tilde = self.game.payoffs + np.einsum("sp...S,Sp->sp...", self.game.transitions, V)

        if num_p > 1:
            Eu_tilde_a = np.empty((num_s, num_p, num_a_max))
            for p in range(num_p):
                Eu_tilde_a[:, p] = np.einsum(self.einsum_eqs['Eu_tilde_a_H'][p], u_tilde[:, p],
                                             *(sigma_p_list[:p] + sigma_p_list[(p+1):]))
        else:
            Eu_tilde_a = u_tilde

        Eu_tilde = np.einsum("spa,spa->sp", sigma, Eu_tilde_a)

        # assemble H

        H_strat = (self.T_H[0] + np.einsum("spaSPA,SPA->spa", self.T_H[1], sigma)
                   + np.einsum("spaSPA,SPA->spa", self.T_H[2], beta)
                   + gamma * np.einsum("spaSPA,SPA->spa", -self.T_H[2], Eu_tilde_a))

        H_val = np.einsum("spSP,SP->sp", self.T_H[3], V) + np.einsum("spSP,SP->sp", -self.T_H[3], Eu_tilde)

        return np.append(H_strat.ravel(), H_val.ravel())[self.H_mask]

    def J(self, y: np.ndarray, old: bool = True) -> np.ndarray:
        """Jacobian matrix of homotopy function."""

        num_s, num_p, num_a_max = self.game.num_states, self.game.num_players, self.game.num_actions_max

        # extract log-strategies beta, state values V and homotopy parameter gamma from y
        sigma, V, gamma = self.y_to_sigma_V_t(y, zeros=True)

        # generate building blocks of J
        sigma_p_list = [sigma[:, p, :] for p in range(num_p)]
        # TODO: delete normalization
        # u_tilde = self.game.payoffs_normalized + np.einsum("sp...S,Sp->sp...", self.game.transitions, V)
        u_tilde = self.game.payoffs + np.einsum("sp...S,Sp->sp...", self.game.transitions, V)

        sigma_prod = np.einsum(self.einsum_eqs["sigma_prod"], *sigma_p_list)

        # TODO: replace this pattern:
        # sigma_prod_with_p = []
        # for p in range(num_p):
        #     sigma_p_list_with_p = sigma_p_list[:p] + [np.ones_like(sigma[:, p, :])] \
        #                           + sigma_p_list[(p + 1):]
        #     sigma_prod_with_p.append(np.einsum(self.einsum_eqs['sigma_prod_with_p'][p], *sigma_p_list_with_p))
        # sigma_prod_with_p = np.stack(sigma_prod_with_p, axis=1)
        # TODO: with this:
        sigma_prod_with_p = np.empty((num_s, num_p, *[num_a_max]*num_p))
        for p in range(num_p):
            sigma_p_list_with_p = sigma_p_list[:p] + [np.ones_like(sigma[:, p, :])] \
                                  + sigma_p_list[(p + 1):]
            sigma_prod_with_p[:, p] = np.einsum(self.einsum_eqs['sigma_prod_with_p'][p], *sigma_p_list_with_p)
        # TODO (done within qre)

        if num_p > 1:
            Eu_tilde_a = np.empty((num_s, num_p, num_a_max))
            dEu_tilde_a_dbeta = np.empty((num_s, num_p, num_a_max, num_s, num_p, num_a_max))
            dEu_tilde_a_dV = np.empty((num_s, num_p, num_a_max, num_s, num_p))
            for p in range(num_p):
                Eu_tilde_a[:, p] = np.einsum(self.einsum_eqs['Eu_tilde_a_J'][p],
                                             u_tilde[:, p], sigma_prod_with_p[:, p])
                dEu_tilde_a_dV[:, p] = np.einsum(self.einsum_eqs['dEu_tilde_a_dV'][p],
                                                 self.T_J[3][:, p], sigma_prod_with_p[:, p])
                T_temp = np.einsum('s...,s...->s...', u_tilde[:, p], sigma_prod_with_p[:, p])
                dEu_tilde_a_dbeta[:, p] = np.einsum(self.einsum_eqs['dEu_tilde_a_dbeta'][p],
                                                    T_temp, self.T_J[2][:, p])

        else:
            Eu_tilde_a = u_tilde
            dEu_tilde_a_dbeta = np.zeros(shape=(num_s, num_p, num_a_max, num_s, num_p, num_a_max))
            dEu_tilde_a_dV = self.T_J[3]

        T_temp = np.einsum("sp...,s...->sp...", u_tilde, sigma_prod)
        dEu_tilde_dbeta = np.einsum(
            self.einsum_eqs["dEu_tilde_dbeta"], T_temp, self.T_J[4]
        )

        dEu_tilde_dV = np.einsum("spa,spaSP->spSP", sigma, dEu_tilde_a_dV)

        # TODO: tried "new" below - seems to bring no improvement.
        if old:
            # assemble J
            dH_strat_dbeta = self.T_J[0] + np.einsum('spaSPA,SPA->spaSPA', self.T_J[1], sigma) \
                             + gamma * np.einsum('spatqb,tqbSPA->spaSPA', -self.T_H[2], dEu_tilde_a_dbeta)
            dH_strat_dV = gamma * np.einsum('spatqb,tqbSP->spaSP', -self.T_H[2], dEu_tilde_a_dV)
            dH_strat_dlambda = np.einsum('spatqb,tqb->spa', -self.T_H[2], Eu_tilde_a)
            dH_val_dbeta = np.einsum('sptq,tqSPA->spSPA', -self.T_H[3], dEu_tilde_dbeta)
            dH_val_dV = self.T_J[5] + np.einsum('sptq,tqSP->spSP', -self.T_H[3], dEu_tilde_dV)
            dH_val_dlambda = np.zeros(shape=(num_s, num_p), dtype=np.float64)

            spa = num_s * num_p * num_a_max
            sp = num_s * num_p

            return np.concatenate([
                np.concatenate([
                    dH_strat_dbeta.reshape((spa, spa)),
                    dH_strat_dV.reshape((spa, sp)),
                    dH_strat_dlambda.reshape((spa, 1))
                ], axis=1),
                np.concatenate([
                    dH_val_dbeta.reshape((sp, spa)),
                    dH_val_dV.reshape((sp, sp)),
                    dH_val_dlambda.reshape((sp, 1))
                ], axis=1)
            ], axis=0)[self.J_mask]

        if not old:
            # assemble J
            spa = num_s * num_p * num_a_max
            sp = num_s * num_p
            J = np.empty((spa+sp, spa+sp+1))

            # dH_strat_dbeta
            J[0:spa, 0:spa] = (self.T_J[0] + np.einsum('spaSPA,SPA->spaSPA', self.T_J[1], sigma)
                               + gamma * np.einsum('spatqb,tqbSPA->spaSPA', -self.T_H[2], dEu_tilde_a_dbeta)
                               ).reshape((spa, spa))
            # dH_strat_dV
            J[0:spa, spa:spa+sp] = gamma * np.einsum('spatqb,tqbSP->spaSP', -self.T_H[2],
                                                     dEu_tilde_a_dV).reshape((spa, sp))
            # dH_strat_dlambda
            J[0:spa, spa+sp] = np.einsum('spatqb,tqb->spa', -self.T_H[2], Eu_tilde_a).reshape((spa))
            # dH_val_dbeta
            J[spa:spa+sp, 0:spa] = np.einsum('sptq,tqSPA->spSPA', -self.T_H[3], dEu_tilde_dbeta).reshape((sp, spa))
            # dH_val_dV
            J[spa:spa+sp, spa:spa+sp] = (self.T_J[5] + np.einsum('sptq,tqSP->spSP', -self.T_H[3],
                                                                 dEu_tilde_dV)).reshape((sp, sp))
            # dH_val_dlambda
            J[spa:spa+sp, spa+sp] = 0

            return J[self.J_mask]


# %% Numpy implementation of QRE with Numba boost


@jitclass
class QRE_np_numba(QRE_np):
    """QRE homotopy: Numpy implementation with Numba boost."""

    # def __init__(self, game: SGame) -> None:
    #     super().__init__(game)

    def H(self, y: np.ndarray) -> np.ndarray:
        return super().H(y)

    def J(self, y: np.ndarray, old: bool = True) -> np.ndarray:
        return super().J(y, old)


# %% Cython implementation of QRE


class QRE_ct(QRE):
    """QRE homotopy: Cython implementation"""

    def __init__(self, game: SGame):
        super().__init__(game)
        # temporary:
        game = game

    def H(self, y):
        return QRE_np(self.game).H(y)

    def J(self, y):
        return QRE_np(self.game).J(y)

    # TODO: all


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
            pass
        else:
            # TODO: how are etas specified / should they be checked?
            self.etas = etas

    def find_y0(self):
        # TODO
        y0 = None
        self.y0 = y0
