# TODO: is there a canonical way in which strategies, values are ordered in y?
# TODO: in which the equations are ordered in H?
# TODO (if so, this should be documented; and could be used for symmetry helpers etc.)
import numpy as np
import homcont.sgame as sgame
import string
import warnings

import homcont.homclass


class sgameHomotopy():
    """ General homotopy class for some sgame."""

    def __init__(self, game: sgame.sGame):
        self.game = game
        self.y0 = None
        self.tracking_parameters = {}

        self.solver = None

    def initialize(self):
        """Any steps in preparation to start solver:
            - set priors, weights etc. if needed
            - set starting point y0
            - prepare symmetry helpers
                + (make sure priors and other parameters are in accordance with symmetries)

            - set up homCont to solve the game.
        """

    def find_y0(self):
        """calculate starting point y0"""
        pass

    def H(self, y):
        """Homotopy function evaluated at y."""
        pass

    def J(self, y):
        """ Jacobian of homotopy function evaluated at y."""
        pass

    def H_reduced(self, y):
        """H reduced by exploiting symmetries."""
        pass

    def J_reduced(self, y):
        """J reduced by exploiting symmetries."""
        pass

    def x_transformer(self, y):
        """Transform the relevant part a vector y to strategies.
           Needed only if the homotopy uses transformed strategies (e.g. logarithmized),
           but one would like to use convergence checks that operate on untransformed strategies.
           (Example: qre, which uses beta=log(sigma), but convergence criterion is based on sigma).
           If not needed: can simply pass None to homCont
        """
        pass

    def y_to_sigma_V(self):
        """ Translate a y-vector to arrays representing sigma / V """
        pass

    def sigma_V_to_y(self):
        """ Translate arrays representing sigma and V to a vector  y """
        pass


class QRE(sgameHomotopy):
    """QRE homotopy: base class"""

    def __init__(self, game: sgame.sGame):
        super().__init__(game)

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
            "bifurc_angle_min": 175
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
            "bifurc_angle_min": 175
        }

        self.find_y0()

    def solver_setup(self, target_lambda=np.infty):
        self.solver = homcont.homclass.HomCont(self.H, self.y0, self.J, parameters=self.tracking_parameters["normal"],
                                               x_transformer=self.x_transformer)
        self.solver.t_target = target_lambda * (self.game.u_max - self.game.u_min)

    def find_y0(self):
        num_s, num_p, nums_a = self.game.num_states, self.game.num_players, self.game.nums_actions
        strategy_axes = tuple(np.arange(1, 1 + num_p))

        # strategies: players randomize uniformly; beta is logarithmized.
        beta = np.concatenate(
            [np.log(np.ones(nums_a[s, p]) / nums_a[s, p]) for s in range(num_s) for p in range(num_p)])

        # state values: solve linear system of equations for each player
        V = np.nan * np.ones(num_s * num_p, dtype=np.float64)
        for p in range(num_p):
            A = np.identity(num_s) - np.nanmean(self.game.transitionArray_withNaN[:, p], axis=strategy_axes)
            b = np.nanmean(self.game.u_norm_with_nan[:, p], axis=strategy_axes)
            mu_p = np.linalg.solve(A, b)
            for s in range(num_s):
                V[s * num_p + p] = mu_p[s]

        self.y0 = np.concatenate([beta, V, [0.0]])

        if np.isnan(self.y0).any():
            warnings.warn('Encountered a problem when setting starting point y0: '
                          'Linear system of equations could not be solved.')

    def x_transformer(self, y):
        """Reverts logarithmization of strategies in vector y:
           Transformed values are needed to check whether sigmas have converged.
        """
        out = y.copy()
        out[:self.game.num_actions_total] = np.exp(out[:self.game.num_actions_total])
        return out


class QRE_np(QRE):

    def __init__(self, game: sgame.sGame):
        """prepares the following for QRE homotopy (numpy version):
            - T_y2beta
            - H_mask, J_mask
            - T_H, T_J
            - einsum_eqs
        """

        super().__init__(game)

        num_s, num_p, nums_a = self.game.num_states, self.game.num_players, self.game.nums_actions
        num_a_max = nums_a.max()

        # array to extract beta[s,p,a] from y[:num_a_tot]
        T_y2beta = np.zeros(shape=(num_s, num_p, num_a_max, nums_a.sum()), dtype=np.float64)
        flat_index = 0
        for s in range(num_s):
            for p in range(num_p):
                for a in range(nums_a[s, p]):
                    T_y2beta[s, p, a, flat_index] = 1
                    flat_index += 1
                for a in range(nums_a[s, p], num_a_max):
                    T_y2beta[s, p, a] = np.nan
        self.T_y2beta = T_y2beta

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
            np.meshgrid(H_mask, np.append(H_mask, [num_s * num_p * num_a_max + num_s * num_p]),
                        indexing='ij', sparse=True))

        self.H_mask = H_mask
        self.J_mask = J_mask

        # arrays to assemble H_qre
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

        self.T_H = {
            0: T_H_qre_0,
            1: T_H_qre_1,
            2: T_H_qre_2,
            3: T_H_qre_3
        }

        # arrays to assemble J_qre
        T_J_qre_temp = np.zeros(shape=(num_s, num_p, num_a_max, num_s, num_p, num_a_max), dtype=np.float64)
        for s in range(num_s):
            for p in range(num_p):
                for a in range(nums_a[s, p]):
                    T_J_qre_temp[s, p, a, s, p, a] = 1

        T_J_qre_0 = np.einsum('spatqb,tqbSPA->spaSPA', T_H_qre_2, T_J_qre_temp)

        T_J_qre_1 = np.einsum('spatqb,tqbSPA->spaSPA', T_H_qre_1, T_J_qre_temp)

        T_J_qre_temp = np.zeros(shape=(num_s, num_p, num_s, num_p), dtype=np.float64)
        for s in range(num_s):
            for p in range(num_p):
                T_J_qre_temp[s, p, s, p] = 1

        T_J_qre_3 = np.einsum('sp...t,tpSP->sp...SP', self.game.phi, T_J_qre_temp)

        T_J_qre_5 = np.einsum('sptq,tqSP->spSP', T_H_qre_3, T_J_qre_temp)

        T_J_qre_2 = np.zeros(shape=(num_s, num_p, *[num_a_max] * (num_p - 1), num_s, num_p, num_a_max),
                             dtype=np.float64)
        for s in range(num_s):
            for p in range(num_p):
                a_profiles_without_p = list(np.ndindex(tuple(nums_a[s, :p]) + tuple(nums_a[s, (p + 1):])))
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
            5: T_J_qre_5
        }

        # defining equations to be used by einsum
        ABC = string.ascii_uppercase
        self.einsum_eqs = {
            'sigma_prod': 's' + ',s'.join(ABC[0:num_p]) + '->s' + ABC[0:num_p],
            'sigma_prod_with_p': ['s' + ',s'.join(ABC[0:num_p]) + '->s' + ABC[0:num_p] for p in range(num_p)],
            'Eu_tilde_a_H': ['s' + ABC[0:num_p] + ',s' + ',s'.join([ABC[p_] for p_ in range(num_p) if p_ != p])
                             + '->s' + ABC[p] for p in range(num_p)],
            'Eu_tilde_a_J': ['s' + ABC[0:num_p] + ',s' + ABC[0:num_p] + '->s' + ABC[p] for p in range(num_p)],
            'dEu_tilde_a_dbeta': ['s' + ABC[0:num_p] + ',s' + ''.join([ABC[p_] for p_ in range(num_p) if p_ != p]) +
                                  'tqb->s' + ABC[p] + 'tqb' for p in range(num_p)],
            'dEu_tilde_a_dV': ['s' + ABC[0:num_p] + 'tp,s' + ABC[0:num_p] + '->s' + ABC[p] + 'tp' for p in
                               range(num_p)],
            'dEu_tilde_dbeta': 'sp' + ABC[0:num_p] + ',sp' + ABC[0:num_p] + 'tqb->sptqb'
        }

    def H(self, y):

        num_s, num_p, num_a_tot = self.game.num_states, self.game.num_players, self.game.num_actions_total

        # extract log-strategies beta, state values V and homotopy parameter gamma from y
        beta = np.einsum('spaN,N->spa', self.T_y2beta, y[:num_a_tot])
        V = y[num_a_tot:-1].reshape((num_s, num_p))
        gamma = y[-1]

        # generate building blocks of H
        sigma = np.exp(beta)
        sigma[np.isnan(sigma)] = 0
        beta[np.isnan(beta)] = 0
        # TODO: original also defined beta_withnan - any reason why?

        sigma_p_list = [sigma[:, p, :] for p in range(num_p)]
        u_tilde = self.game.u_norm + np.einsum('sp...S,Sp->sp...', self.game.phi, V)

        if num_p > 1:
            Eu_tilde_a = []
            for p in range(num_p):
                Eu_tilde_a.append(
                    np.einsum(self.einsum_eqs['Eu_tilde_a_H'][p], u_tilde[:, p],
                              *(sigma_p_list[:p] + sigma_p_list[(p + 1):])))
            Eu_tilde_a = np.stack(Eu_tilde_a, axis=1)
        else:
            Eu_tilde_a = u_tilde

        Eu_tilde = np.einsum('spa,spa->sp', sigma, Eu_tilde_a)

        # assemble H
        H_strat = self.T_H[0] + np.einsum('spaSPA,SPA->spa', self.T_H[1], sigma) \
                  + np.einsum('spaSPA,SPA->spa', self.T_H[2], beta) \
                  + gamma * np.einsum('spaSPA,SPA->spa', -self.T_H[2], Eu_tilde_a)

        H_val = np.einsum('spSP,SP->sp', self.T_H[3], V) + np.einsum('spSP,SP->sp', -self.T_H[3], Eu_tilde)

        H = np.append(H_strat.ravel(), H_val.ravel())[self.H_mask]

        return H

    def J(self, y):

        num_s = self.game.num_states
        num_p = self.game.num_players
        num_a_max = self.game.num_actions_max
        num_a_tot = self.game.num_actions_total

        # extract log-strategies beta, state values V and homotopy parameter gamma from y
        beta = np.einsum('spaN,N->spa', self.T_y2beta, y[:num_a_tot])
        V = y[num_a_tot:-1].reshape((num_s, num_p))
        gamma = y[-1]

        # generate building blocks of J
        sigma = np.exp(beta)
        sigma[np.isnan(sigma)] = 0
        beta[np.isnan(beta)] = 0
        # TODO: original also defined beta_withnan - any reason why? seems that it was not used anywhere

        sigma_p_list = [sigma[:, p, :] for p in range(num_p)]
        u_tilde = self.game.u_norm + np.einsum('sp...S,Sp->sp...', self.game.phi, V)

        sigma_prod = np.einsum(self.einsum_eqs['sigma_prod'], *sigma_p_list)

        sigma_prod_with_p = []
        for p in range(num_p):
            sigma_p_list_with_p = sigma_p_list[:p] + [np.ones(shape=sigma[:, p, :].shape, dtype=np.float64)] \
                                  + sigma_p_list[(p + 1):]
            sigma_prod_with_p.append(np.einsum(self.einsum_eqs['sigma_prod_with_p'][p], *sigma_p_list_with_p))
        sigma_prod_with_p = np.stack(sigma_prod_with_p, axis=1)

        if num_p > 1:
            Eu_tilde_a = []
            dEu_tilde_a_dbeta = []
            dEu_tilde_a_dV = []

            for p in range(num_p):
                Eu_tilde_a.append(np.einsum(self.einsum_eqs['Eu_tilde_a_J'][p], u_tilde[:, p], sigma_prod_with_p[:, p]))
                dEu_tilde_a_dV.append(
                    np.einsum(self.einsum_eqs['dEu_tilde_a_dV'][p], self.T_J[3][:, p], sigma_prod_with_p[:, p]))

                T_temp = np.einsum('s...,s...->s...', u_tilde[:, p], sigma_prod_with_p[:, p])
                dEu_tilde_a_dbeta.append(np.einsum(self.einsum_eqs['dEu_tilde_a_dbeta'][p], T_temp, self.T_J[2][:, p]))

            Eu_tilde_a = np.stack(Eu_tilde_a, axis=1)
            dEu_tilde_a_dbeta = np.stack(dEu_tilde_a_dbeta, axis=1)
            dEu_tilde_a_dV = np.stack(dEu_tilde_a_dV, axis=1)

        else:
            Eu_tilde_a = u_tilde
            dEu_tilde_a_dbeta = np.zeros(shape=(num_s, num_p, num_a_max, num_s, num_p, num_a_max), dtype=np.float64)
            dEu_tilde_a_dV = self.T_J[3]

        T_temp = np.einsum('sp...,s...->sp...', u_tilde, sigma_prod)
        dEu_tilde_dbeta = np.einsum(self.einsum_eqs['dEu_tilde_dbeta'], T_temp, self.T_J[4])

        dEu_tilde_dV = np.einsum('spa,spaSP->spSP', sigma, dEu_tilde_a_dV)

        # assemble J
        dH_strat_dbeta = self.T_J[0] + np.einsum('spaSPA,SPA->spaSPA', self.T_J[1], sigma) \
                         + gamma * np.einsum('spatqb,tqbSPA->spaSPA', -self.T_H[2], dEu_tilde_a_dbeta)
        dH_strat_dV = gamma * np.einsum('spatqb,tqbSP->spaSP', -self.T_H[2], dEu_tilde_a_dV)
        dH_strat_dlambda = np.einsum('spatqb,tqb->spa', -self.T_H[2], Eu_tilde_a)
        dH_val_dbeta = np.einsum('sptq,tqSPA->spSPA', -self.T_H[3], dEu_tilde_dbeta)
        dH_val_dV = self.T_J[5] + np.einsum('sptq,tqSP->spSP', -self.T_H[3], dEu_tilde_dV)
        dH_val_dlambda = np.zeros(shape=(num_s, num_p), dtype=np.float64)

        J = np.concatenate([
            np.concatenate([
                dH_strat_dbeta.reshape((num_s * num_p * num_a_max, num_s * num_p * num_a_max)),
                dH_strat_dV.reshape((num_s * num_p * num_a_max, num_s * num_p)),
                dH_strat_dlambda.reshape((num_s * num_p * num_a_max, 1))
            ], axis=1),
            np.concatenate([
                dH_val_dbeta.reshape((num_s * num_p, num_s * num_p * num_a_max)),
                dH_val_dV.reshape((num_s * num_p, num_s * num_p)),
                dH_val_dlambda.reshape((num_s * num_p, 1))
            ], axis=1)
        ], axis=0)[self.J_mask]

        return J


class QRE_ct(QRE):
    def __init__(self, game: sgame.sGame):
        super().__init__(game)

    # TODO: all


class Tracing(sgameHomotopy):
    def __init__(self, game: sgame.sGame, priors="centroid", etas=None, nu=1.0):
        super().__init__(game)

        if priors == "random":
            priors = np.empty(self.game.num_actions_total, dtype=np.float64)
            idx = 0
            for s in range(self.game.num_states):
                for p in range(self.game.num_players):
                    sigma = np.random.exponential(scale=1, size=self.game.nums_actions[s, p])
                    sigma = sigma / sigma.sum()
                    etas[idx:idx + self.game.nums_actions[s, p]] = sigma
                    idx += self.game.nums_actions[s, p]
            self.priors = priors
        elif priors == "centroid":
            priors = np.empty(self.game.num_actions_total, dtype=np.float64)
            idx = 0
            for s in range(self.game.num_states):
                for p in range(self.game.num_players):
                    etas[idx:idx + self.game.nums_actions[s, p]] = 1 / self.game.nums_actions[s, p]
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
