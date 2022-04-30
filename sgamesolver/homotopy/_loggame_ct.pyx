"""Cython implementation of LogGame homotopy."""

cimport cython
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def H(np.ndarray[np.float64_t] y, np.ndarray[np.float64_t] u, np.ndarray[np.float64_t] phi,
      np.ndarray[np.float64_t] delta, np.ndarray[np.float64_t, ndim=3] nu, int num_s, int num_p,
      np.ndarray[np.int32_t, ndim=2] nums_a, int num_a_max, int num_a_tot, bint parallel):
    """Homotopy function.

    H(y) = [  H_val[s,i,a]  ]
           [  H_strat[s,i]  ]
    with
    y = [ beta[s,i,a],  V[s,i],  t ]
    """

    cdef:
        np.ndarray[np.float64_t, ndim=1] out_ = np.zeros(num_a_tot + num_s * num_p)
        np.ndarray[np.float64_t, ndim=3] beta = np.ones((num_s, num_p, num_a_max))
        np.ndarray[np.float64_t, ndim=3] sigma
        np.ndarray[np.float64_t, ndim=3] sigma_inv
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot: num_a_tot + num_s * num_p].reshape(num_s, num_p)
        double t = y[num_a_tot + num_s * num_p]

        np.ndarray[np.float64_t, ndim=1] u_tilde_ev_ravel
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev

        int state, player, action
        int flat_index = 0
        double nu_beta_sum

    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state, player]):
                beta[state, player, action] = y[flat_index]
                flat_index += 1
    sigma = np.exp(beta)
    sigma_inv = np.exp(-beta)

    u_tilde_ev_ravel = u_tilde(u, phi, delta, V, num_s, num_p, nums_a, num_a_max, parallel)
    u_tilde_sia_ev = u_tilde_sia(u_tilde_ev_ravel, sigma, num_s, num_p, nums_a, num_a_max, parallel)

    flat_index = 0
    for state in range(num_s):
        for player in range(num_p):
            nu_beta_sum = 0
            for action in range(nums_a[state, player]):
                nu_beta_sum += nu[state, player, action] * (beta[state, player, action] - 1)

            for action in range(nums_a[state, player]):
                out_[flat_index] = (- V[state, player] + t * u_tilde_sia_ev[state, player, action] + (1 - t)
                                    * (nu[state, player, action] * sigma_inv[state, player, action] + nu_beta_sum))
                flat_index += 1

    for state in range(num_s):
        for player in range(num_p):
            out_[flat_index] = -1
            for action in range(nums_a[state, player]):
                out_[flat_index] += sigma[state, player, action]
            flat_index += 1

    return out_

# %% Jacobian matrix

@cython.boundscheck(False)
@cython.wraparound(False)
def J(np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t] u, np.ndarray[np.float64_t] phi,
      np.ndarray[np.float64_t] delta, np.ndarray[np.float64_t, ndim=3] nu, int num_s, int num_p,
      np.ndarray[np.int32_t, ndim=2] nums_a, int num_a_max, int num_a_tot, bint parallel):
    """Jacobian matrix.

    J(y) = [  d_H_val[s,i]     / d_beta[s',i',a'],  d_H_val[s,i]     / d_V[s',i'],  d_H_val[s,i]     / d_t  ]
           [  d_H_strat[s,i,a] / d_beta[s',i',a'],  d_H_strat[s,i,a] / d_V[s',i'],  d_H_strat[s,i,a] / d_t  ]
    with
    y = [ beta[s,i,a],  V[s,i],  t ]
    """

    cdef:
        np.ndarray[np.float64_t, ndim=2] out_ = np.zeros((num_a_tot + num_s * num_p, num_a_tot + num_s * num_p + 1))
        np.ndarray[np.float64_t, ndim=3] beta = np.ones((num_s, num_p, num_a_max))
        np.ndarray[np.float64_t, ndim=3] sigma
        np.ndarray[np.float64_t, ndim=3] sigma_inv
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot: num_a_tot + num_s * num_p].reshape(num_s, num_p)
        double t = y[num_a_tot + num_s * num_p]

        np.ndarray[np.float64_t, ndim=1] u_tilde_ev_ravel
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev
        np.ndarray[np.float64_t, ndim=5] u_tilde_sijab_ev
        np.ndarray[np.float64_t, ndim=4] phi_siat_ev
        int state, player, action
        int flat_index = 0
        double nu_beta_sum
        int row_state, row_player, row_action
        int col_state, col_player, col_action
        int row_index, col_index, col_index_init

    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state, player]):
                beta[state, player, action] = y[flat_index]
                flat_index += 1
    sigma = np.exp(beta)
    sigma_inv = np.exp(-beta)

    u_tilde_ev_ravel = u_tilde(u, phi, delta, V, num_s, num_p, nums_a, num_a_max, parallel)
    u_tilde_sia_ev = u_tilde_sia(u_tilde_ev_ravel, sigma, num_s, num_p, nums_a, num_a_max, parallel)
    u_tilde_sijab_ev = u_tilde_sijab(u_tilde_ev_ravel, sigma, num_s, num_p, nums_a, num_a_max, parallel)
    phi_siat_ev = phi_siat(phi, delta, sigma, num_s, num_p, nums_a, num_a_max, parallel)

    # first block: rows with d_H_val[s,i,a]
    row_index = 0
    col_index_init = 0
    for row_state in range(num_s):
        for row_player in range(num_p):
            nu_beta_sum = 0
            for row_action in range(nums_a[row_state, row_player]):
                nu_beta_sum += nu[row_state, row_player, row_action] * (beta[row_state, row_player, row_action] - 1)

            for row_action in range(nums_a[row_state, row_player]):

                # derivatives w.r.t. beta[s',i',a']
                # entries with s' != s are 0, thus no looping over s'
                col_index = col_index_init
                for col_player in range(num_p):
                    for col_action in range(nums_a[row_state, col_player]):

                        # diagonal blocks: derivatives w.r.t. beta[s,i,a']
                        # (own actions in same state)
                        if row_player == col_player:
                            if row_action == col_action:
                                out_[row_index, col_index] = ((1 - t) * nu[row_state, row_player, row_action]
                                                              * (1 - sigma_inv[row_state, row_player, row_action]))
                            else:
                                out_[row_index, col_index] = (1 - t) * nu[row_state, row_player, col_action]

                        # off-diagonal sub-blocks: derivatives w.r.t. beta[s,i',a']
                        # (other players' actions in same state)
                        else:
                            out_[row_index, col_index] = (
                                    t * sigma[row_state, col_player, col_action]
                                    * u_tilde_sijab_ev[row_state, row_player, col_player, row_action, col_action]
                            )

                        col_index += 1

                # derivatives w.r.t. V[s',i']
                col_index = num_a_tot + row_player
                for col_state in range(num_s):
                    if row_state == col_state:
                        out_[row_index, col_index] = t * phi_siat_ev[row_state, row_player, row_action, col_state] - 1
                    else:
                        out_[row_index, col_index] = t * phi_siat_ev[row_state, row_player, row_action, col_state]
                    col_index += num_p
                    if col_state == num_s - 1:
                        col_index -= row_player

                # derivative w.r.t. t
                out_[row_index, col_index] = (u_tilde_sia_ev[row_state, row_player, row_action]
                                              - (nu[row_state, row_player, row_action]
                                                 * sigma_inv[row_state, row_player, row_action] + nu_beta_sum))

                row_index += 1

        for row_player in range(num_p):
            col_index_init += nums_a[row_state, row_player]

    # second block: rows with d_H_strat[s,i]
    row_index = num_a_tot
    col_index_init = 0
    for row_state in range(num_s):
        for row_player in range(num_p):

            # derivatives w.r.t. beta[s',i',a']
            # entries with s' != s and i' != i are 0, thus no looping over s' and i'
            col_index = col_index_init
            for col_player in range(row_player):
                col_index += nums_a[row_state, col_player]

            for col_action in range(nums_a[row_state, row_player]):
                out_[row_index, col_index] = sigma[row_state, row_player, col_action]
                col_index += 1

            # derivatives w.r.t. V[s',i'] = 0

            # derivative w.r.t. t = 0

            row_index += 1

        for row_player in range(num_p):
            col_index_init += nums_a[row_state, row_player]

    return out_


# include function definitions for u_tilde, u_tilde_sia, u_tilde_sijab, phi_sia, arrays_equal
include "_shared_ct.pyx"

