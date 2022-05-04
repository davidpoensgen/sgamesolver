"""Cython implementation of QRE homotopy."""

cimport cython
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
np.import_array()

from _shared_ct cimport u_tilde, u_tilde_sia, phi_siat, arrays_equal

@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def H(np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] u, np.ndarray[np.float64_t, ndim=1] phi,
      np.ndarray[np.float64_t, ndim=1] delta, int [:,::1] nums_a, QreCache cache, bint parallel):
    """Homotopy function.

    H(y) = [  H_strat[s,i,a]  ]
           [  H_val[s,i]      ]
    with
    y = [ beta[s,i,a],  V[s,i],  lambda ]
    """

    cdef:
        int num_s = nums_a.shape[0]
        int num_p = nums_a.shape[1]
        int num_a_max = np.max(nums_a)
        int num_a_tot = np.sum(nums_a)
        np.ndarray[np.float64_t, ndim=1] out_ = np.zeros(num_a_tot + num_s*num_p)
        np.ndarray[np.float64_t, ndim=3] beta = np.zeros((num_s, num_p, num_a_max))
        np.ndarray[np.float64_t, ndim=3] sigma
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot : num_a_tot + num_s*num_p].reshape((num_s, num_p))
        double lambda_ = y[num_a_tot + num_s*num_p]

        np.ndarray[np.float64_t, ndim=1] u_tilde_ev_ravel
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev

        int state, player, action, a
        int flat_index = 0

    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state, player]):
                beta[state, player, action] = y[flat_index]
                flat_index += 1
    sigma = np.exp(beta)

    if cache is None:
        # cache disabled -> always calculate all intermediate variables.
        u_tilde_ev_ravel = u_tilde(u, phi, delta, V, num_s, num_p, nums_a, num_a_max, parallel)
        u_tilde_sia_ev = u_tilde_sia(u_tilde_ev_ravel, sigma, num_s, num_p, nums_a, num_a_max, parallel)
    elif arrays_equal(y, cache.y):
        # intermediate values already in cache. for H, u_tilde_ev_ravel is not needed.
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)
    else:
        cache.y = y
        cache.u_tilde_ev_ravel = u_tilde(u, phi, delta, V, num_s, num_p, nums_a, num_a_max, parallel)
        u_tilde_ev_ravel = np.asarray(cache.u_tilde_ev_ravel)
        cache.u_tilde_sia_ev = u_tilde_sia(u_tilde_ev_ravel, sigma, num_s, num_p, nums_a, num_a_max, parallel)
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)

    flat_index = 0
    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state, player]):

                if action == 0:
                    out_[flat_index] += 1
                    for a in range(nums_a[state, player]):
                        out_[flat_index] -= sigma[state, player, a]
                else:
                    out_[flat_index] = (beta[state, player, 0] - beta[state, player, action] + lambda_ *
                                        (u_tilde_sia_ev[state,player,action] - u_tilde_sia_ev[state,player,0]))

                flat_index += 1

    for state in range(num_s):
        for player in range(num_p):
            out_[flat_index] -= V[state, player]
            for action in range(nums_a[state, player]):
                out_[flat_index] += sigma[state, player, action] * u_tilde_sia_ev[state, player, action]
            flat_index += 1

    return out_


# %% Jacobian matrix


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def J(np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] u, np.ndarray[np.float64_t, ndim=1] phi,
      np.ndarray[np.float64_t, ndim=1] delta, int [:,::1] nums_a, QreCache cache, bint parallel):
    """Jacobian matrix.

    J(y) = [  d_H_strat[s,i,a] / d_beta[s',i',a'],  d_H_strat[s,i,a] / d_V[s',i'],  d_H_strat[s,i,a] / d_lambda  ]
           [  d_H_val[s,i]     / d_beta[s',i',a'],  d_H_val[s,i]     / d_V[s',i'],  d_H_val[s,i]     / d_lambda  ]
    with
    y = [ beta[s,i,a],  V[s,i],  lambda ]
    """

    cdef:
        int num_s = nums_a.shape[0]
        int num_p = nums_a.shape[1]
        int num_a_max = np.max(nums_a)
        int num_a_tot = np.sum(nums_a)

        np.ndarray[np.float64_t, ndim=2] out_ = np.zeros((num_a_tot + num_s*num_p, num_a_tot + num_s*num_p + 1))
        np.ndarray[np.float64_t, ndim=3] beta = np.zeros((num_s, num_p, num_a_max))
        np.ndarray[np.float64_t, ndim=3] sigma
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot : num_a_tot + num_s*num_p].reshape((num_s, num_p))
        double lambda_ = y[num_a_tot + num_s*num_p]

        np.ndarray[np.float64_t, ndim=1] u_tilde_ev_ravel
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev
        np.ndarray[np.float64_t, ndim=5] u_tilde_sia_partial_beta_ev
        np.ndarray[np.float64_t, ndim=4] u_tilde_sia_partial_V_ev

        int state, player, action
        int flat_index = 0
        int row_state, row_player, row_action
        int col_state, col_player, col_action
        int row_index, col_index, col_index_init

    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state, player]):
                beta[state, player, action] = y[flat_index]
                flat_index += 1

    sigma = np.exp(beta)

    if cache is None:
        # cache disabled
        u_tilde_ev_ravel = u_tilde(u, phi, delta, V, num_s, num_p, nums_a, num_a_max, parallel)
        u_tilde_sia_ev = u_tilde_sia(u_tilde_ev_ravel, sigma, num_s, num_p, nums_a, num_a_max, parallel)
    elif arrays_equal(y, cache.y):
        # read intermediate values from cache
        u_tilde_ev_ravel = np.asarray(cache.u_tilde_ev_ravel)
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)
    else:
        # compute and write to cache
        cache.y = y
        cache.u_tilde_ev_ravel = u_tilde(u, phi, delta, V, num_s, num_p, nums_a, num_a_max, parallel)
        u_tilde_ev_ravel = np.asarray(cache.u_tilde_ev_ravel)
        cache.u_tilde_sia_ev = u_tilde_sia(u_tilde_ev_ravel, sigma, num_s, num_p, nums_a, num_a_max, parallel)
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)

    u_tilde_sia_partial_beta_ev = u_tilde_sia_partial_beta(u_tilde_ev_ravel, sigma,
                                                           num_s, num_p, nums_a, num_a_max, parallel)
    u_tilde_sia_partial_V_ev = phi_siat(phi, delta, sigma, num_s, num_p, nums_a, num_a_max, parallel)

    # first block: rows with d_H_strat[s,i,a]
    row_index = 0
    col_index_init = 0
    for row_state in range(num_s):
        for row_player in range(num_p):
            for row_action in range(nums_a[row_state, row_player]):

                # derivatives w.r.t. beta[s',i',a']
                # entries with s' != s are 0, thus no looping over s'
                col_index = col_index_init
                for col_player in range(num_p):
                    for col_action in range(nums_a[row_state, col_player]):

                        # diagonal sub-blocks: derivatives w.r.t. beta[s,i,a']
                        # (own actions in same state)
                        if row_player == col_player:
                            if row_action == 0:
                                out_[row_index, col_index] = -sigma[row_state, col_player, col_action]
                            else:
                                if col_action == 0:
                                    out_[row_index, col_index] = 1
                                elif row_action == col_action:
                                    out_[row_index, col_index] = -1

                        # off-diagonal sub-blocks: derivatives w.r.t. beta[s,i',a']
                        # (other players' actions in same state)
                        else:
                            # row_action == 0 -> entry = 0
                            if row_action != 0:
                                out_[row_index, col_index] = lambda_ * (
                                      u_tilde_sia_partial_beta_ev[row_state, row_player, row_action, col_player, col_action]
                                    - u_tilde_sia_partial_beta_ev[row_state, row_player,          0, col_player, col_action]
                                    )

                        col_index += 1

                # derivatives w.r.t. V[s',i']
                col_index = num_a_tot
                for col_state in range(num_s):
                    for col_player in range(num_p):
                        if col_player == row_player and row_action != 0:
                            out_[row_index, col_index] = lambda_ * (
                                  u_tilde_sia_partial_V_ev[row_state, row_player, row_action, col_state]
                                - u_tilde_sia_partial_V_ev[row_state, row_player,          0, col_state]
                                )
                        col_index += 1

                # derivative w.r.t. lambda
                # row_action == 0 -> entry = 0
                if row_action != 0:
                    out_[row_index, col_index] = (u_tilde_sia_ev[row_state, row_player, row_action]
                                                  - u_tilde_sia_ev[row_state, row_player, 0])

                row_index += 1

        for row_player in range(num_p):
            col_index_init += nums_a[row_state,row_player]

    # second block: rows with d_H_val[s,i]
    row_index = num_a_tot
    col_index_init = 0
    for row_state in range(num_s):
        for row_player in range(num_p):

            # derivatives w.r.t. beta[s',i',a']
            # entries with s' != s are 0, thus no looping over s'
            col_index = col_index_init

            for col_player in range(num_p):
                for col_action in range(nums_a[row_state, col_player]):

                    if row_player == col_player:
                        out_[row_index, col_index] += (sigma[row_state, row_player, col_action]
                                                       * u_tilde_sia_ev[row_state, row_player, col_action])
                    else:
                        for row_action in range(nums_a[row_state, row_player]):
                            out_[row_index, col_index] += sigma[row_state, row_player, row_action] \
                                * u_tilde_sia_partial_beta_ev[row_state, row_player, row_action, col_player, col_action]

                    col_index += 1

            # derivatives w.r.t. V[s',i']
            col_index = num_a_tot
            for col_state in range(num_s):
                for col_player in range(num_p):
                    if col_player == row_player:
                        if col_state == row_state:
                            out_[row_index, col_index] -= 1
                        for row_action in range(nums_a[row_state, row_player]):
                            out_[row_index, col_index] += sigma[row_state, row_player, row_action] \
                                * u_tilde_sia_partial_V_ev[row_state, row_player, row_action, col_state]


                    col_index += 1

            # derivative w.r.t. lambda = 0

            row_index += 1

        for row_player in range(num_p):
            col_index_init += nums_a[row_state, row_player]

    return out_


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=5] u_tilde_sia_partial_beta(np.ndarray[np.float64_t] u_tilde_ravel,
                                                               double[:,:,::1] sigma,
                                                               int num_s, int num_p,
                                                               int [:,::1] nums_a, int num_a_max, bint parallel):
    """Derivatives of u_tilde[s,i,a] w.r.t. log strategies beta[i',a'].
    No index s' in beta because the corresponding derivative is zero.
    """
    # see _shared_ct.pyx for comments on the general structure of these functions

    cdef:
        double[:,::1] u_tilde_reshaped = u_tilde_ravel.reshape((num_s, -1))
        np.ndarray[np.float64_t, ndim=5] out_np = np.zeros((num_s, num_p, num_a_max, num_p, num_a_max))
        double[:,:,:,:,::1] out_ = out_np
        int[:,::1] loop_profiles = np.zeros((num_s, num_p + 1), dtype=np.int32)
        int[::1] u_shape = np.array((num_s, num_p, *(num_a_max,) * num_p), dtype=np.int32)
        int[::1] u_strides = np.ones(2 + num_p, dtype=np.int32)
        int s, n

    # strides: offsets of the respective indices in u_ravel, so that: flat_index = multi-index (dot) u_strides
    # strides[-1] is 1; strides[-2] is 1*shape[-1]; strides[-3] is 1*shape[-1]*shape[-2] etc
    for n in range(num_p+2):
        for s in range(n):
            u_strides[s] *= u_shape[n]

    if parallel:
        for s in prange(num_s, schedule="static", nogil=True):
            u_tilde_sia_partial_beta_inner(out_[s,:,:,:,:], u_tilde_reshaped[s,:],
                                           sigma[s,:,:], u_strides, num_p, nums_a[s,:], loop_profiles[s,:])
    else:
        for s in range(num_s):
            u_tilde_sia_partial_beta_inner(out_[s,:,:,:,:], u_tilde_reshaped[s,:],
                                           sigma[s,:,:], u_strides, num_p, nums_a[s,:], loop_profiles[s,:])

    return out_np


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void u_tilde_sia_partial_beta_inner(double[:,:,:,::1] out_s, double[::1] u_tilde_s, double[:,::1] sigma,
                                    int[::1] u_strides,  int num_p, int[::1] nums_a, int[::1] loop_profile) nogil:
    cdef:
        int n, p, a, other, flat_index
        double temp_prob, u_a

    while loop_profile[0] == 0:
        for p in range(num_p):
            if loop_profile[p+1] != 0:
                continue
            # calc temp prob (sigma of all others) and flat index
            # can skip p for both: not part of temp_prob, action is 0 anyways
            temp_prob = 1.0
            flat_index = p * u_strides[1]
            for n in range(num_p):
                if n == p:
                    continue
                flat_index += loop_profile[n + 1] * u_strides[n + 2]
                temp_prob *= sigma[n, loop_profile[n + 1]]

            for a in range(nums_a[p]):
                u_a = temp_prob * u_tilde_s[flat_index]
                for other in range(num_p):
                    if other == p:
                        continue
                    out_s[p, a, other, loop_profile[other+1]] += u_a
                # done with a - increase index
                flat_index += u_strides[p+2]

        loop_profile[num_p] += 1
        for n in range(num_p):
            if loop_profile[num_p - n] == nums_a[num_p - n - 1]:
                loop_profile[num_p - n - 1] += 1
                loop_profile[num_p - n] = 0
            else:
                break


cdef class QreCache:
    """Caches intermediate results during computation of H or J, to be used later by the other Function."""
    cdef:
        double [::1] y
        double [::1] u_tilde_ev_ravel
        double [:,:,::1] u_tilde_sia_ev

    def __cinit__(self):
        self.y = np.zeros(1)
        self.u_tilde_ev_ravel = np.zeros(1)
        self.u_tilde_sia_ev = np.zeros((1,1,1))
