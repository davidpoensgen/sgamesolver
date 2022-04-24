# cython: profile=True

import cython
import numpy as np
cimport numpy as np
np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def H(np.ndarray[np.float64_t] y, u, phi, np.ndarray[np.float64_t, ndim=3] rho,
      np.ndarray[np.float64_t, ndim=3] nu, double eta,
      np.ndarray[np.float64_t, ndim=3] u_rho, np.ndarray[np.float64_t, ndim=4] phi_rho,
      int [:,::1] nums_a, bint eta_fix, TracingCache cache):
    """Homotopy function.
    
    H(y) = [  H_val[s,i,a]  ]
           [  H_strat[s,i]  ]
    with
    y = [ beta[s,i,a],  V[s,i],  t ]
    """

    cdef:
        int num_s = nums_a.shape[0]
        int num_p = nums_a.shape[1]
        int num_a_max = np.max(nums_a)
        int num_a_tot = np.sum(nums_a)

        np.ndarray[np.float64_t, ndim=1] out_ = np.zeros(num_a_tot + num_s * num_p)
        np.ndarray[np.float64_t, ndim=3] beta = np.ones((num_s, num_p, num_a_max))
        np.ndarray[np.float64_t, ndim=3] sigma
        np.ndarray[np.float64_t, ndim=3] sigma_inv
        double t = y[num_a_tot + num_s * num_p]
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot: num_a_tot + num_s * num_p].reshape(num_s, num_p)

        np.ndarray[np.float64_t, ndim=3] u_sigma
        np.ndarray[np.float64_t, ndim=4] phi_sigma
        np.ndarray[np.float64_t, ndim=3] u_bar
        np.ndarray[np.float64_t, ndim=4] phi_bar
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev
        int state, player, action
        double nu_beta_sum
        int flat_index = 0

    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state, player]):
                beta[state, player, action] = y[flat_index]
                flat_index += 1

    sigma = np.exp(beta)
    sigma_inv = np.exp(-beta)

    # eta_fix == False is a version of the logarithmic tracing procedure that sets η(t) = (1-t)*η_0
    if not eta_fix:
        eta = (1-t)*eta

    if cache is None:
        # cache is disabled: all calculations are performed and nothing is saved.
        # u_sigma: derivatives of u wrt sigma_sia: u_si if i plays a, others play sigma (without continuation values)
        u_sigma = u_tilde_sia(u.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        # phi_sigma : derivatives of phi wrt sigma (phi_si if i plays a, rest plays sigma)
        phi_sigma = phi_tilde_siat(phi.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        # u_bar : pure strat utilities if others play sigma/rho mixture
        u_bar = t * u_sigma + (1 - t) * u_rho
        # phi_bar : pure strat transition probabilities if others play sigma/rho mixture
        phi_bar = t * phi_sigma + (1 - t) * phi_rho
        # u_tilde_sia_ev : total discounted utilities if others play sigma/rho mixture.
        # note: uses u_tilde, but not same shape as in qre (here, derivatives are taken first).
        u_tilde_sia_ev = u_tilde(u_bar, V, phi_bar)
    elif arrays_equal(y, cache.y):
        # All other variables potentially in the cache are just intermediate steps in H:
        # u_sigma, phi_sigma, u_bar, phi_bar
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)
    else:
        cache.y = y
        cache.u_sigma = u_tilde_sia(u.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        cache.phi_sigma = phi_tilde_siat(phi.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        # u_bar not needed in cache
        u_bar = t * np.asarray(cache.u_sigma) + (1 - t) * u_rho
        cache.phi_bar = t * np.asarray(cache.phi_sigma) + (1 - t) * phi_rho
        cache.u_tilde_sia_ev = u_tilde(u_bar, V, cache.phi_bar)
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)

    flat_index = 0
    for state in range(num_s):
        for player in range(num_p):
            nu_beta_sum = 0
            for action in range(nums_a[state, player]):
                nu_beta_sum += nu[state, player, action] * (beta[state, player, action] - 1)

            for action in range(nums_a[state, player]):
                out_[flat_index] = (u_tilde_sia_ev[state, player, action] - V[state, player] + (1-t) * eta
                                    * (nu[state, player, action] * sigma_inv[state, player, action] + nu_beta_sum))
                flat_index += 1

    for state in range(num_s):
        for player in range(num_p):
            out_[flat_index] = -1
            for action in range(nums_a[state, player]):
                out_[flat_index] += sigma[state, player, action]
            flat_index += 1

    return out_


@cython.boundscheck(False)
@cython.wraparound(False)
def J(np.ndarray[np.float64_t] y, u, phi, np.ndarray[np.float64_t, ndim=3] rho,
      np.ndarray[np.float64_t, ndim=3] nu, double eta,
      np.ndarray[np.float64_t, ndim=3] u_rho, np.ndarray[np.float64_t, ndim=4] phi_rho,
      int [:,::1] nums_a, bint eta_fix, TracingCache cache):
    """Jacobian matrix.

    J(y) = [  d_H_val[s,i]     / d_beta[s',i',a'],  d_H_val[s,i]     / d_V[s',i'],  d_H_val[s,i]     / d_t  ]
           [  d_H_strat[s,i,a] / d_beta[s',i',a'],  d_H_strat[s,i,a] / d_V[s',i'],  d_H_strat[s,i,a] / d_t  ]
    with
    y = [ beta[s,i,a],  V[s,i],  t ]
    """

    cdef:
        int num_s = nums_a.shape[0]
        int num_p = nums_a.shape[1]
        int num_a_max = np.max(nums_a)
        int num_a_tot = np.sum(nums_a)

        np.ndarray[np.float64_t, ndim=2] out_ = np.zeros((num_a_tot + num_s*num_p, num_a_tot + num_s*num_p + 1))
        np.ndarray[np.float64_t, ndim=3] beta = np.ones((num_s, num_p, num_a_max))
        np.ndarray[np.float64_t, ndim=3] sigma
        np.ndarray[np.float64_t, ndim=3] sigma_inv
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot : num_a_tot + num_s*num_p].reshape(num_s, num_p)
        double t = y[num_a_tot + num_s*num_p]

        np.ndarray[np.float64_t, ndim=3] u_sigma
        np.ndarray[np.float64_t, ndim=4] phi_sigma
        np.ndarray[np.float64_t, ndim=3] u_bar
        np.ndarray[np.float64_t, ndim=4] phi_bar
        np.ndarray[np.float64_t, ndim=3] u_hat
        np.ndarray[np.float64_t, ndim=4] phi_hat
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev
        np.ndarray[np.float64_t, ndim=3] u_hat_sia_ev
        np.ndarray[np.float64_t, ndim=1] u_tilde_ev_ravel
        np.ndarray[np.float64_t, ndim=5] u_tilde_sijab_ev

        double nu_beta_sum
        int state, player, action
        int row_state, row_player, row_action
        int col_state, col_player, col_action
        int row_index, col_index, col_index_init

        int flat_index = 0

    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state, player]):
                beta[state, player, action] = y[flat_index]
                flat_index += 1

    sigma = np.exp(beta)
    sigma_inv = np.exp(-beta)

    # eta_fix == False is a version of the logarithmic tracing procedure that sets η(t) = (1-t)*η_0
    # For J, not only eta itself is adjusted, but also a factor in the column containing dH/dt:
    # eta fixed: d/dt (1-t)η = -η
    # eta varies in t: d/dt (1-t)η = d/dt (1-t)^2 η_0 = -2(1-t)η_0 = -2η
    cdef double eta_col_factor = 1.0
    if not eta_fix:
        eta = (1-t)*eta
        eta_col_factor = 2.0

    if cache is None:
        # cache is none -> disabled. Always compute all intermediate variables:
        # u_sigma: derivative of u_si wrt sigma_sia -> u of pure a if others play sigma
        u_sigma = u_tilde_sia(u.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        # phi_sigma: derivative of phi_si wrt sigma_sia -> u of pure a if others play sigma
        phi_sigma = phi_tilde_siat(phi.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        # u_bar, phi_bar: u/phi of si playing a if rest plays mixture sigma / rho
        u_bar = t * u_sigma + (1 - t) * u_rho
        phi_bar = t * phi_sigma + (1 - t) * phi_rho

    elif arrays_equal(y, cache.y):
        u_sigma = np.asarray(cache.u_sigma)
        phi_sigma = np.asarray(cache.phi_sigma)
        # u_bar = cache.u_bar
        phi_bar = np.asarray(cache.phi_bar)
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)
    else:
        cache.y = y
        cache.u_sigma = u_tilde_sia(u.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        u_sigma = np.asarray(cache.u_sigma)
        cache.phi_sigma = phi_tilde_siat(phi.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        phi_sigma = np.asarray(cache.phi_sigma)
        u_bar = t * u_sigma + (1 - t) * u_rho
        cache.phi_bar = t * phi_sigma + (1 - t) * phi_rho
        phi_bar = np.asarray(cache.phi_bar)
        cache.u_tilde_sia_ev = u_tilde(u_bar, V, phi_bar)
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)

    # The following are used only for J, not for H -> never in cache.
    # u_hat, phi_hat: derivatives of u_bar, phi_bar wrt to t
    u_hat = u_sigma - u_rho
    phi_hat = phi_sigma - phi_rho
    # now: use previous to compute total discounted versions by ein-summing in continuation values.
    # u_tilde_sia_ev = u_tilde(u_bar, V, phi_bar)
    u_hat_sia_ev = u_tilde(u_hat, V, phi_hat)
    # this is the full array of total discounted utilities for all pure strat profiles
    u_tilde_ev_ravel = u_tilde(u, V, phi).ravel()
    # cross derivatives d^2 u / d sigma_sia d sigma_sjb
    u_tilde_sijab_ev = u_tilde_sijab(u_tilde_ev_ravel, sigma, num_s, num_p, nums_a, num_a_max)


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
                                out_[row_index, col_index] = ((1-t) * eta * nu[row_state, row_player, row_action]
                                                              * (1 - sigma_inv[row_state, row_player, row_action]))
                            else:
                                out_[row_index, col_index] = (1-t) * eta * nu[row_state, row_player, col_action]

                        # off-diagonal sub-blocks: derivatives w.r.t. beta[s,i',a']
                        # (other players' actions in same state)
                        else:
                            out_[row_index, col_index] = (t * sigma[row_state, col_player, col_action]
                                * u_tilde_sijab_ev[row_state, row_player, col_player, row_action, col_action])

                        col_index += 1

                # derivatives w.r.t. V[s',i']
                col_index = num_a_tot + row_player
                for col_state in range(num_s):
                    if row_state == col_state:
                        out_[row_index, col_index] = phi_bar[row_state, row_player, row_action, col_state] - 1
                    else:
                        out_[row_index, col_index] = phi_bar[row_state, row_player, row_action, col_state]
                    col_index += num_p
                    if col_state == num_s - 1:
                        col_index -= row_player

                # derivative w.r.t. t
                out_[row_index, col_index] = (u_hat_sia_ev[row_state, row_player, row_action] - eta_col_factor * eta
                                              * (nu[row_state, row_player, row_action]
                                              * sigma_inv[row_state, row_player, row_action] + nu_beta_sum))

                row_index += 1

        for row_player in range(num_p):
            col_index_init += nums_a[row_state,row_player]

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


def u_tilde(u, V, phi):
    """Payoffs including continuation values."""
    return u + np.einsum('sp...S,Sp->sp...', phi, V)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=3] u_tilde_sia(np.ndarray[np.float64_t, ndim=1] u_tilde_ravel,
                                                   np.ndarray[np.float64_t, ndim=3] sigma,
                                                   int num_s, int num_p, int [:,::1] nums_a, int num_a_max):
    """Payoffs (including continuation values) of player i using pure action a in state s,
    given other players play according to mixed strategy profile sigma[s,p,a].
    """

    cdef:
        np.ndarray[np.float64_t, ndim = 3] out_ = np.zeros((num_s, num_p, num_a_max))
        np.ndarray[np.int32_t, ndim = 1] loop_profile = np.zeros(num_p + 1, dtype=np.int32)
        int s, p, a, n
        double temp_prob
        int flat_index
        np.ndarray[np.int32_t, ndim = 1] u_shape = np.array((num_s, num_p, *(num_a_max,) * num_p), dtype=np.int32)
        np.ndarray[np.int32_t, ndim = 1] u_strides = np.ones(2 + num_p, dtype=np.int32)

    # strides: multi_index dot strides = flatindex
    for n in range(num_p+1, 0, -1):
        u_strides[:n] *= u_shape[n]

    for s in range(num_s):
        loop_profile[:] = 0
        while loop_profile[0] == 0:
            for p in range(num_p):
                if loop_profile[p+1] != 0:
                    continue

                # calc temp_prob, and flat_index
                # can skip p for both: temp_prob refers to others, action of p is 0 anyways (in loop_profile)
                temp_prob = 1.0
                flat_index = s * u_strides[0] + p * u_strides[1]
                for n in range(num_p):
                    if n == p:
                        continue
                    flat_index += loop_profile[n + 1] * u_strides[n + 2]
                    temp_prob *= sigma[s, n, loop_profile[n + 1]]

                for a in range(nums_a[s,p]):
                    out_[s, p, a] += temp_prob * u_tilde_ravel[flat_index]
                    flat_index += u_strides[p+2]

            loop_profile[num_p] += 1
            for n in range(num_p):
                if loop_profile[num_p - n] == nums_a[s, num_p - n - 1]:
                    loop_profile[num_p - n - 1] += 1
                    loop_profile[num_p - n] = 0
                else:
                    break

    return out_


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=5] u_tilde_sijab(np.ndarray[np.float64_t, ndim=1] u_tilde_ravel,
                                                    np.ndarray[np.float64_t, ndim=3] sigma,
                                                    int num_s, int num_p, int [:,::1] nums_a,
                                                    int num_a_max):
    """Payoffs u_tilde_[s,i,j',a,b] (including continuation values) of player i using pure action a in state s,
    given player j uses pure action b and other players use mixed strategy profile sigma[s,i,a].
    """

    cdef:
        np.ndarray[np.float64_t, ndim=5] out_ = np.zeros((num_s, num_p, num_p, num_a_max, num_a_max))
        np.ndarray[np.int32_t, ndim = 1] loop_profile = np.zeros(num_p + 1, dtype=np.int32)
        double temp_prob
        int s, p0, p1, a0, a1, n
        int flat_index, index_offset

        np.ndarray[np.int32_t, ndim = 1] u_shape = np.array((num_s, num_p, *(num_a_max,) * num_p), dtype=np.int32)
        np.ndarray[np.int32_t, ndim = 1] u_strides = np.ones(2 + num_p, dtype=np.int32)

    # strides: offsets of the specific indices in u_ravel, so that: flat_index = multi-index (dot) u_strides
    for n in range(num_p+1, 0, -1):
        u_strides[:n] *= u_shape[n]

    for s in range(num_s):
        loop_profile[:] = 0
        # loop once over all action profiles.
        while loop_profile[0] == 0:
            # values are updated only for pairs (p0,p1) with p0>p1 for which a0=a1=0. (p0=p1 not needed / 0 anyways)
            # looping over their actions is then done within.
            for p0 in range(num_p):
                if loop_profile[p0+1] != 0:
                    continue
                for p1 in range(p0+1, num_p):
                    if loop_profile[p1+1] != 0:
                        continue

                    # get temp_prob for all others; flat_index for p0.
                    # can skip p0 and p1: action is 0 for both (in loop_profile). temp_prob only includes others anyway.
                    temp_prob = 1.0
                    flat_index = s * u_strides[0] + p0 * u_strides[1]
                    for n in range(num_p):
                        if n == p0 or n == p1:
                            continue
                        flat_index += loop_profile[n + 1] * u_strides[n + 2]
                        temp_prob *= sigma[s, n, loop_profile[n + 1]]
                    # index_offset is the difference of indices: u[s,p1,...] - [s,p0,...]
                    index_offset = (p1-p0) * u_strides[1]

                    # now : loop over both players' actions
                    for a0 in range(nums_a[s, p0]):
                        for a1 in range(nums_a[s, p1]):
                            # update out-array for p0:
                            out_[s, p0, p1, a0, a1] += u_tilde_ravel[flat_index]*temp_prob
                            # same, but for p1: (reverse p0,p1, a0,a1, taking offset into account)
                            out_[s, p1, p0, a1, a0] += u_tilde_ravel[flat_index + index_offset]*temp_prob
                            # increase index for next a1:
                            flat_index += u_strides[p1+2]
                        # index: increase a0, but reset a1 to 0
                        flat_index += u_strides[p0+2] - nums_a[s, p1] * u_strides[p1+2]

            # go to next action profile
            loop_profile[num_p] += 1
            for n in range(num_p):
                if loop_profile[num_p - n] == nums_a[s, num_p - n - 1]:
                    loop_profile[num_p - n - 1] += 1
                    loop_profile[num_p - n] = 0
                else:
                    break

    return out_


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=4] phi_tilde_siat(np.ndarray[np.float64_t, ndim=1] phi_ravel,
                                                     np.ndarray[np.float64_t, ndim=3] sigma,
                                                     int num_s, int num_p, int [:,::1] nums_a,
                                                     int num_a_max):
    """Transition probabilities phi_[s,i,a,s'] of player i using pure action a in state s,
    given other players use mixed strategy profile sigma[s,i,a].
    (NOTE: phi: contains a player index, delta already multiplied in)
    """

    cdef:
        np.ndarray[np.float64_t, ndim=4] out_ = np.zeros((num_s, num_p, num_a_max, num_s))
        int [:] loop_profile = np.zeros(num_p + 1, dtype=np.int32)
        double temp_prob
        int state, player, other, to_state, n
        int flat_index = 0

    for state in range(num_s):
        for player in range(num_p):
            loop_profile[0: num_p + 1] = 0
            while loop_profile[0] == 0:

                temp_prob = 1
                for other in range(num_p):
                    if other == player:
                        continue
                    temp_prob *= sigma[state, other, loop_profile[other + 1]]

                for to_state in range(num_s):
                    out_[state, player, loop_profile[player + 1], to_state] += temp_prob * phi_ravel[flat_index]
                    flat_index += 1

                loop_profile[num_p] += 1
                for n in range(num_p):
                    if loop_profile[num_p - n] == nums_a[state, num_p - n - 1]:
                        loop_profile[num_p - n - 1] += 1
                        loop_profile[num_p - n] = 0
                        flat_index += num_s * (num_a_max - nums_a[state, num_p - n - 1]) * num_a_max ** n

    return out_


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=4] phi_tilde_siat_old(np.ndarray[np.float64_t, ndim=1] phi_ravel,
                                                     np.ndarray[np.float64_t, ndim=3] sigma,
                                                     int num_s, int num_p, int [:,::1] nums_a,
                                                     int num_a_max):
    """Transition probabilities phi_[s,i,a,s'] of player i using pure action a in state s,
    given other players use mixed strategy profile sigma[s,i,a]
    """

    cdef:
        np.ndarray[np.float64_t, ndim=4] out_ = np.zeros((num_s, num_p, num_a_max, num_s))
        int [:] loop_profile = np.zeros(num_p + 1, dtype=np.int32)
        double temp_prob
        int state, player, other, to_state, n
        int flat_index = 0

    for state in range(num_s):
        for player in range(num_p):
            loop_profile[0: num_p + 1] = 0
            while loop_profile[0] == 0:

                temp_prob = 1
                for other in range(num_p):
                    if other == player:
                        continue
                    temp_prob *= sigma[state, other, loop_profile[other + 1]]

                for to_state in range(num_s):
                    out_[state, player, loop_profile[player + 1], to_state] += temp_prob * phi_ravel[flat_index]
                    flat_index += 1

                loop_profile[num_p] += 1
                for n in range(num_p):
                    if loop_profile[num_p - n] == nums_a[state, num_p - n - 1]:
                        loop_profile[num_p - n - 1] += 1
                        loop_profile[num_p - n] = 0
                        flat_index += num_s * (num_a_max - nums_a[state, num_p - n - 1]) * num_a_max ** n

    return out_


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint arrays_equal(double [:] a, double [:] b):
    """Check if two 1d-arrays are identical."""
    cdef int i
    if a.shape[0] != b.shape[0]:
        return False
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False
    return True


cdef class TracingCache:
    """Caches intermediate results during computation of H or J, to be used later by the other function."""
    cdef:
        double [::1] y
        double [:,:,::1] u_sigma
        double [:,:,:,::1] phi_sigma
        double [:,:,:,::1] phi_bar
        double [:,:,::1] u_tilde_sia_ev

    def __cinit__(self):
        self.y = np.zeros(1)
        self.u_sigma = np.zeros((1,1,1))
        self.phi_sigma = np.zeros((1,1,1,1))
        self.phi_bar = np.zeros((1,1,1,1))
        self.u_tilde_sia_ev = np.zeros((1,1,1))
