# cython: profile=True

# note: this file is adapted so that phi => shared between all players.

cimport cython
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
np.import_array()


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def H(np.ndarray[np.float64_t] y, u, phi, np.ndarray[np.float64_t, ndim=1] delta, np.ndarray[np.float64_t, ndim=3] rho,
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
        phi_sigma = phi_siat(phi.ravel(), delta, sigma, num_s, num_p, nums_a, num_a_max)
        # u_bar : pure strat utilities if others play sigma/rho mixture
        u_bar = t * u_sigma + (1 - t) * u_rho
        # phi_bar : pure strat transition probabilities if others play sigma/rho mixture
        phi_bar = t * phi_sigma + (1 - t) * phi_rho
        # u_tilde_sia_ev : total discounted utilities if others play sigma/rho mixture.
        # note: uses u_tilde, but not same shape as in qre (here, derivatives are taken first).
        u_tilde_sia_ev = u_tilde_deriv(u_bar, phi_bar, V)  # TODO: derivative version DONE
    elif arrays_equal(y, cache.y):
        # All other variables potentially in the cache are just intermediate steps in H:
        # u_sigma, phi_sigma, u_bar, phi_bar
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)
    else:
        cache.y = y
        cache.u_sigma = u_tilde_sia(u.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        cache.phi_sigma = phi_siat(phi.ravel(), delta, sigma, num_s, num_p, nums_a, num_a_max)
        # u_bar not needed in cache
        u_bar = t * np.asarray(cache.u_sigma) + (1 - t) * u_rho
        cache.phi_bar = t * np.asarray(cache.phi_sigma) + (1 - t) * phi_rho
        cache.u_tilde_sia_ev = u_tilde_deriv(u_bar, np.asarray(cache.phi_bar), V) # TODO: derivative version DONE
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


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def J(np.ndarray[np.float64_t] y, u, phi, np.ndarray[np.float64_t, ndim=1] delta, np.ndarray[np.float64_t, ndim=3] rho,
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
        phi_sigma = phi_siat(phi.ravel(), delta, sigma, num_s, num_p, nums_a, num_a_max)
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
        cache.phi_sigma = phi_siat(phi.ravel(), delta, sigma, num_s, num_p, nums_a, num_a_max)
        phi_sigma = np.asarray(cache.phi_sigma)
        u_bar = t * u_sigma + (1 - t) * u_rho
        cache.phi_bar = t * phi_sigma + (1 - t) * phi_rho
        phi_bar = np.asarray(cache.phi_bar)
        cache.u_tilde_sia_ev = u_tilde_deriv(u_bar, phi_bar, V)# TODO: derivative version DONE
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)

    # The following are used only for J, not for H -> never in cache.
    # u_hat, phi_hat: derivatives of u_bar, phi_bar wrt to t
    u_hat = u_sigma - u_rho
    phi_hat = phi_sigma - phi_rho
    # now: use previous to compute total discounted versions by ein-summing in continuation values.
    # u_tilde_sia_ev = u_tilde(u_bar, V, phi_bar)
    u_hat_sia_ev = u_tilde_deriv(u_hat, phi_hat, V)  # TODO: derivative version DONE
    # this is the full array of total discounted utilities for all pure strat profiles
    u_tilde_ev_ravel = u_tilde(u.ravel(), phi.ravel(), delta, V, num_s, num_p, nums_a, num_a_max)  # TODO: regular version - will come raveled!
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


# def u_tilde(u, V, phi):
#     """Add continuation values V to utilities u."""
#     return u + np.einsum('sp...S,Sp->sp...', phi, V)

def u_tilde_deriv(np.ndarray[np.float64_t, ndim=3] u_sia, np.ndarray[np.float64_t, ndim=4] phi_sia,
                  np.ndarray[np.float64_t, ndim=2] V):
    """Add continuation values V to derivatives u_sia (= ∂u/∂sigma)."""
    # note that currently, phi_sia already contains discount factors delta.
    return u_sia + np.einsum('spaS,Sp->spa', phi_sia, V)


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef u_tilde(np.ndarray[np.float64_t, ndim=1] u_ravel, np.ndarray[np.float64_t, ndim=1] phi_ravel,
                    np.ndarray[np.float64_t, ndim=1] delta, np.ndarray[np.float64_t, ndim=2] V,
                    int num_s, int num_p, int [:,::1] nums_a, int num_a_max):
    """Add continuation values V to utilities u."""
    # note: here, phi_ravel is without delta / player index.
    cdef:
        double [:,:,::1] out_
        double [:,:,::1] phi_reshaped = phi_ravel.reshape(num_s, -1, num_s)
        double [:,:,::1] u_reshaped =u_ravel.reshape(num_s, num_p, -1)

        int[:,::1] loop_profiles = np.zeros((num_s, num_p + 1), dtype=np.int32)
        int[::1] u_shape = np.array((num_s, num_p, *(num_a_max,) * num_p), dtype=np.int32)
        int[::1] u_strides = np.ones(2 + num_p, dtype=np.int32)

        int s, n

        np.ndarray[np.float64_t, ndim=2] dV = V * delta

    for n in range(num_p+2):
        for s in range(n):
            u_strides[s] *= u_shape[n]

    out_ = u_ravel.copy().reshape(num_s, num_p, -1)

    for s in range(num_s):
        u_tilde_inner(out_[s,:,:], phi_reshaped[s,:,:], dV, u_strides,
                              num_s, num_p, nums_a[s,:], num_a_max, loop_profiles[s,:])

    return np.asarray(out_).ravel()

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void u_tilde_inner(double[:,::1] out_s, double[:,::1] phi_s, double [:, ::1] dV, int[::1] u_strides,
                                int num_s, int num_p, int[::1] nums_a, int num_a_max, int[::1] loop_profile) nogil:
    cdef:
        int s, p, n, flat_index = 0

    while loop_profile[0] == 0:
        for p in range(num_p):
            for s in range(num_s):
                out_s[p, flat_index] += phi_s[flat_index, s] * dV[s, p]

        loop_profile[num_p] += 1
        flat_index += 1
        for n in range(num_p):
            if loop_profile[num_p - n] == nums_a[num_p - n - 1]:
                loop_profile[num_p - n - 1] += 1  # a of player (num_p-n) is increased
                loop_profile[num_p - n] = 0 # action of player (num_p-n-1) is reset to 0
                # we are possibly skipping (nums_a_max - nums_a(num_p-n-1)) actions of the latter.
                # player p has index p+2 in strides. Thus:
                flat_index += u_strides[num_p - n + 1] * (num_a_max - nums_a[num_p-n-1])
            else:
                break


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=3] u_tilde_sia(double[::1] u_tilde_ravel,
                                                   double[:,:,::1] sigma,
                                                   int num_s, int num_p, int [:,::1] nums_a, int num_a_max):
    """Payoffs (including continuation values) of player i using pure action a in state s,
    given other players play according to mixed strategy profile sigma[s,p,a].
    """

    cdef:
        double[:,:,::1] out_ = np.zeros((num_s, num_p, num_a_max))
        int[:,::1] loop_profiles = np.zeros((num_s, num_p + 1), dtype=np.int32)
        int[::1] u_shape = np.array((num_s, num_p, *(num_a_max,) * num_p), dtype=np.int32)
        int[::1] u_strides = np.ones(2 + num_p, dtype=np.int32)
        int s, n

    # strides: offsets of the respective indices in u_ravel, so that: flat_index = multi-index (dot) u_strides
    # strides[-1] is 1; strides[-2] is 1*shape[-1]; strides[-3] is 1*shape[-1]*shape[-2] etc
    for n in range(num_p+2):
        for s in range(n):
            u_strides[s] *= u_shape[n]

    # for s in prange(num_s, schedule="static", nogil=True):
    for s in range(num_s):
        u_tilde_sia_inner(out_[s,:,:], u_tilde_ravel[s*u_strides[0]:(s+1)*u_strides[0]], sigma[s,:,:],
                          u_strides, num_p, nums_a[s,:], loop_profiles[s,:])

    return np.asarray(out_)


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void u_tilde_sia_inner(double[:,::1] out_s, double[::1] u_tilde_s, double[:,::1] sigma,
                            int[::1] u_strides,  int num_p, int[::1] nums_a, int[::1] loop_profile) nogil:
    """Inner function (per state) of u_tilde_sia."""
    cdef:
        int p, a, n, flat_index
        double temp_prob

    while loop_profile[0] == 0:
        for p in range(num_p):
            if loop_profile[p + 1] != 0:
                continue

            # calc temp_prob, and flat_index
            # can skip p for both: temp_prob refers to others, action of p is 0 anyways (in loop_profile)
            temp_prob = 1.0
            flat_index = p * u_strides[1]
            for n in range(num_p):
                if n == p:
                    continue
                flat_index += loop_profile[n + 1] * u_strides[n + 2]
                temp_prob *= sigma[n, loop_profile[n + 1]]

            for a in range(nums_a[p]):
                out_s[p, a] += temp_prob * u_tilde_s[flat_index]
                flat_index += u_strides[p + 2]

        loop_profile[num_p] += 1
        for n in range(num_p):
            if loop_profile[num_p - n] == nums_a[num_p - n - 1]:
                loop_profile[num_p - n - 1] += 1
                loop_profile[num_p - n] = 0
            else:
                break

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=5] u_tilde_sijab(double [::1] u_tilde_ravel,
                                                     double [:,:,::1] sigma,
                                                     int num_s, int num_p, int [:,::1] nums_a,
                                                     int num_a_max):
    """Payoffs u_tilde_[s,i,j',a,b] (including continuation values) of player i using pure action a in state s,
    given player j uses pure action b and other players use mixed strategy profile sigma[s,i,a].
    """

    cdef:
        double [:,:,:,:,::1] out_ = np.zeros((num_s, num_p, num_p, num_a_max, num_a_max))
        int [:,::1] loop_profiles = np.zeros((num_s, num_p + 1), dtype=np.int32)
        int [::1] u_shape = np.array((num_s, num_p, *(num_a_max,) * num_p), dtype=np.int32)
        int [::1] u_strides = np.ones(2 + num_p, dtype=np.int32)
        int s, n

    # strides: offsets of the respective indices in u_ravel, so that: flat_index = multi-index (dot) u_strides
    # strides[-1] is 1; strides[-2] is 1*shape[-1]; strides[-3] is 1*shape[-1]*shape[-2] etc
    for n in range(num_p+2):
        for s in range(n):
            u_strides[s] *= u_shape[n]

    # for s in prange(num_s, schedule="static", nogil=True):
    for s in range(num_s):
        u_tilde_sijab_inner(out_[s,:,:,:,:], u_tilde_ravel[s*u_strides[0]:(s+1)*u_strides[0]], sigma[s,:,:],
                            u_strides,  num_p, nums_a[s,:], loop_profiles[s,:])

    return np.asarray(out_)

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void u_tilde_sijab_inner(double[:,:,:,::1] out_s, double[::1] u_tilde_s, double[:,::1] sigma,
                              int[::1] u_strides, int num_p, int[::1] nums_a,  int[::1] loop_profile) nogil:
    """Inner function (per state) of u_tilde_sijab."""
    cdef:
        int p0, p1, a0, a1, n
        int flat_index, index_offset
        double temp_prob

    # loop once over all action profiles.
    while loop_profile[0] == 0:
        # values are updated only for pairs p0, p1 (with p0<p1) for which a0=a1=0. (p0=p1 not needed)
        # looping over their actions a0, a1 is then done within.
        for p0 in range(num_p):
            if loop_profile[p0 + 1] != 0:
                continue
            for p1 in range(p0 + 1, num_p):
                if loop_profile[p1 + 1] != 0:
                    continue

                # get flat_index (for p0); temp_prob for all players except p0, p1 .
                # can skip p0 and p1: action is 0 for both (in loop_profile). temp_prob only includes others anyway.
                temp_prob = 1.0
                flat_index = p0 * u_strides[1]
                for n in range(num_p):
                    if n == p0 or n == p1:
                        continue
                    flat_index += loop_profile[n + 1] * u_strides[n + 2]
                    temp_prob *= sigma[n, loop_profile[n + 1]]
                # index_offset is the difference of indices: u[s,p1,...] - [s,p0,...]
                index_offset = (p1 - p0) * u_strides[1]

                # now : loop over both players' actions
                for a0 in range(nums_a[p0]):
                    for a1 in range(nums_a[p1]):
                        # update out-array for p0:
                        out_s[p0, p1, a0, a1] += u_tilde_s[flat_index] * temp_prob
                        # same, but for p1: (reverse p0,p1, a0,a1, taking offset into account)
                        out_s[p1, p0, a1, a0] += u_tilde_s[flat_index + index_offset] * temp_prob
                        # increase index for next a1:
                        flat_index += u_strides[p1 + 2]
                    # index: increase a0, but reset a1 to 0
                    flat_index += u_strides[p0 + 2] - nums_a[p1] * u_strides[p1 + 2]

        # go to next action profile
        loop_profile[num_p] += 1
        for n in range(num_p):
            if loop_profile[num_p - n] == nums_a[num_p - n - 1]:
                loop_profile[num_p - n - 1] += 1
                loop_profile[num_p - n] = 0
            else:
                break


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=4] phi_tilde_siat_old(double [::1] phi_ravel,
                                                     double [:,:,::1] sigma,
                                                     int num_s, int num_p, int [:,::1] nums_a,
                                                     int num_a_max):
    """Transition probabilities phi_[s,i,a,s'] of player i using pure action a in state s,
    given other players use mixed strategy profile sigma[s,i,a].
    """
    # NOTE: phi_ravel contains a player index / delta already multiplied in

    cdef:
        double[:,:,:,::1] out_ = np.zeros((num_s, num_p, num_a_max, num_s))
        int [:,::1] loop_profiles = np.zeros((num_s, num_p + 1), dtype=np.int32)
        int [::1] phi_shape = np.array((num_s, num_p, *(num_a_max,) * num_p, num_s), dtype=np.int32)
        int [::1] phi_strides = np.ones(3 + num_p, dtype=np.int32)
        int s, n

    # note: as of now, phi-indexes are: [s, p, a0, ...., aI, s], i.e. contain a player index.
    # strides: offsets of the respective indices in phi_ravel, so that: flat_index = multi-index (dot) u_strides
    # strides[-1] is 1; strides[-2] is 1*shape[-1]; strides[-3] is 1*shape[-1]*shape[-2] etc
    for n in range(num_p+3):
        for s in range(n):
            phi_strides[s] *= phi_shape[n]

    # for s in prange(num_s, schedule="static", nogil=True):
    for s in range(num_s):
        phi_tilde_siat_inner_old(out_[s,:,:,:], phi_ravel[s*phi_strides[0]:(s+1)*phi_strides[0]], sigma[s,:,:],
                            phi_strides, num_s, num_p, nums_a[s,:], loop_profiles[s,:])

    return np.asarray(out_)


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void phi_tilde_siat_inner_old(double[:,:,::1] out_s, double[::1] phi_s, double[:,::1] sigma,
                       int[::1] phi_strides,  int num_s, int num_p, int[::1] nums_a, int[::1] loop_profile) nogil:
    cdef:
        int p, a, n, to_state, flat_index
        double temp_prob

    while loop_profile[0] == 0:
        for p in range(num_p):
            if loop_profile[p + 1] != 0:
                continue

            temp_prob = 1.0
            flat_index = p * phi_strides[1]  # phi contains a player index (delta is multiplied in already).
            for n in range(num_p):
                if n == p:
                    continue
                temp_prob *= sigma[n, loop_profile[n + 1]]
                flat_index += loop_profile[n + 1] * phi_strides[n + 2]
            # flat-index: initial to-state is 0, so no increase necessary.

            for a in range(nums_a[p]):
                for to_state in range(num_s):
                    out_s[p, a, to_state] += temp_prob * phi_s[flat_index]
                    # increase index for next to_state
                    flat_index += 1
                # increase index for next action; reset to_state index to 0.
                flat_index += phi_strides[p + 2] - num_s

        # go to next action profile
        loop_profile[num_p] += 1
        for n in range(num_p):
            if loop_profile[num_p - n] == nums_a[num_p - n - 1]:
                loop_profile[num_p - n - 1] += 1
                loop_profile[num_p - n] = 0
            else:
                break

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=4] phi_siat(double [::1] phi_ravel, double[:] delta,
                                                     double [:,:,::1] sigma,
                                                     int num_s, int num_p, int [:,::1] nums_a,
                                                     int num_a_max):
    """Transition probabilities phi_[s,i,a,s'] of player i using pure action a in state s,
    given other players use mixed strategy profile sigma[s,i,a].
    """
    # note: this uses phi_uni.

    cdef:
        np.ndarray[np.float64_t, ndim = 4] out_np = np.zeros((num_s, num_p, num_a_max, num_s))
        double[:,:,:,::1] out_ = out_np
        int [:,::1] loop_profiles = np.zeros((num_s, num_p + 1), dtype=np.int32)
        int [::1] phi_shape = np.array((num_s, *(num_a_max,) * num_p, num_s), dtype=np.int32)
        int [::1] phi_strides = np.ones(2 + num_p, dtype=np.int32)
        int s, n, p

    # note: as of now, phi-indexes are: [s, a0, ...., aI, s], i.e. contain no player index.
    # strides: offsets of the respective indices in phi_ravel, so that: flat_index = multi-index (dot) u_strides
    # strides[-1] is 1; strides[-2] is 1*shape[-1]; strides[-3] is 1*shape[-1]*shape[-2] etc
    for n in range(num_p+2):
        for s in range(n):
            phi_strides[s] *= phi_shape[n]

    # for s in prange(num_s, schedule="static", nogil=True):
    for s in range(num_s):
        phi_siat_inner(out_[s,:,:,:], phi_ravel[s*phi_strides[0]:(s+1)*phi_strides[0]], sigma[s,:,:],
                       phi_strides, num_s, num_p, nums_a[s,:], loop_profiles[s,:])

    # multiply in delta for each player
    for p in range(num_p):
        out_np[:,p,:,:] *= delta[p]

    return out_np


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void phi_siat_inner(double[:,:,::1] out_s, double[::1] phi_s, double[:,::1] sigma,
                       int[::1] phi_strides,  int num_s, int num_p, int[::1] nums_a, int[::1] loop_profile) nogil:
    cdef:
        int p, a, n, to_state, flat_index
        double temp_prob

    while loop_profile[0] == 0:
        for p in range(num_p):
            if loop_profile[p + 1] != 0:
                continue

            temp_prob = 1.0
            # TODO: in this version, I removed the p-index!
            flat_index = 0
            for n in range(num_p):
                if n == p:
                    continue
                temp_prob *= sigma[n, loop_profile[n + 1]]
                flat_index += loop_profile[n + 1] * phi_strides[n + 1]
            # flat-index: initial to-state is 0, so no increase necessary.

            for a in range(nums_a[p]):
                for to_state in range(num_s):
                    out_s[p, a, to_state] += temp_prob * phi_s[flat_index]
                    # increase index for next to_state
                    flat_index += 1
                # increase index for next action; reset to_state index to 0.
                flat_index += phi_strides[p + 1] - num_s

        # go to next action profile
        loop_profile[num_p] += 1
        for n in range(num_p):
            if loop_profile[num_p - n] == nums_a[num_p - n - 1]:
                loop_profile[num_p - n - 1] += 1
                loop_profile[num_p - n] = 0
            else:
                break


@cython.nonecheck(False)
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


# cdef class GameShape:  # TODO: finish or remove
#     cdef:
#         int num_s
#         int num_p
#         int [:,::1] nums_a
#         int num_a_max
#         int num_a_tot
#         u_strides = int[::1]
#         phi_strides = int[::1]
#
#     def __cinit__(self, int num_s, int num_p, num):
#         pass


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
