# cython: profile=False
# cython: language_level=3

cimport cython
import numpy as np
cimport numpy as np
np.import_array()

from ._shared_ct cimport u_tilde, u_tilde_sia, u_tilde_sijab, phi_siat, arrays_equal

@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def H(np.ndarray[np.float64_t] y, np.ndarray[np.float64_t, ndim=1] u, np.ndarray[np.float64_t, ndim=1] phi,
      np.ndarray[np.float64_t, ndim=1] delta, np.ndarray[np.float64_t, ndim=3] rho,
      np.ndarray[np.float64_t, ndim=3] nu, double eta,
      np.ndarray[np.float64_t, ndim=3] u_rho, np.ndarray[np.float64_t, ndim=4] phi_rho,
      int [:,::1] nums_a, bint eta_fix, bint parallel, TracingCache cache):
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
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot: num_a_tot + num_s * num_p].reshape((num_s, num_p))

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
        u_sigma = u_tilde_sia(u, sigma, num_s, num_p, nums_a, num_a_max, parallel)
        # phi_sigma : derivatives of phi wrt sigma (phi_si if i plays a, rest plays sigma)
        phi_sigma = phi_siat(phi, delta, sigma, num_s, num_p, nums_a, num_a_max, parallel)
        # u_bar : pure strat utilities if others play sigma/rho mixture
        u_bar = t * u_sigma + (1 - t) * u_rho
        # phi_bar : pure strat transition probabilities if others play sigma/rho mixture
        phi_bar = t * phi_sigma + (1 - t) * phi_rho
        # u_tilde_sia_ev : total discounted utilities if others play sigma/rho mixture.
        # note: uses u_tilde, but not same shape as in qre (here, derivatives are taken first).
        u_tilde_sia_ev = u_tilde_deriv(u_bar, phi_bar, V)
    elif arrays_equal(y, cache.y):
        # All other variables potentially in the cache are just intermediate steps in H:
        # u_sigma, phi_sigma, u_bar, phi_bar
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)
    else:
        cache.y = y
        cache.u_sigma = u_tilde_sia(u, sigma, num_s, num_p, nums_a, num_a_max, parallel)
        cache.phi_sigma = phi_siat(phi, delta, sigma, num_s, num_p, nums_a, num_a_max, parallel)
        # u_bar not needed in cache
        u_bar = t * np.asarray(cache.u_sigma) + (1 - t) * u_rho
        cache.phi_bar = t * np.asarray(cache.phi_sigma) + (1 - t) * phi_rho
        cache.u_tilde_sia_ev = u_tilde_deriv(u_bar, np.asarray(cache.phi_bar), V)
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


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def J(np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] u, np.ndarray[np.float64_t, ndim=1] phi,
      np.ndarray[np.float64_t, ndim=1] delta, np.ndarray[np.float64_t, ndim=3] rho,
      np.ndarray[np.float64_t, ndim=3] nu, double eta,
      np.ndarray[np.float64_t, ndim=3] u_rho, np.ndarray[np.float64_t, ndim=4] phi_rho,
      int [:,::1] nums_a, bint eta_fix, bint parallel, TracingCache cache):
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
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot : num_a_tot + num_s*num_p].reshape((num_s, num_p))
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
        u_sigma = u_tilde_sia(u, sigma, num_s, num_p, nums_a, num_a_max, parallel)
        # phi_sigma: derivative of phi_si wrt sigma_sia -> u of pure a if others play sigma
        phi_sigma = phi_siat(phi, delta, sigma, num_s, num_p, nums_a, num_a_max, parallel)
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
        cache.u_sigma = u_tilde_sia(u, sigma, num_s, num_p, nums_a, num_a_max, parallel)
        u_sigma = np.asarray(cache.u_sigma)
        cache.phi_sigma = phi_siat(phi, delta, sigma, num_s, num_p, nums_a, num_a_max, parallel)
        phi_sigma = np.asarray(cache.phi_sigma)
        u_bar = t * u_sigma + (1 - t) * u_rho
        cache.phi_bar = t * phi_sigma + (1 - t) * phi_rho
        phi_bar = np.asarray(cache.phi_bar)
        # u_tilde_sia_ev not needed in J, but in H if cache is loaded
        cache.u_tilde_sia_ev = u_tilde_deriv(u_bar, phi_bar, V)
        u_tilde_sia_ev = np.asarray(cache.u_tilde_sia_ev)

    # The following are used only for J, not for H -> never in cache.
    # u_hat, phi_hat: derivatives of u_bar, phi_bar wrt to t
    u_hat = u_sigma - u_rho
    phi_hat = phi_sigma - phi_rho
    # now: use previous to compute total discounted versions by ein-summing in continuation values.
    # u_tilde_sia_ev = u_tilde(u_bar, V, phi_bar)
    u_hat_sia_ev = u_tilde_deriv(u_hat, phi_hat, V)
    # this is the full array of total discounted utilities for all pure strat profiles
    u_tilde_ev_ravel = u_tilde(u, phi, delta, V, num_s, num_p, nums_a, num_a_max, parallel)
    # cross derivatives d^2 u / d sigma_sia d sigma_sjb
    u_tilde_sijab_ev = u_tilde_sijab(u_tilde_ev_ravel, sigma, num_s, num_p, nums_a, num_a_max, parallel)


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


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=3] u_tilde_deriv(np.ndarray[np.float64_t, ndim=3] u_sia,
                                                    np.ndarray[np.float64_t, ndim=4] phi_sia,
                                                    np.ndarray[np.float64_t, ndim=2] V):
    """Add continuation values V to derivatives u_sia (= ∂u/∂sigma_sia)."""
    # because the involved arrays are very tiny in comparison, there is no scope for optimization of this function
    # note that currently, phi_sia already contains discount factors delta
    return u_sia + np.einsum('spaS,Sp->spa', phi_sia, V)


cdef class TracingCache:
    """Cache for intermediate results from the computation of H or J, to be used next by the other function."""
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
