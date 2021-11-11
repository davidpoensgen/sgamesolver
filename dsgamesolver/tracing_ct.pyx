"""Cython implementation of Tracing homotopy."""

import cython
import numpy as np
cimport numpy as np


# %% auxiliary functions


def u_tilde(u, V, phi):
    """Payoffs including continuation values."""
    return u + np.einsum('sp...S,Sp->sp...', phi, V)


@cython.boundscheck(False)
@cython.wraparound(False)
def u_tilde_sia(np.ndarray[np.float64_t, ndim=1] u_tilde_ravel, np.ndarray[np.float64_t, ndim=3] sigma,
                int num_s, int num_p, np.ndarray[np.int32_t, ndim=2] nums_a, int num_a_max):
    """Payoffs (including continuation values) of player i using pure action a in state s,
    given other players play according to mixed strategy profile sigma[s,p,a].
    """

    cdef: 
        np.ndarray[np.float64_t, ndim=3] out_ = np.zeros((num_s, num_p, num_a_max))

        np.ndarray[np.int32_t, ndim=1] loop_profile = np.zeros(num_p + 1, dtype=np.int32)
        # loop_profile is used to loop over all action profiles.
        # loop_profile[1:num_p+1] gives current action profile.
        # loop_profile[0] in {0,1} indicates whether all action profiles have been explored (1) or not (0).
        # Last element of action profile is iterated first.
        # Once that is done, increase second last element by one and set last element to zero again, and so on.
        # Continue until very first element of loop_profile is increased from zero to one.

        double temp_prob 
        int state, player, other, n
        int flat_index = 0

    for state in range(num_s):
        for player in range(num_p):
            loop_profile[0 : num_p+1] = 0
            while loop_profile[0] == 0:

                temp_prob = 1
                for other in range(num_p):
                    if other == player:
                        continue
                    temp_prob *= sigma[state, other, loop_profile[other+1]]

                out_[state, player, loop_profile[player+1]] += temp_prob * u_tilde_ravel[flat_index]
                flat_index += 1

                loop_profile[num_p] += 1
                for n in range(num_p):
                    if loop_profile[num_p-n] == nums_a[state, num_p-n-1]:
                        loop_profile[num_p-n-1] += 1
                        loop_profile[num_p-n] = 0
                        flat_index += (num_a_max - nums_a[state, num_p-n-1]) * num_a_max**n        

    return out_


@cython.boundscheck(False)
@cython.wraparound(False)
def u_tilde_sijab(np.ndarray[np.float64_t, ndim=1] u_tilde_ravel, np.ndarray[np.float64_t, ndim=3] sigma,
                  int num_s, int num_p, np.ndarray[np.int32_t, ndim=2] nums_a, int num_a_max):
    """Payoffs u_tilde_[s,i,i',a,a'] (including continuation values) of player i using pure action a in state s,
    given player i' uses pure action a' and other players use mixed strategy profile sigma[s,i,a].
    
    (The case i'=i is explicitly included.)
    """

    cdef: 
        np.ndarray[np.float64_t, ndim=5] out_ = np.zeros((num_s, num_p, num_p, num_a_max, num_a_max))
        np.ndarray[np.int32_t, ndim=1] loop_profile = np.zeros(num_p+1, dtype=np.int32)
        double temp_prob 
        int state, player1, player2, other, n
        int flat_index = 0

    for state in range(num_s):
        for player1 in range(num_p):
            for player2 in range(num_p):
                loop_profile[0 : num_p+1] = 0
                while loop_profile[0] == 0:

                    temp_prob = 1
                    for other in range(num_p):
                        if other == player1 or other == player2:
                            continue
                        temp_prob *= sigma[state, other, loop_profile[other+1]]

                    out_[state, player1, player2, loop_profile[player1+1], loop_profile[player2+1]] += (
                        temp_prob * u_tilde_ravel[flat_index]
                        )
                    flat_index +=1  

                    loop_profile[num_p] +=1
                    for n in range(num_p):
                        if loop_profile[num_p-n] == nums_a[state, num_p-n-1]:
                            loop_profile[num_p-n-1] += 1
                            loop_profile[num_p-n] = 0
                            flat_index += (num_a_max - nums_a[state, num_p-n-1]) * num_a_max**n

                if player2 < num_p - 1:
                    flat_index -= num_a_max**num_p

    return out_


@cython.boundscheck(False)
@cython.wraparound(False)
def phi_tilde_siat(np.ndarray[np.float64_t, ndim=1] phi_ravel, np.ndarray[np.float64_t, ndim=3] sigma,
        int num_s, int num_p, np.ndarray[np.int32_t, ndim=2] nums_a, int num_a_max):
    """Transition probabilities phi_[s,i,a,s'] of player i using pure action a in state s, 
    given other players use mixed strategy profile sigma[s,i,a]
    """

    cdef: 
        np.ndarray[np.float64_t, ndim=4] out_ = np.zeros((num_s, num_p, num_a_max, num_s))
        np.ndarray[np.int32_t, ndim=1] loop_profile = np.zeros(num_p+1, dtype=np.int32)
        double temp_prob 
        int state, player, other, to_state, n
        int flat_index = 0

    for state in range(num_s):
        for player in range(num_p):
            loop_profile[0 : num_p+1] = 0
            while loop_profile[0] == 0:

                temp_prob = 1
                for other in range(num_p):
                    if other == player:
                        continue
                    temp_prob *= sigma[state, other, loop_profile[other+1]]

                for to_state in range(num_s):
                    out_[state, player, loop_profile[player+1], to_state] += temp_prob * phi_ravel[flat_index]
                    flat_index += 1

                loop_profile[num_p] +=1
                for n in range(num_p):
                    if loop_profile[num_p-n] == nums_a[state, num_p-n-1]:
                        loop_profile[num_p-n-1] += 1
                        loop_profile[num_p-n] = 0
                        flat_index += num_s * (num_a_max - nums_a[state, num_p-n-1]) * num_a_max**n

    return out_     


# %% homotopy function


@cython.boundscheck(False)
@cython.wraparound(False)
def H(np.ndarray[np.float64_t] y, u, phi, np.ndarray[np.float64_t, ndim=3] rho,
      np.ndarray[np.float64_t, ndim=3] nu, double eta_0,
      np.ndarray[np.float64_t, ndim=3] u_rho, np.ndarray[np.float64_t, ndim=4] phi_rho,
      int num_s, int num_p, np.ndarray[np.int32_t, ndim=2] nums_a, int num_a_max, int num_a_tot):
    """Homotopy function.
    
    H(y) = [  H_val[s,i,a]  ]
           [  H_strat[s,i]      ]
    with
    y = [ beta[s,i,a],  V[s,i],  t ]
    """

    cdef:
        np.ndarray[np.float64_t, ndim=1] out_ = np.zeros(num_a_tot + num_s*num_p)
        np.ndarray[np.float64_t, ndim=3] beta = np.ones((num_s, num_p, num_a_max))
        int state, player, action
        int flat_index = 0

    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state, player]):
                beta[state, player, action] = y[flat_index]
                flat_index += 1

    cdef:
        np.ndarray[np.float64_t, ndim=3] sigma = np.exp(beta)
        np.ndarray[np.float64_t, ndim=3] sigma_inv = np.exp(-beta)
        double nu_beta_sum
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot : num_a_tot + num_s*num_p].reshape(num_s, num_p)
        double t = y[num_a_tot + num_s*num_p]
        np.ndarray[np.float64_t, ndim=3] u_sigma = u_tilde_sia(u.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        np.ndarray[np.float64_t, ndim=4] phi_sigma = phi_tilde_siat(phi.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        np.ndarray[np.float64_t, ndim=3] u_bar = t*u_sigma + (1-t)*u_rho
        np.ndarray[np.float64_t, ndim=4] phi_bar = t*phi_sigma + (1-t)*phi_rho
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev = u_tilde(u_bar, V, phi_bar)

    flat_index = 0
    for state in range(num_s):
        for player in range(num_p):
            nu_beta_sum = 0
            for action in range(nums_a[state, player]):
                nu_beta_sum += nu[state, player, action] * (beta[state, player, action] - 1)

            for action in range(nums_a[state, player]):
                out_[flat_index] = (u_tilde_sia_ev[state, player, action] - V[state, player] + (1-t)**2 * eta_0
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
def J(np.ndarray[np.float64_t] y, u, phi, np.ndarray[np.float64_t, ndim=3] rho,
      np.ndarray[np.float64_t, ndim=3] nu, double eta_0,
      np.ndarray[np.float64_t, ndim=3] u_rho, np.ndarray[np.float64_t, ndim=4] phi_rho,
      int num_s, int num_p, np.ndarray[np.int32_t, ndim=2] nums_a, int num_a_max, int num_a_tot):
    """Jacobian matrix.

    J(y) = [  d_H_val[s,i]     / d_beta[s',i',a'],  d_H_val[s,i]     / d_V[s',i'],  d_H_val[s,i]     / d_t  ]
           [  d_H_strat[s,i,a] / d_beta[s',i',a'],  d_H_strat[s,i,a] / d_V[s',i'],  d_H_strat[s,i,a] / d_t  ]
    with
    y = [ beta[s,i,a],  V[s,i],  t ]
    """

    cdef:
        np.ndarray[np.float64_t, ndim=2] out_ = np.zeros((num_a_tot + num_s*num_p, num_a_tot + num_s*num_p + 1))
        np.ndarray[np.float64_t, ndim=3] beta = np.ones((num_s, num_p, num_a_max))
        int state, player, action
        int flat_index = 0
    
    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state, player]):
                beta[state, player, action] = y[flat_index]
                flat_index += 1
    
    cdef:
        np.ndarray[np.float64_t, ndim=3] sigma = np.exp(beta)
        np.ndarray[np.float64_t, ndim=3] sigma_inv = np.exp(-beta)
        double nu_beta_sum
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot : num_a_tot + num_s*num_p].reshape(num_s, num_p)
        double t = y[num_a_tot + num_s*num_p]
        np.ndarray[np.float64_t, ndim=3] u_sigma = u_tilde_sia(u.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        np.ndarray[np.float64_t, ndim=4] phi_sigma = phi_tilde_siat(phi.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        np.ndarray[np.float64_t, ndim=3] u_bar = t*u_sigma + (1-t)*u_rho
        np.ndarray[np.float64_t, ndim=4] phi_bar = t*phi_sigma + (1-t)*phi_rho
        np.ndarray[np.float64_t, ndim=3] u_hat = u_sigma - u_rho
        np.ndarray[np.float64_t, ndim=4] phi_hat = phi_sigma - phi_rho
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev = u_tilde(u_bar, V, phi_bar)
        np.ndarray[np.float64_t, ndim=3] u_hat_sia_ev = u_tilde(u_hat, V, phi_hat)
        np.ndarray[np.float64_t, ndim=1] u_tilde_ev_ravel = u_tilde(u, V, phi).ravel()
        np.ndarray[np.float64_t, ndim=5] u_tilde_sijab_ev = u_tilde_sijab(u_tilde_ev_ravel, sigma,
                                                                          num_s, num_p, nums_a, num_a_max)
        int row_state, row_player, row_action
        int col_state, col_player, col_action
        int row_index, col_index, col_index_init

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
                                out_[row_index, col_index] = ((1-t)**2 * eta_0 * nu[row_state, row_player, row_action]
                                                              * (1 - sigma_inv[row_state, row_player, row_action]))
                            else:
                                out_[row_index, col_index] = (1-t)**2 * eta_0 * nu[row_state, row_player, col_action]

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
                        out_[row_index, col_index] = phi_bar[row_state, row_player, row_action, col_state] - 1
                    else:
                        out_[row_index, col_index] = phi_bar[row_state, row_player, row_action, col_state]
                    col_index += num_p
                    if col_state == num_s - 1:
                        col_index -= row_player

                # derivative w.r.t. t
                out_[row_index, col_index] = (u_hat_sia_ev[row_state, row_player, row_action] - 2*(1-t) * eta_0
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


# %% variation with fixed eta: homotopy function


@cython.boundscheck(False)
@cython.wraparound(False)
def H_fixed_eta(np.ndarray[np.float64_t] y, u, phi, np.ndarray[np.float64_t, ndim=3] rho,
                np.ndarray[np.float64_t, ndim=3] nu, double eta,
                np.ndarray[np.float64_t, ndim=3] u_rho, np.ndarray[np.float64_t, ndim=4] phi_rho,
                int num_s, int num_p, np.ndarray[np.int32_t, ndim=2] nums_a, int num_a_max, int num_a_tot):
    """Homotopy function.
    
    H(y) = [  H_val[s,i,a]  ]
           [  H_strat[s,i]      ]
    with
    y = [ beta[s,i,a],  V[s,i],  t ]
    """

    cdef:
        np.ndarray[np.float64_t, ndim=1] out_ = np.zeros(num_a_tot + num_s*num_p)
        np.ndarray[np.float64_t, ndim=3] beta = np.ones((num_s, num_p, num_a_max))
        int state, player, action
        int flat_index = 0

    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state, player]):
                beta[state, player, action] = y[flat_index]
                flat_index += 1

    cdef:
        np.ndarray[np.float64_t, ndim=3] sigma = np.exp(beta)
        np.ndarray[np.float64_t, ndim=3] sigma_inv = np.exp(-beta)
        double nu_beta_sum
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot : num_a_tot + num_s*num_p].reshape(num_s, num_p)
        double t = y[num_a_tot + num_s*num_p]
        np.ndarray[np.float64_t, ndim=3] u_sigma = u_tilde_sia(u.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        np.ndarray[np.float64_t, ndim=4] phi_sigma = phi_tilde_siat(phi.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        np.ndarray[np.float64_t, ndim=3] u_bar = t*u_sigma + (1-t)*u_rho
        np.ndarray[np.float64_t, ndim=4] phi_bar = t*phi_sigma + (1-t)*phi_rho
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev = u_tilde(u_bar, V, phi_bar)

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


# %% variation with fixed eta: Jacobian matrix


@cython.boundscheck(False)
@cython.wraparound(False)
def J_fixed_eta(np.ndarray[np.float64_t] y, u, phi, np.ndarray[np.float64_t, ndim=3] rho,
                np.ndarray[np.float64_t, ndim=3] nu, double eta,
                np.ndarray[np.float64_t, ndim=3] u_rho, np.ndarray[np.float64_t, ndim=4] phi_rho,
                int num_s, int num_p, np.ndarray[np.int32_t, ndim=2] nums_a, int num_a_max, int num_a_tot):
    """Jacobian matrix.

    J(y) = [  d_H_val[s,i]     / d_beta[s',i',a'],  d_H_val[s,i]     / d_V[s',i'],  d_H_val[s,i]     / d_t  ]
           [  d_H_strat[s,i,a] / d_beta[s',i',a'],  d_H_strat[s,i,a] / d_V[s',i'],  d_H_strat[s,i,a] / d_t  ]
    with
    y = [ beta[s,i,a],  V[s,i],  t ]
    """

    cdef:
        np.ndarray[np.float64_t, ndim=2] out_ = np.zeros((num_a_tot + num_s*num_p, num_a_tot + num_s*num_p + 1))
        np.ndarray[np.float64_t, ndim=3] beta = np.ones((num_s, num_p, num_a_max))
        int state, player, action
        int flat_index = 0
    
    for state in range(num_s):
        for player in range(num_p):
            for action in range(nums_a[state, player]):
                beta[state, player, action] = y[flat_index]
                flat_index += 1
    
    cdef:
        np.ndarray[np.float64_t, ndim=3] sigma = np.exp(beta)
        np.ndarray[np.float64_t, ndim=3] sigma_inv = np.exp(-beta)
        double nu_beta_sum
        np.ndarray[np.float64_t, ndim=2] V = y[num_a_tot : num_a_tot + num_s*num_p].reshape(num_s, num_p)
        double t = y[num_a_tot + num_s*num_p]
        np.ndarray[np.float64_t, ndim=3] u_sigma = u_tilde_sia(u.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        np.ndarray[np.float64_t, ndim=4] phi_sigma = phi_tilde_siat(phi.ravel(), sigma, num_s, num_p, nums_a, num_a_max)
        np.ndarray[np.float64_t, ndim=3] u_bar = t*u_sigma + (1-t)*u_rho
        np.ndarray[np.float64_t, ndim=4] phi_bar = t*phi_sigma + (1-t)*phi_rho
        np.ndarray[np.float64_t, ndim=3] u_hat = u_sigma - u_rho
        np.ndarray[np.float64_t, ndim=4] phi_hat = phi_sigma - phi_rho
        np.ndarray[np.float64_t, ndim=3] u_tilde_sia_ev = u_tilde(u_bar, V, phi_bar)
        np.ndarray[np.float64_t, ndim=3] u_hat_sia_ev = u_tilde(u_hat, V, phi_hat)
        np.ndarray[np.float64_t, ndim=1] u_tilde_ev_ravel = u_tilde(u, V, phi).ravel()
        np.ndarray[np.float64_t, ndim=5] u_tilde_sijab_ev = u_tilde_sijab(u_tilde_ev_ravel, sigma,
                                                                          num_s, num_p, nums_a, num_a_max)
        int row_state, row_player, row_action
        int col_state, col_player, col_action
        int row_index, col_index, col_index_init

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
                            out_[row_index, col_index] = (
                                t * sigma[row_state, col_player, col_action]
                                * u_tilde_sijab_ev[row_state, row_player, col_player, row_action, col_action]
                                )

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
                out_[row_index, col_index] = (u_hat_sia_ev[row_state, row_player, row_action] - eta
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
