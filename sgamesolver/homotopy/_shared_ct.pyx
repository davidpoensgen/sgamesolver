"""Module containing the following subfunctions shared between the different homotopies:
u_tilde, u_tilde_sia, u_tilde_sijab, phi_siat, arrays_equal
"""
# cython: profile=False
# cython: language_level=3

# general comments on these functions:
"""
- all these functions operate on u/phi in some form
- because the number of players (and thus their dimensions) is variable, this is done on raveled arrays

- generally speaking, the outer function prepares some variables, and then delegates the actual work to the inner fct 
- (this allows parallelization, but is actually also faster than a single function even when using a simple range)

- the inner functions loop over all action profiles; because the number of players is variable (and thus the number of
  nested loops this would take), this uses the array loop_profile instead of nested loops
- loop_profile[1:] just counts action profiles from [0,  ... 0, 0], [0, .., 0, 1] ... 
  to [num_actions_p0, num_actions_p1, ....] 
- (i.e., loop_profile[p+1] contains the current action of p)
- loop_profile[0] is just a flag that switches from 0 to 1 once this is finished

- u/phi are indexed using a flat index. The array u_strides/phi_strides help convert from multi-index to flat:
  e.g. from u[s,p,a0,a1,...,aN] to u[flat] using the relation: multi_index (dot) strides = flat_index.
- technically, strides[-1] is 1; strides[-2] is 1*u.shape[-1]; strides[-3] is 1*u.shape[-1]*u.shape[-2] etc.
- (the inner function typically operate on a per-state-slice, so that this index is dropped there)

- calculations are ordered so that specific intermediate results do not have to be re-calculated later on.
- in particular, for a given action_profile of all players except p, calculations for all actions of p are 
  performed at once (allowing to re-use the sigma-product for all others)
- this is implemented as follows: for any action profile, calculations are done for all players playing action 0; 
  looping over their respective actions is then done in the inner loop. 
- (thus the frequent if loop_profile[p + 1] != 0: continue)
"""

cimport cython
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
np.import_array()

@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] u_tilde(np.ndarray[np.float64_t, ndim=1] u_ravel,
                                              np.ndarray[np.float64_t, ndim=1] phi_ravel,
                                              np.ndarray[np.float64_t, ndim=1] delta,
                                              np.ndarray[np.float64_t, ndim=2] V,
                                              int num_s, int num_p, int [:,::1] nums_a, int num_a_max, bint parallel):
    """Add continuation values V to utilities u[s,p,a0,a1,...aN]: u_tilde = u + δϕV"""
    cdef:
        double [:,:,::1] out_
        double [:,:,::1] phi_reshaped = phi_ravel.reshape((num_s, -1, num_s))
        double [:,::1] dV

        int[:,::1] loop_profiles = np.zeros((num_s, num_p + 1), dtype=np.int32)
        int[::1] u_shape = np.array((num_s, num_p, *(num_a_max,) * num_p), dtype=np.int32)
        int[::1] u_strides = np.ones(2 + num_p, dtype=np.int32)

        int s, n

    for n in range(num_p+2):
        for s in range(n):
            u_strides[s] *= u_shape[n]

    dV = V * delta

    out_ = u_ravel.copy().reshape((num_s, num_p, -1))

    if parallel:
        for s in prange(num_s, schedule="static", nogil=True):
            u_tilde_inner(out_[s, :, :], phi_reshaped[s, :, :], dV, u_strides,
                          num_s, num_p, nums_a[s, :], num_a_max, loop_profiles[s, :])
    else:
        for s in range(num_s):
            u_tilde_inner(out_[s, :, :], phi_reshaped[s, :, :], dV, u_strides,
                          num_s, num_p, nums_a[s, :], num_a_max, loop_profiles[s,:])

    return np.asarray(out_).ravel()


@cython.initializedcheck(False)
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
                loop_profile[num_p - n - 1] += 1  # action of player (num_p-n-2) is increased
                loop_profile[num_p - n] = 0 # action of player (num_p-n-1) is reset to 0
                # we are (possibly) skipping nums_a_max - nums_a[num_p-n-1] actions of the latter.
                # player x has index x+2 in strides. Thus:
                flat_index += u_strides[num_p-n+1] * (num_a_max - nums_a[num_p-n-1])
            else:
                break


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=3] u_tilde_sia(np.ndarray[np.float64_t, ndim=1] u_tilde_ravel,
                                                  double[:,:,::1] sigma,
                                                  int num_s, int num_p, int [:,::1] nums_a, int num_a_max,
                                                  bint parallel):
    """Derivatives of u_tilde_si(sigma) w.r.t. sigma_sia. 
    Put differently, u (including continuation values) of player i using pure action 
    a in state s, given other players play according to mixed strategy profile sigma.
    """

    cdef:
        double[:,::1] u_tilde_reshaped = u_tilde_ravel.reshape((num_s, -1))
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

    if parallel:
        for s in prange(num_s, schedule="static", nogil=True):
            u_tilde_sia_inner(out_[s, :, :], u_tilde_reshaped[s,:], sigma[s, :, :],
                              u_strides, num_p, nums_a[s, :], loop_profiles[s, :])
    else:
        for s in range(num_s):
            u_tilde_sia_inner(out_[s,:,:], u_tilde_reshaped[s,:], sigma[s,:,:],
                              u_strides, num_p, nums_a[s,:], loop_profiles[s,:])

    return np.asarray(out_)


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void u_tilde_sia_inner(double[:,::1] out_s, double[::1] u_tilde_s, double[:,::1] sigma,
                            int[::1] u_strides,  int num_p, int[::1] nums_a, int[::1] loop_profile) nogil:
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

@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=4] phi_siat(np.ndarray[np.float64_t, ndim=1] phi_ravel, double[:] delta,
                                               double [:,:,::1] sigma, int num_s, int num_p, int [:,::1] nums_a,
                                               int num_a_max, bint parallel):
    """Derivatives of phi(sigma) w.r.t. sigma_sia. Put differently, transition probabilities phi_[s,i,a,t]
    from state s to state t, if player i uses pure action a in state s, while other players play according to mixed
    strategy profile sigma.
    """

    cdef:
        double[:,::1] phi_reshaped = phi_ravel.reshape((num_s, -1))
        np.ndarray[np.float64_t, ndim = 4] out_np = np.zeros((num_s, num_p, num_a_max, num_s))
        double[:,:,:,::1] out_ = out_np
        int [:,::1] loop_profiles = np.zeros((num_s, num_p + 1), dtype=np.int32)
        int [::1] phi_shape = np.array((num_s, *(num_a_max,) * num_p, num_s), dtype=np.int32)
        int [::1] phi_strides = np.ones(2 + num_p, dtype=np.int32)
        int s, n, p

    # note: as of now, phi-indexes are: [s, a0, ...., aI, s], i.e. contain no player index.
    # strides: offsets of the respective indices in phi_ravel, so that: flat_index = multi-index (dot) u_strides
    # construction: strides[-1] is 1; strides[-2] is 1*shape[-1]; strides[-3] is 1*shape[-1]*shape[-2] etc
    for n in range(num_p+2):
        for s in range(n):
            phi_strides[s] *= phi_shape[n]

    if parallel:
        for s in prange(num_s, schedule="static", nogil=True):
            phi_siat_inner(out_[s, :, :, :], phi_reshaped[s,:], sigma[s, :, :],
                           phi_strides, num_s, num_p, nums_a[s, :], loop_profiles[s, :])
    else:
        for s in range(num_s):
            phi_siat_inner(out_[s,:,:,:], phi_reshaped[s,:], sigma[s,:,:],
                           phi_strides, num_s, num_p, nums_a[s,:], loop_profiles[s,:])

    # multiply in delta for each player
    for p in range(num_p):
        out_np[:,p,:,:] *= delta[p]

    return out_np


@cython.initializedcheck(False)
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


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=5] u_tilde_sijab(np.ndarray[np.float64_t, ndim=1] u_tilde_ravel,
                                                     double [:,:,::1] sigma,
                                                     int num_s, int num_p, int [:,::1] nums_a,
                                                     int num_a_max, bint parallel):
    """Cross derivatives d² u_tilde / d sigma_sia d sigma_sjb.
    Payoffs u_tilde_[s,i,j',a,b] (including continuation values) of player i using pure action a in state s,
    given player j uses pure action b and other players use mixed strategy profile sigma[s,i,a].
    """

    cdef:
        double[:,::1] u_tilde_reshaped = u_tilde_ravel.reshape((num_s, -1))
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

    if parallel:
        for s in prange(num_s, schedule="static", nogil=True):
            u_tilde_sijab_inner(out_[s, :, :, :, :], u_tilde_reshaped[s,:], sigma[s, :, :],
                                u_strides, num_p, nums_a[s, :], loop_profiles[s, :])
    else:
        for s in range(num_s):
            u_tilde_sijab_inner(out_[s,:,:,:,:], u_tilde_reshaped[s,:], sigma[s,:,:],
                                u_strides,  num_p, nums_a[s, :], loop_profiles[s, :])

    return np.asarray(out_)


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void u_tilde_sijab_inner(double[:,:,:,::1] out_s, double[::1] u_tilde_s, double[:,::1] sigma,
                              int[::1] u_strides, int num_p, int[::1] nums_a,  int[::1] loop_profile) nogil:
    cdef:
        int p0, p1, a0, a1, n
        int flat_index, index_offset
        double temp_prob

    # loop once over all action profiles.
    while loop_profile[0] == 0:
        # values are updated only for pairs p0, p1 (with p0<p1) for which a0=a1=0. (the case p0=p1 yields 0 anyway)
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
                        # same, but for p1: (reverse p0,p1; a0,a1; take offset into account)
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


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint arrays_equal(double [::1] a, double [::1] b):
    """Check if two 1d-arrays are identical."""
    cdef int i
    if a.shape[0] != b.shape[0]:
        return False
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False
    return True
