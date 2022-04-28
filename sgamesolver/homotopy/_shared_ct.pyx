# this file includes:
# u_tilde
# u_tilde_sia
# phi_sia

# note: imports just not to confuse IDE - make sure to comment out before compilation
import numpy as np
cimport numpy as np
import cython
from cython.parallel cimport prange


@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] u_tilde(np.ndarray[np.float64_t, ndim=1] u_ravel,
                                              np.ndarray[np.float64_t, ndim=1] phi_ravel,
                                              np.ndarray[np.float64_t, ndim=1] delta,
                                              np.ndarray[np.float64_t, ndim=2] V,
                                              int num_s, int num_p, int [:,::1] nums_a, int num_a_max, bint parallel):
    """Add continuation values V to utilities u."""
    # note: here, phi_ravel is without delta / player index.
    cdef:
        double [:,:,::1] out_
        double [:,:,::1] phi_reshaped = phi_ravel.reshape(num_s, -1, num_s)
        double [:,:,::1] u_reshaped =u_ravel.reshape(num_s, num_p, -1)
        double [:,::1] dV

        int[:,::1] loop_profiles = np.zeros((num_s, num_p + 1), dtype=np.int32)
        int[::1] u_shape = np.array((num_s, num_p, *(num_a_max,) * num_p), dtype=np.int32)
        int[::1] u_strides = np.ones(2 + num_p, dtype=np.int32)

        int s, n

    for n in range(num_p+2):
        for s in range(n):
            u_strides[s] *= u_shape[n]

    dV = V * delta
    out_ = u_ravel.copy().reshape(num_s, num_p, -1)

    if parallel:
        for s in prange(num_s, schedule="static", nogil=True):
            u_tilde_inner(out_[s, :, :], phi_reshaped[s, :, :], dV, u_strides,
                          num_s, num_p, nums_a[s, :], num_a_max, loop_profiles[s, :])
    else:
        for s in range(num_s):
            u_tilde_inner(out_[s,:,:], phi_reshaped[s, :, :], dV, u_strides,
                          num_s, num_p, nums_a[s, :], num_a_max, loop_profiles[s,: ])

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
                loop_profile[num_p - n - 1] += 1  # a of player (num_p-n) is increased
                loop_profile[num_p - n] = 0 # action of player (num_p-n-1) is reset to 0
                # we are possibly skipping (nums_a_max - nums_a(num_p-n-1)) actions of the latter.
                # player p has index p+2 in strides. Thus:
                flat_index += u_strides[num_p - n + 1] * (num_a_max - nums_a[num_p-n-1])
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
    """Payoffs (including continuation values) of player i using pure action a in state s,
    given other players play according to mixed strategy profile sigma[s,p,a].
    """

    cdef:
        double[:,::1] u_tilde_reshaped = u_tilde_ravel.reshape(num_s, -1)
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

@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=4] phi_siat(double [::1] phi_ravel, double[:] delta,  double [:,:,::1] sigma,
                                                int num_s, int num_p, int [:,::1] nums_a, int num_a_max, bint parallel):
    """Transition probabilities phi_[s,i,a,s'] of player i using pure action a in state s,
    given other players use mixed strategy profile sigma[s,i,a].
    Corresponds to the derivative of u_tilde w.r.t. to sigma_sia
    """

    cdef:
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
            phi_siat_inner(out_[s, :, :, :], phi_ravel[s * phi_strides[0]:(s + 1) * phi_strides[0]], sigma[s, :, :],
                           phi_strides, num_s, num_p, nums_a[s, :], loop_profiles[s, :])
    else:
        for s in range(num_s):
            phi_siat_inner(out_[s,:,:,:], phi_ravel[s*phi_strides[0]:(s+1)*phi_strides[0]], sigma[s,:,:],
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
cdef bint arrays_equal(double [::1] a, double [::1] b):
    """Check if two 1d-arrays are identical."""
    cdef int i
    if a.shape[0] != b.shape[0]:
        return False
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False
    return True
