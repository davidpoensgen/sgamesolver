cimport numpy as np

# headers for those functions to be imported by other modules

cdef np.ndarray[np.float64_t, ndim=1] u_tilde(np.ndarray[np.float64_t, ndim=1] u_ravel,
                                              np.ndarray[np.float64_t, ndim=1] phi_ravel,
                                              np.ndarray[np.float64_t, ndim=1] delta,
                                              np.ndarray[np.float64_t, ndim=2] V,
                                              int num_s, int num_p, int [:,::1] nums_a, int num_a_max, bint parallel)

cdef np.ndarray[np.float64_t, ndim=3] u_tilde_sia(np.ndarray[np.float64_t, ndim=1] u_tilde_ravel,
                                                   double[:,:,::1] sigma,
                                                   int num_s, int num_p, int [:,::1] nums_a, int num_a_max,
                                                   bint parallel)

cdef np.ndarray[np.float64_t, ndim=4] phi_siat(np.ndarray[np.float64_t, ndim=1] phi_ravel, double[:] delta,
                                               double [:,:,::1] sigma, int num_s, int num_p, int [:,::1] nums_a,
                                               int num_a_max, bint parallel)

cdef np.ndarray[np.float64_t, ndim=5] u_tilde_sijab(np.ndarray[np.float64_t, ndim=1] u_tilde_ravel,
                                                     double [:,:,::1] sigma,
                                                     int num_s, int num_p, int [:,::1] nums_a,
                                                     int num_a_max, bint parallel)

cdef bint arrays_equal(double [::1] a, double [::1] b)
