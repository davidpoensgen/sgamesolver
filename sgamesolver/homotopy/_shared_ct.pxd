# just the signatures of _shared_ct

cimport numpy as np

cdef np.ndarray[np.float64_t, ndim=1] u_tilde(np.ndarray[np.float64_t, ndim=1],
                                              np.ndarray[np.float64_t, ndim=1],
                                              np.ndarray[np.float64_t, ndim=1],
                                              np.ndarray[np.float64_t, ndim=2],
                                              int, int, int [:,::1], int num_a_max, bint parallel)

cdef void u_tilde_inner(double[:,::1], double[:,::1], double, int[::1],
                                int, int, int[::1], int, int[::1])

cdef np.ndarray[np.float64_t, ndim=3] u_tilde_sia(np.ndarray[np.float64_t, ndim=1],
                                                   double[:,:,::1], int, int, int [:,::1], int, bint)

cdef void u_tilde_sia_inner(double[:,::1], double[::1], double[:,::1],
                            int[::1],  int, int[::1], int[::1])


cdef np.ndarray[np.float64_t, ndim=4] phi_siat(np.ndarray[np.float64_t, ndim=1], double[:],
                                               double [:,:,::1], int, int, int [:,::1],
                                               int, bint)

cdef void phi_siat_inner(double[:,:,::1], double[::1], double[:,::1],
                       int[::1],  int, int, int[::1], int[::1])

cdef bint arrays_equal(double [::1], double [::1])

cdef np.ndarray[np.float64_t, ndim=5] u_tilde_sijab(np.ndarray[np.float64_t, ndim=1], double [:,:,::1],
                                                    int, int, int [:,::1], int, bint):
