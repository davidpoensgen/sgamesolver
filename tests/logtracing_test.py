"""Test implementation of (logarithmic stochastic) tracing homotopy."""


import numpy as np

from sgamesolver.sgame import SGame
from sgamesolver.homotopy._logtracing import LogTracing, LogTracing_np, LogTracing_ct
from tests.random_game import create_random_game


# %% test log tracing homotopy


class TestLogTracing:

    game = SGame(*create_random_game())
    y_rand = np.random.random(game.num_actions_total + game.num_states * game.num_players + 1)

    hom = LogTracing(game)
    hom_np = LogTracing_np(game)
    hom_ct = LogTracing_ct(game)

    hom.solver_setup()

    def test_H_np_equal_ct(cls):
        assert np.allclose(cls.hom_ct.H(cls.y_rand), cls.hom_np.H(cls.y_rand))

    def test_J_np_equal_ct(cls):
        assert np.allclose(cls.hom_ct.J(cls.y_rand), cls.hom_np.J(cls.y_rand))

    def test_H_zero_at_starting_point(cls):
        H_y0 = cls.hom.H(cls.hom.y0)
        assert np.max(np.abs(H_y0)) < cls.hom.tracking_parameters['normal']['corrector_tol']

    def test_detJ_nonzero_at_starting_point(cls):
        detJ_y0 = np.linalg.det(cls.hom.J(cls.hom.y0)[:, :-1])
        assert np.abs(detJ_y0) > cls.hom.tracking_parameters['normal']['corrector_tol']

    def test_solve(cls):
        cls.hom.solver.verbose = 0
        sol = cls.hom.solver.start()
        assert sol['success']
        assert not sol['failure reason']
        assert np.max(np.abs(cls.hom.H(sol['y']))) < cls.hom_np.tracking_parameters['normal']['corrector_tol']
        sigma, V, t = cls.hom.y_to_sigma_V_t(sol['y'])
        assert np.max(cls.game.check_equilibrium(sigma)) < cls.hom.tracking_parameters['normal']['convergence_tol']


# %% run


if __name__ == '__main__':

    test_log_tracing = TestLogTracing()
    test_log_tracing.test_H_np_equal_ct()
    test_log_tracing.test_J_np_equal_ct()
    test_log_tracing.test_H_zero_at_starting_point()
    test_log_tracing.test_detJ_nonzero_at_starting_point()
    test_log_tracing.test_solve()

    # run large game:

    num_s = 5
    num_p = 5
    num_a = 5
    delta = 0.95
    rng = np.random.RandomState(42)

    game = SGame(*create_random_game(num_s, num_p, num_a, num_a, delta, delta, rng=rng))

    hom = LogTracing_ct(game)

    # hom.find_y0(dev=True)
    # print('done')

    """
    %timeit hom.find_y0(dev=True)
    %timeit hom.find_y0()
    """

    hom.solver_setup()
    hom.solver.max_steps = 1e6
    hom.solver.verbose = 2

    sol = hom.solver.start()
    print(sol)
