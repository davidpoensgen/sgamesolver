"""Test implementation of (logarithmic stochastic) tracing homotopy."""


import numpy as np

from dsgamesolver.sgame import SGame
from dsgamesolver.tracing import Tracing_np, Tracing_ct, TracingFixedEta_np, TracingFixedEta_ct
from tests.random_game import create_random_game


# %% test log tracing homotopy


class TestTracing:

    game = SGame(*create_random_game())
    y_rand = np.random.random(game.num_actions_total + game.num_states * game.num_players + 1)
    hom_np = Tracing_np(game)
    hom_np.initialize()
    hom_ct = Tracing_ct(game)

    def test_H_np_equal_ct(cls):
        assert np.allclose(cls.hom_ct.H(cls.y_rand), cls.hom_np.H(cls.y_rand))

    def test_J_np_equal_ct(cls):
        assert np.allclose(cls.hom_ct.J(cls.y_rand), cls.hom_np.J(cls.y_rand))

    # for all tests below: np implementation only

    def test_H_zero_at_starting_point(cls):
        H_y0 = cls.hom_np.H(cls.hom_np.y0)
        assert np.max(np.abs(H_y0)) < cls.hom_np.tracking_parameters['normal']['H_tol']

    def test_detJ_nonzero_at_starting_point(cls):
        detJ_y0 = np.linalg.det(cls.hom_np.J(cls.hom_np.y0)[:, :-1])
        assert np.abs(detJ_y0) > cls.hom_np.tracking_parameters['normal']['H_tol']

    def test_solve(cls):
        sol = cls.hom_np.solver.solve()
        assert sol['success']
        assert not sol['failure reason']
        assert np.max(np.abs(cls.hom_np.H(sol['y']))) < cls.hom_np.tracking_parameters['normal']['H_tol']
        sigma, V, t = cls.hom_np.y_to_sigma_V_t(sol['y'])
        assert np.max(cls.game.check_equilibrium(sigma)) < 0.01
        # TODO: assert np.max(cls.game.check_equilibrium(sigma)) < cls.hom_np.tracking_parameters['normal']['eq_tol']


# %% test log tracing homotopy with fixed eta


class TestTracingFixedEta:

    game = SGame(*create_random_game())
    y_rand = np.random.random(game.num_actions_total + game.num_states * game.num_players + 1)
    hom_np = TracingFixedEta_np(game)
    hom_np.initialize()
    hom_ct = TracingFixedEta_ct(game)

    def test_H_np_equal_ct(cls):
        assert np.allclose(cls.hom_ct.H(cls.y_rand), cls.hom_np.H(cls.y_rand))

    def test_J_np_equal_ct(cls):
        assert np.allclose(cls.hom_ct.J(cls.y_rand), cls.hom_np.J(cls.y_rand))

    # for all tests below: np implementation only

    def test_H_zero_at_starting_point(cls):
        H_y0 = cls.hom_np.H(cls.hom_np.y0)
        assert np.max(np.abs(H_y0)) < cls.hom_np.tracking_parameters['normal']['H_tol']

    def test_detJ_nonzero_at_starting_point(cls):
        detJ_y0 = np.linalg.det(cls.hom_np.J(cls.hom_np.y0)[:, :-1])
        assert np.abs(detJ_y0) > cls.hom_np.tracking_parameters['normal']['H_tol']

    def test_solve(cls):
        sol = cls.hom_np.solver.solve()
        assert sol['success']
        assert not sol['failure reason']
        assert np.max(np.abs(cls.hom_np.H(sol['y']))) < cls.hom_np.tracking_parameters['normal']['H_tol']
        sigma, V, t = cls.hom_np.y_to_sigma_V_t(sol['y'])
        assert np.max(cls.game.check_equilibrium(sigma)) < 0.01
        # TODO: assert np.max(cls.game.check_equilibrium(sigma)) < cls.hom_np.tracking_parameters['normal']['eq_tol']


# %% run large game


if __name__ == '__main__':

    num_s = 20
    num_p = 5
    num_a = 10
    delta = 0.95
    rng = np.random.RandomState(42)

    game = SGame(*create_random_game(num_s, num_p, num_a, num_a, delta, delta, rng=rng))

    hom = TracingFixedEta_ct(game)
    hom.initialize()
    hom.solver.store_path = True
    hom.solver.max_steps = 1e6
    hom.solver.verbose = 2

    sol = hom.solver.solve()
    print(sol)
