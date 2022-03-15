"""Test implementation of logarithmic game homotopy."""


import numpy as np

from dsgamesolver.sgame import SGame
from dsgamesolver.homotopies.loggame import LogGame_np, LogGame_ct
from tests.random_game import create_random_game


# %% test LogGame homotopy


class TestLogGame:

    game = SGame(*create_random_game())
    y_rand = np.random.random(game.num_actions_total + game.num_states * game.num_players + 1)
    hom_np = LogGame_np(game)
    hom_np.initialize()
    hom_ct = LogGame_ct(game)

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
