"""Test implementation of interior point method homotopy."""


import numpy as np

from dsgamesolver.sgame import SGame
from dsgamesolver.homotopy.ipm import IPM_ct, IPM_sp
from tests.random_game import create_random_game


# %% test IPM homotopy


class TestIPM:

    game = SGame(*create_random_game())
    y_rand = np.random.random(game.num_actions_total + game.num_states * game.num_players + 1)
    hom_ct = IPM_ct(game)
    hom_ct.solver_setup()
    hom_sp = IPM_sp(game)

    # def test_H_ct_equal_sp(cls):
    #     assert np.allclose(cls.hom_ct.H(cls.y_rand), cls.hom_sp.H(cls.y_rand))

    # def test_J_ct_equal_sp(cls):
    #     assert np.allclose(cls.hom_ct.J(cls.y_rand), cls.hom_sp.J(cls.y_rand))

    # for all tests below: ct implementation only

    def test_H_zero_at_starting_point(cls):
        H_y0 = cls.hom_ct.H(cls.hom_ct.y0)
        assert np.max(np.abs(H_y0)) < cls.hom_ct.tracking_parameters['normal']['H_tol']

    def test_detJ_nonzero_at_starting_point(cls):
        detJ_y0 = np.linalg.det(cls.hom_ct.J(cls.hom_ct.y0)[:, :-1])
        assert np.abs(detJ_y0) > cls.hom_ct.tracking_parameters['normal']['H_tol']

    # def test_solve(cls):
    #     sol = cls.hom_ct.solver.solve()
    #     assert sol['success']
    #     assert not sol['failure reason']
    #     assert np.max(np.abs(cls.hom_ct.H(sol['y']))) < cls.hom_ct.tracking_parameters['normal']['H_tol']
    #     sigma, V, t = cls.hom_ct.y_to_sigma_V_t(sol['y'])
    #     assert np.max(cls.game.check_equilibrium(sigma)) < 0.01
    #     # TODO: assert np.max(cls.game.check_equilibrium(sigma)) < cls.hom_ct.tracking_parameters['normal']['eq_tol']
