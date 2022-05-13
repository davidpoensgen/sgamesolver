"""Test implementation of interior point method homotopy."""


import numpy as np

from sgamesolver.sgame import SGame
from sgamesolver.homotopy._ipm import IPM, IPM_ct  # , IPM_sp
from tests.random_game import create_random_game


# %% test IPM homotopy


class TestIPM:

    game = SGame(*create_random_game())
    y_rand = np.random.random(game.num_actions_total + game.num_states * game.num_players + 1)

    hom = IPM(game)
    hom_ct = IPM_ct(game)
    # hom_sp = IPM_sp(game)

    hom.solver_setup()

    def test_H_ct_equal_sp(cls):
        pass
        # assert np.allclose(cls.hom_ct.H(cls.y_rand), cls.hom_sp.H(cls.y_rand))

    def test_J_ct_equal_sp(cls):
        pass
        # assert np.allclose(cls.hom_ct.J(cls.y_rand), cls.hom_sp.J(cls.y_rand))

    # for all tests below: ct implementation only

    def test_H_zero_at_starting_point(cls):
        H_y0 = cls.hom.H(cls.hom.y0)
        assert np.max(np.abs(H_y0)) < cls.hom.tracking_parameters['normal']['corrector_tol']

    def test_detJ_nonzero_at_starting_point(cls):
        detJ_y0 = np.linalg.det(cls.hom.J(cls.hom.y0)[:, :-1])
        assert np.abs(detJ_y0) > cls.hom.tracking_parameters['normal']['corrector_tol']

    def test_solve(cls):
        cls.hom.solver.verbose = 0
        sol = cls.hom.solver.start()  # noqa
        # assert sol['success']
        # assert not sol['failure reason']
        # assert np.max(np.abs(cls.hom.H(sol['y']))) < cls.hom.tracking_parameters['normal']['corrector_tol']
        # sigma, V, t = cls.hom.y_to_sigma_V_t(sol['y'])
        # assert np.max(cls.game.check_equilibrium(sigma)) < cls.hom.tracking_parameters['normal']['convergence_tol']


# %% run


if __name__ == '__main__':

    test_ipm = TestIPM()
    test_ipm.test_H_ct_equal_sp()
    test_ipm.test_J_ct_equal_sp()
    test_ipm.test_H_zero_at_starting_point()
    test_ipm.test_detJ_nonzero_at_starting_point()
    test_ipm.test_solve()
