"""Test implementation of QRE homotopy."""


import numpy as np

from sgamesolver.sgame import SGame
from sgamesolver.homotopy._qre import QRE, QRE_np, QRE_ct
from tests.random_game import create_random_game


# %% test QRE homotopy


class TestQRE:

    game = SGame(*create_random_game())
    y_rand = np.random.random(game.num_actions_total + game.num_states * game.num_players + 1)

    hom = QRE(game)
    hom_np = QRE_np(game)
    hom_ct = QRE_ct(game)

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
        assert np.max(np.abs(cls.hom.H(sol['y']))) < cls.hom.tracking_parameters['normal']['corrector_tol']
        sigma, V, t = cls.hom.y_to_sigma_V_t(sol['y'])
        assert np.max(cls.game.check_equilibrium(sigma)) < 0.01
        # relative (not absolute) convergence criterion for QRE, hence following line not applicable
        # assert np.max(cls.game.check_equilibrium(sigma)) < cls.hom.tracking_parameters['normal']['convergence_tol']


# %% run


if __name__ == '__main__':

    test_qre = TestQRE()
    test_qre.test_H_np_equal_ct()
    test_qre.test_J_np_equal_ct()
    test_qre.test_H_zero_at_starting_point()
    test_qre.test_detJ_nonzero_at_starting_point()
    test_qre.test_solve()
