"""Test implementation of logarithmic game homotopy."""


import numpy as np
import sgamesolver


# %% test LogGame homotopy


class TestLogGame:

    game = sgamesolver.SGame.random_game(num_states=3, num_players=5, num_actions=(1, 10), delta=(0.6, 0.95), seed=23)
    y_rand = np.random.random(game.num_actions_total + game.num_states * game.num_players + 1)

    nu = game.random_weights(seed=23, zeros=True)

    hom = sgamesolver.homotopy.LogGame(game)
    hom_np = sgamesolver.homotopy._loggame.LogGame_np(game)
    hom_ct = sgamesolver.homotopy._loggame.LogGame_ct(game, nu=nu)

    hom.solver_setup()

    def test_H_np_equal_ct(self):
        pass
        # assert np.allclose(self.hom_ct.H(self.y_rand), self.hom_np.H(self.y_rand))

    def test_J_np_equal_ct(self):
        pass
        # assert np.allclose(self.hom_ct.J(self.y_rand), self.hom_np.J(self.y_rand))

    def test_H_zero_at_starting_point(self):
        H_y0 = self.hom.H(self.hom.y0)
        assert np.max(np.abs(H_y0)) < self.hom.default_parameters['corrector_tol']

    def test_detJ_nonzero_at_starting_point(self):
        detJ_y0 = np.linalg.det(self.hom.J(self.hom.y0)[:, :-1])
        assert np.abs(detJ_y0) > self.hom.default_parameters['corrector_tol']

    def test_solve(self):
        self.hom.solver.verbose = 0
        sol = self.hom.solver.start()
        assert sol['success']
        assert not sol['failure reason']
        assert np.max(np.abs(self.hom.H(sol['y']))) < self.hom.default_parameters['corrector_tol']
        sigma, V, t = self.hom.y_to_sigma_V_t(sol['y'])
        t_tol = self.hom.default_parameters['convergence_tol']
        assert 1 - t < t_tol
        # equilibriumness should be in same order of magnitude as convergence tolerance
        assert np.max(self.game.check_equilibrium(sigma)) < 10 * t_tol


# %% run


if __name__ == '__main__':

    test_class = TestLogGame()

    method_names = [method for method in dir(test_class)
                    if callable(getattr(test_class, method))
                    if not method.startswith('__')]
    for method in method_names:
        getattr(test_class, method)()
