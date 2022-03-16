# TODO: this file collects tests that were found spread over the individual modules
#  when re-structuring the project in March/22 - D


# %% sgame.py
if __name__ == '__main__':

    from tests.random_game import create_random_game

    # SGame

    test_game = SGame(*create_random_game())

    test_game.detect_symmetries()

    test_sigma = test_game.centroid_strategy()
    test_V = test_game.get_values(test_sigma)

    test_losses = test_game.check_equilibrium(test_sigma)

    test_sigma_flat = test_game.flatten_strategies(test_sigma)
    test_V_flat = test_game.flatten_values(test_V)

    test_y = np.concatenate([np.log(test_sigma_flat), test_V_flat, [0.0]])

    # SGameHomotopy

    test_homotopy = SGameHomotopy(test_game)

    assert np.allclose(test_y, test_homotopy.sigma_V_t_to_y(test_sigma, test_V, 0.0))



# %% ipm.py
if __name__ == '__main__':

    from tests.random_game import create_random_game
    game = SGame(*create_random_game())

    # cython
    ipm_ct = IPM_ct(game)
    ipm_ct.initialize()
    ipm_ct.solver.solve()

    y0 = ipm_ct.y0
    """
    %timeit ipm_ct.H(y0)
    %timeit ipm_ct.J(y0)
    """

    # sympy
    ipm_sp = IPM_sp(game)
    ipm_sp.initialize()
    ipm_sp.solver.verbose = 2
    ipm_sp.solver.solve()

    """
    assert np.allclose(ipm_sp.y0, y0)
    %timeit ipm_sp.H(y0)
    %timeit ipm_sp.J(y0)
    """


# %% loggame.py
if __name__ == '__main__':

    from tests.random_game import create_random_game
    game = SGame(*create_random_game())

    # numpy
    loggame_np = LogGame_np(game)
    loggame_np.initialize()
    loggame_np.solver.solve()

    y0 = loggame_np.find_y0()
    """
    %timeit loggame_np.H(y0)
    %timeit loggame_np.J(y0)
    """

    # cython
    loggame_ct = LogGame_ct(game)
    loggame_ct.initialize()
    loggame_ct.solver.solve()
    """
    %timeit loggame_ct.H(y0)
    %timeit loggame_ct.J(y0)
    """



# %% qre.py
if __name__ == '__main__':

    from tests.random_game import create_random_game
    game = SGame(*create_random_game())

    # numpy
    qre_np = QRE_np(game)
    qre_np.initialize()
    qre_np.solver.solve()

    y0 = qre_np.find_y0()
    """
    %timeit qre_np.H(y0)
    %timeit qre_np.J(y0)
    """

    # cython
    qre_ct = QRE_ct(game)
    qre_ct.initialize()
    qre_ct.solver.solve()

    """
    %timeit qre_ct.H(y0)
    %timeit qre_ct.J(y0)
    """

    # numba
    # qre_nb = QRE_nb(game)
    # qre_nb.initialize()
    # qre_nb.solver.solve()
    # """
    # %timeit qre_nb.H(y0)
    # %timeit qre_nb.J(y0)
    # """


# %% tracing.py
if __name__ == '__main__':

    from tests.random_game import create_random_game
    game = SGame(*create_random_game())

    # numpy
    tracing_np = Tracing_np(game)
    tracing_np.initialize()
    tracing_np.solver.solve()

    y0 = tracing_np.find_y0()
    """
    %timeit tracing_np.H(y0)
    %timeit tracing_np.J(y0)
    """

    # cython
    tracing_ct = Tracing_ct(game)
    tracing_ct.initialize()
    tracing_ct.solver.solve()
    """
    %timeit tracing_ct.H(y0)
    %timeit tracing_ct.J(y0)
    """

    # Tracing with fixed eta

    tracing_fixed_eta_np = TracingFixedEta_np(game)
    tracing_fixed_eta_np.initialize()
    tracing_fixed_eta_np.solver.solve()

    tracing_fixed_eta_ct = TracingFixedEta_ct(game)
    tracing_fixed_eta_ct.initialize()
    tracing_fixed_eta_ct.solver.solve()

