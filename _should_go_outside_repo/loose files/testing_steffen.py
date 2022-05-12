"""Testing stuff for development."""


import numpy as np

from sgamesolver.sgame import SGame
from sgamesolver._qre import QRE_np, QRE_ct
from sgamesolver._tracing import Tracing_np, Tracing_ct
# from dsgamesolver.loggame import LogGame_np, LogGame_ct
# from dsgamesolver.ipm import IPM_ct, IPM_sp
from tests.random_game import create_random_game
from tests.timings import HomotopyTimer


np.random.seed(42)


# %% random game


num_s = 5           # number of states
num_p = 5           # number of players
num_a_max = 5       # maximum number of actions
num_a_min = 5       # minimum number of actions
delta_max = 0.95    # maximum discount factor
delta_min = 0.95    # minimum discount factor

a = 0               # payoffs in [a, a+b]
b = 1               # payoffs in [a, a+b]


# %% homotopy testing


u, phi, delta = create_random_game(num_s, num_p, num_a_min, num_a_max, delta_min, delta_max)
u = [a + b*u_s for u_s in u]

game = SGame(u, phi, delta)

y_rand = np.random.random(game.num_actions_total + game.num_states*game.num_players + 1)


# tracing_ct = Tracing_ct(game)
# tracing_ct.initialize()
# tracing_ct.solver.verbose = 2
# tracing_ct.solver.max_steps = 9000
# sol_tracing_ct = tracing_ct.solver.solve()


# %% run all

# QRE:

qre_np = QRE_np(game)
qre_np.solver_setup()
sol_qre_np = qre_np.solver.start()

qre_ct = QRE_ct(game)
qre_ct.solver_setup()
qre_ct.solver.verbose = 2
sol_qre_ct = qre_ct.solver.start()

assert np.allclose(sol_qre_np["y"], sol_qre_ct["y"])


# Tracing:

tracing_np = Tracing_np(game)
tracing_np.solver_setup()
sol_tracing_np = tracing_np.solver.start()

tracing_ct = Tracing_ct(game)
tracing_ct.solver_setup()
tracing_ct.solver.verbose = 2
sol_tracing_ct = tracing_ct.solver.start()

assert np.allclose(sol_tracing_np["y"], sol_tracing_ct["y"])


# LogGame:

# log_game_np = LogGame_np(game)
# log_game_np.solver_setup()
# sol_log_game_np = log_game_np.solver.start()

# log_game_ct = LogGame_ct(game)
# log_game_ct.solver_setup()
# sol_log_game_ct = log_game_ct.solver.start()

# assert np.allclose(sol_log_game_np["y"], sol_log_game_ct["y"])


# IPM:

# ipm_ct = IPM_ct(game)
# ipm_ct.solver_setup()
# sol_ipm_ct = ipm_ct.solver.start()

# ipm_sp = IPM_sp(game)
# ipm_sp.solver_setup()
# sol_ipm_sp = ipm_sp.solver.start()

# assert np.allclose(sol_ipm_ct["y"], sol_ipm_sp["y"])
# assert np.allclose(ipm_ct.H(ipm_ct.y0), ipm_sp.H(ipm_ct.y0))
# assert np.allclose(ipm_ct.H(y_rand), ipm_sp.H(y_rand))


# %% time Jacobian


"""
%timeit -n 1000 -r 10 qre_np.H(qre_np.y0)
%timeit -n 1000 -r 10 qre_np.H(qre_np.y0, dev=True)

%timeit -n 100 -r 10 qre_np.J(qre_np.y0)
%timeit -n 100 -r 10 qre_np.J(qre_np.y0, dev=True)


%timeit -n 1000 -r 10 qre_ct.H(qre_ct.y0)
%timeit -n 1000 -r 10 qre_ct.J(qre_ct.y0)
"""


# %% time solver


np.random.seed(42)

timer = HomotopyTimer()
# timer.timing()
