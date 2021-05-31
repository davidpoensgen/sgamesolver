"""Testing stuff for development."""


import numpy as np

from dsgamesolver.sgame import SGame
from dsgamesolver.qre import QRE_np, QRE_ct
from dsgamesolver.tracing import Tracing_ct
from tests.random_game import create_random_game
from tests.timings import HomotopyTimer


# %% random game


num_s = 3           # number of states
num_p = 3           # number of players
num_a_max = 4       # maximum number of actions
num_a_min = 2       # minimum number of actions
delta_max = 0.95    # maximum discount factor
delta_min = 0.90    # minimum discount factor

a = 0               # payoffs in [a, a+b]
b = 1               # payoffs in [a, a+b]


# %% homotopy testing


u, phi, delta = create_random_game(num_s, num_p, num_a_min, num_a_max, delta_min, delta_max)
u = [a + b*u_s for u_s in u]

game = SGame(u, phi, delta)

# QRE:

qre_np = QRE_np(game)
qre_np.initialize()
sol_qre_np = qre_np.solver.solve()

qre_ct = QRE_ct(game)
qre_ct.initialize()
sol_qre_ct = qre_ct.solver.solve()

assert np.allclose(sol_qre_np["y"], sol_qre_ct["y"])

# Tracing:

# tracing_np = Tracing_np(game)
# tracing_np.initialize()
# sol_tracing_np = tracing_np.solver.solve()

tracing_ct = Tracing_ct(game)
tracing_ct.initialize()
sol_tracing_ct = tracing_ct.solver.solve()

# assert np.allclose(sol_tracing_np["y"], sol_tracing_ct["y"])


# %% time Jacobian


"""
import timeit

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
