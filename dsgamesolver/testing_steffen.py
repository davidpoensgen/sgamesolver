"""Testing stuff for development."""

import numpy as np

from dsgamesolver.sgame import SGame
from dsgamesolver.qre import QRE_np, QRE_ct
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

qre_np = QRE_np(game)
qre_np.initialize()
# sol_np = qre_np.solver.solve()

qre_ct = QRE_ct(game)
qre_ct.initialize()
# sol_ct = qre_ct.solver.solve()


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
