import numpy as np

from ..dsgamesolver.sgame import SGame
from dsgamesolver.homotopies.qre import QRE_np, QRE_ct
from tests.timings import HomotopyTimer

# random game

num_s = 3           # number of states
num_p = 3           # number of players
num_a_max = 4       # maximum number of actions
num_a_min = 2       # minimum number of actions
delta_max = 0.95    # maximum discount factor
delta_min = 0.90    # minimum discount factor


game = SGame.random_game(num_s, num_p, num_actions=(2, 4), delta=0.95)
si = game.centroid_strategy()

qre = QRE_np(game)
qre.initialize()

"""
import timeit

%timeit.timeit("qre.H(qre.y0, dev=True)", number=100000, globals=globals())
%timeit.timeit("qre.H(qre.y0, dev=False)", number=100000, globals=globals())

%timeit qre.H(qre.y0, dev=True)
%timeit qre.H(qre.y0, dev=False)
"""

qre.solver.verbose = 2
# qre.solver.max_steps = 50
sol = qre.solver.solve()
print(sol)

# qre.solver.return_to_step(5)
# sol2 = qre.solver.solve()

# import timeit
# %timeit -n 10 -r 10 qre.J(qre.y0, dev=True)
# %timeit -n 10 -r 10 qre.J(qre.y0, dev=False)


# cython test:
qre2 = QRE_ct(game)
qre2.initialize()
sol2 = qre2.solver.solve()


# timings:
timer = HomotopyTimer('QRE_np')
timer.timing()


# numba test:
# qre3 = QRE_np_numba(game)
# qre3.initialize()
# sol3 = qre3.solver.solve()
