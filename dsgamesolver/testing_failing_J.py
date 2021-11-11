"""Get large games to work without overflows."""


import numpy as np

from dsgamesolver.sgame import SGame
from dsgamesolver.tracing import Tracing_ct
from tests.random_game import create_random_game


np.random.seed(42)


# %% random game


num_s = 10          # number of states
num_p = 5           # number of players
num_a_max = 10      # maximum number of actions
num_a_min = 10      # minimum number of actions
delta_max = 0.95    # maximum discount factor
delta_min = 0.95    # minimum discount factor

a = 0               # payoffs in [a, a+b]
b = 1               # payoffs in [a, a+b]

u, phi, delta = create_random_game(num_s, num_p, num_a_min, num_a_max, delta_min, delta_max)
u = [a + b*u_s for u_s in u]

game = SGame(u, phi, delta)


# %% recover tracing


tracing = Tracing_ct(game)
tracing.initialize()

tracing.solver.load_file('tracing_failing_Jacobian.json')
tracing.solver.verbose = 2
tracing.solver.max_steps = 9000
tracing.solver.solve()
