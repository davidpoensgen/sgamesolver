"""Get large games to work without overflows."""


import numpy as np

from dsgamesolver.sgame import SGame
from dsgamesolver.homotopies.qre import QRE_ct
from dsgamesolver.homotopies.tracing import Tracing_ct  # , TracingFixedEta_ct
from tests.random_game import create_random_game


rng = np.random.RandomState(42)


# %% random game


num_s = 10          # number of states
num_p = 5           # number of players
num_a_max = 20      # maximum number of actions
num_a_min = 20      # minimum number of actions
delta_max = 0.95    # maximum discount factor
delta_min = 0.95    # minimum discount factor

a = 0               # u in [a, a+b]
b = 1               # u in [a, a+b]

u, phi, delta = create_random_game(num_s, num_p, num_a_min, num_a_max, delta_min, delta_max, rng=rng)
u = [a + b*u_s for u_s in u]

game = SGame(u, phi, delta)


# %% QRE


qre = QRE_ct(game)
qre.solver_setup()
qre.solver.verbose = 2
sol_qre = qre.solver.solve()


# %% tracing save


tracing = Tracing_ct(game)
tracing.solver_setup()
tracing.solver.verbose = 2
tracing.solver.max_steps = 9000
sol_tracing = tracing.solver.solve()

tracing.solver.return_to_step(7680)
tracing.solver.max_steps = 7688
tracing.solver.save_file('tracing_failing_Jacobian.json')
