import numpy as np

from dsgamesolver.sgame import SGame
from dsgamesolver.qre import QRE_np
#from tests.timings import HomotopyTimer

# random game

num_s = 3           # number of states
num_p = 3           # number of players
num_a_max = 4       # maximum number of actions
num_a_min = 2       # minimum number of actions
delta_max = 0.95    # maximum discount factor
delta_min = 0.90    # minimum discount factor

# np.random.seed(0)
nums_a = np.random.randint(low=num_a_min, high=num_a_max + 1, size=(num_s, num_p), dtype=np.int32)

a = 0
b = 1
payoffMatrices = [a + b * np.random.random((num_p, *nums_a[s, :])) for s in range(num_s)]

# stop normalization for testing
A = (0,) * num_p
payoffMatrices[0][(0,) + A] = 0
payoffMatrices[1][(0,) + A] = 1

transitionMatrices = [np.random.exponential(scale=1, size=(*nums_a[s, :], num_s)) for s in range(num_s)]
for s in range(num_s):
    for index, value in np.ndenumerate(np.sum(transitionMatrices[s], axis=-1)):
        transitionMatrices[s][index] *= 1 / value
discountFactors = np.random.uniform(low=delta_min, high=delta_max, size=num_p)


game = SGame(payoffMatrices, transitionMatrices, discountFactors)
si = game.centroid_strategy()

qre = QRE_np(game)
qre.initialize()

import timeit

# timeit.timeit("qre.H(qre.y0, old=True)", number=100000, globals=globals())
# timeit.timeit("qre.H(qre.y0, old=False)", number=100000, globals=globals())
#
# %timeit qre.H(qre.y0, old=True)
# %timeit qre.H(qre.y0, old=False)

qre.solver.verbose = 2
# qre.solver.max_steps = 50
sol = qre.solver.solve()
print(sol)

# qre.solver.return_to_step(5)
# sol2 = qre.solver.solve()

# import timeit
# %timeit -n 10 -r 10 qre.J(qre.y0, old=True)
# %timeit -n 10 -r 10 qre.J(qre.y0, old=False)


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
