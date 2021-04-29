from homcont.homclass import HomCont
from homcont.sgame import sGame
import homcont.homclass
import homcont.homotopy

import numpy as np
# from dsGameSolver.gameClass import dsGame

# random game

num_s = 3           # number of states
num_p = 3           # number of players
num_a_max = 4       # maximum number of actions
num_a_min = 3       # minimum number of actions
delta_max = 0.95    # maximum discount factor
delta_min = 0.90    # minimum discount factor

# np.random.seed(0)
nums_a = np.random.randint(low=num_a_min, high=num_a_max+1, size=(num_s,num_p), dtype=np.int32)

payoffMatrices = [np.random.random((num_p, *nums_a[s,:])) for s in range(num_s)]

transitionMatrices = [np.random.exponential(scale=1, size=(*nums_a[s,:], num_s)) for s in range(num_s)]
for s in range(num_s):
    for index, value in np.ndenumerate(np.sum(transitionMatrices[s], axis=-1)):
        transitionMatrices[s][index] *= 1/value
discountFactors = np.random.uniform(low=delta_min, high=delta_max, size=num_p)


game = sGame(payoffMatrices, transitionMatrices, discountFactors)

qre = homcont.homotopy.QRE_np(game)

qre.solver_setup(target_lambda=10)

qre.solver.verbose =3
qre.solver.solve()