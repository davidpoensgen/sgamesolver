"""Tutorial: First steps with sGameSolver."""


import numpy as np

from sgamesolver import SGame
from sgamesolver.homotopy import QRE

# payoff matrix: https://en.wikipedia.org/wiki/Prisoner%27s_dilemma
payoff_matrix = np.array([[[-1, -3],
                           [+0, -2]],
                          [[-1,  0],
                           [-3, -2]]])

# payoff_matrix.shape = (2,2,2)
# indices: [player, action_1, action_2]

# define game
game = SGame.one_shot_game(payoff_matrix=payoff_matrix)

# choose homotopy: quantal response equilibrium
homotopy = QRE(game=game)

# solve
homotopy.solver_setup()
homotopy.solve()

print(homotopy.equilibrium)
# +++++++ state0 +++++++
# player0: v=-2.00, s=[0.000 1.000]
# player1: v=-2.00, s=[0.000 1.000]
