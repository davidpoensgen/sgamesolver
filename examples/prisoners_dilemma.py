"""One-shot game + QRE: Prisoner's Dilemma"""


import numpy as np
import sgamesolver

payoff_matrix = np.array([[[2, 0],
                           [3, 1]],
                          [[2, 3],
                           [0, 1]]])

# payoff_matrix.shape = (2,2,2)
# indices: [player, action_0, action_1]

# define game
game = sgamesolver.SGame.one_shot_game(payoff_matrix=payoff_matrix)

# choose homotopy: quantal response equilibrium
homotopy = sgamesolver.homotopy.QRE(game=game)

# solve
homotopy.solver_setup()
homotopy.solve()

print(homotopy.equilibrium)
# +++++++++ state0 +++++++++
#                       a0    a1  
# player0 : v=1.00, σ=[0.000 1.000]
# player1 : v=1.00, σ=[0.000 1.000]
