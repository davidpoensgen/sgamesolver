"""One-shot game + log tracing: Stag Hunt"""


import numpy as np
import sgamesolver

payoff_matrix = np.array([[[5, 0],
                           [4, 2]],
                          [[5, 4],
                           [0, 2]]])

# payoff_matrix.shape = (2,2,2)
# indices: [player, action_0, action_1]

# define game
game = sgamesolver.SGame.one_shot_game(payoff_matrix=payoff_matrix)

# choose homotopy: logarithmic tracing procedure
homotopy = sgamesolver.homotopy.LogTracing(game=game)

# solve
homotopy.solver_setup()
homotopy.solve()

print(homotopy.equilibrium)
# +++++++++ state0 +++++++++
#                       a0    a1
# player0 : v=2.00, σ=[0.000 1.000]
# player1 : v=2.00, σ=[0.000 1.000]
