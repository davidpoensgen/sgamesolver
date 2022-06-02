"""Stochastic game + QRE: dynamic Rock-Paper-Scissors"""


import numpy as np
import sgamesolver

# dynamic variation of Rock-Paper-Scissors:
#   - if one player played Scissors in the previous round
#     (and the other player did NOT play Scissors),
#     that player will win a tie on Scissors in the next round
#   - if both players or neither player played Scissors in the previous round,
#     a normal round of Rock-Paper-Scissors is played

# 3 states:
#   0: both players or neither player have played Scissors in the previous round
#   1: player0 has played Scissors in the previous round and player1 has not
#   2: player1 has played Scissors in the previous round and player0 has not

payoff_matrices = [  # state0:
                   np.array([[[0,  -1,  1],
                              [1,   0, -1],
                              [-1,  1,  0]],
                             [[0,   1, -1],
                              [-1,  0,  1],
                              [1,  -1,  0]]]),
                     # state1:
                   np.array([[[0,  -1,  1],
                              [1,   0, -1],
                              [-1,  1,  1]],    # player0 wins tie on Scissors
                             [[0,   1, -1],
                              [-1,  0,  1],
                              [1,  -1, -1]]]),  # player1 loses tie on Scissors
                     # state2:
                   np.array([[[0,  -1,  1],
                              [1,  0, -1],
                              [-1,  1, -1]],    # player0 loses tie on Scissors
                             [[0,   1, -1],
                              [-1,  0,  1],
                              [1,  -1,  1]]])]  # player1 wins tie on Scissors

# transitions identical for each state
transition_matrices = [np.array([[[1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[1, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],
                                 [[0, 1, 0],
                                  [0, 1, 0],
                                  [1, 0, 0]]])] * 3

common_discount_factor = 0.95

# define game
game = sgamesolver.SGame(payoff_matrices=payoff_matrices, transition_matrices=transition_matrices,
                         discount_factors=common_discount_factor)

# choose homotopy: quantal response equilibrium
homotopy = sgamesolver.homotopy.QRE(game=game)

# solve
homotopy.solver_setup()
homotopy.solve()

print(homotopy.equilibrium)
# +++++++++ state0 +++++++++
#                        a0    a1    a2
# player0 : v= 0.00, σ=[0.369 0.298 0.333]
# player1 : v=-0.00, σ=[0.369 0.298 0.333]
# +++++++++ state1 +++++++++
#                        a0    a1    a2
# player0 : v= 0.11, σ=[0.257 0.409 0.333]
# player1 : v=-0.11, σ=[0.480 0.187 0.333]
# +++++++++ state2 +++++++++
#                        a0    a1    a2
# player0 : v=-0.11, σ=[0.480 0.187 0.333]
# player1 : v= 0.11, σ=[0.257 0.409 0.333]
