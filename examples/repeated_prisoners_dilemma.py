"""Repeated game + QRE: Prisoner's Dilemma"""


import numpy as np
import sgamesolver

# 4 states:
#   0: neither player has defected so far
#   1: player0 has defected in the past, but not player1
#   2: player1 has defected in the past, but not player0
#   3: both players have defected previously

# list[payoff_matrix]: one payoff matrix for each state (identical for each state)
#   payoff_matrix.shape = (2,2,2)
#   indices: [player, action_0, action_1]
payoff_matrices = [np.array([[[2, 0],
                              [3, 1]],
                             [[2, 3],
                              [0, 1]]])] * 4

# list[transition_matrix]: one transition matrix for each state
#   transition_matrix.shape = (2,2,2)
#   indices: [action0, action1, state_to]
transition_matrices = [  # state0:
                       np.array([[[1, 0, 0, 0],     # (C,C) -> state0
                                  [0, 0, 1, 0]],    # (C,D) -> state2
                                 [[0, 1, 0, 0],     # (D,C) -> state1
                                  [0, 0, 0, 1]]]),  # (D,D) -> state3
                         # state1:
                       np.array([[[0, 1, 0, 0],     # (C,C) -> state1
                                  [0, 0, 0, 1]],    # (C,D) -> state3
                                 [[0, 1, 0, 0],     # (D,C) -> state1
                                  [0, 0, 0, 1]]]),  # (D,D) -> state3
                         # state2:
                       np.array([[[0, 0, 1, 0],     # (C,C) -> state2
                                  [0, 0, 1, 0]],    # (C,D) -> state2
                                 [[0, 0, 0, 1],     # (D,C) -> state3
                                  [0, 0, 0, 1]]]),  # (D,D) -> state3
                         # state3:
                       np.array([[[0, 0, 0, 1],     # (C,C) -> state3
                                  [0, 0, 0, 1]],    # (C,D) -> state3
                                 [[0, 0, 0, 1],     # (D,C) -> state3
                                  [0, 0, 0, 1]]])]  # (D,D) -> state3

common_discount_factor = 0.99

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
#                         a0    a1
# player0 : v=100.00, σ=[0.000 1.000]
# player1 : v=100.00, σ=[0.000 1.000]
# +++++++++ state1 +++++++++
#                         a0    a1
# player0 : v=100.00, σ=[0.000 1.000]
# player1 : v=100.00, σ=[0.000 1.000]
# +++++++++ state2 +++++++++
#                         a0    a1
# player0 : v=100.00, σ=[0.000 1.000]
# player1 : v=100.00, σ=[0.000 1.000]
# +++++++++ state3 +++++++++
#                         a0    a1
# player0 : v=100.00, σ=[0.000 1.000]
# player1 : v=100.00, σ=[0.000 1.000]
