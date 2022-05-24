"""Markov decision problem + QRE: optimal growth model (Judd and Solnick, 1994)"""


import numpy as np
import matplotlib.pyplot as plt
import sgamesolver

# reference: Judd and Solnick (1994)

alpha = 0.25  # productivity parameter
beta = 0.95   # discount factor
gamma = 2     # relative risk aversion
delta = 0     # capital depreciation

# utility function
def u(c):
    return (c**(1-gamma)) / (1-gamma)

# production function
def f(k):
    return ((1-beta)/(alpha*beta)) * k**alpha

# state space K[a] (capital stock)
k_min, k_max, k_step = 0.4, 1.6, 0.1
num_k = int(1 + (k_max-k_min) / k_step)
K = np.linspace(k_min, k_max, num_k)

# action space C[s,a] (consumption)
#   - state-dependent to let c_{t}(k_{t}) -> k_{t+1}
#     with restriction c_{t} >= 0
C = np.nan * np.ones((num_k, num_k))
for s in range(num_k):
    C[s] = (1-delta)*K[s] + f(K[s]) - K
C[C < 0] = np.nan

# numbers of actions in each state
nums_a = np.zeros(num_k, dtype=np.int32)
for s in range(num_k):
    nums_a[s] = len(C[s][~np.isnan(C[s])])

payoff_matrices = []
for s in range(num_k):
    payoff_matrix = np.nan * np.ones((1, nums_a[s]))
    for a in range(nums_a[s]):
        payoff_matrix[0, a] = u(C[s, a])
    payoff_matrices.append(payoff_matrix)

transition_matrices = []
for s in range(num_k):
    transition_matrix = np.zeros((nums_a[s], num_k))
    for a in range(nums_a[s]):
        for s_ in range(num_k):
            if a == s_:
                transition_matrix[a, s_] = 1
    transition_matrices.append(transition_matrix)

discount_factor = 0.95

# define game
game = sgamesolver.SGame(payoff_matrices=payoff_matrices, transition_matrices=transition_matrices,
                         discount_factors=discount_factor)

# choose homotopy: quantal response equilibrium
homotopy = sgamesolver.homotopy.QRE(game=game)

# solve
# TODO: LogTracing fails
# TODO: QRE takes ~15 minutes
"""
homotopy.solver_setup()
homotopy.solve()

# equilibrium
assert homotopy.equilibrium is not None
policies = np.nan * np.ones(num_k)
values = np.nan * np.ones(num_k)
for s in range(num_k):
    # get optimal actions from pure-strategy equilibrium
    a = np.where((np.round(homotopy.equilibrium.strategies[s, 0]) == 1))[0]
    policies[s] = C[s, a]
    values[s] = homotopy.equilibrium.values[s, 0]

# plot
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121)
ax1.set_title('Policy Function')
ax1.set_xlabel(r'capital stock $k_{t}$')
ax1.set_ylabel(r'consumption $c_{t}$')
ax1.plot(K, policies)
ax1.grid()
ax2 = fig.add_subplot(122)
ax2.set_title('Value Function')
ax2.set_xlabel(r'capital stock $k_{t}$')
ax2.set_ylabel(r'present value of utility $V(k_{t})$')
ax2.plot(K, values)
ax2.grid()
plt.show()
"""
