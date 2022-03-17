import dsgamesolver as dsg
import numpy as np

game = dsg.SGame.random_game(2,20,2, delta=np.random.random(20), seed=123)

sigma = game.random_strategy()
V = game.get_values(sigma)

def test_old():
    vv = np.random.random(size=V.shape)
    game.u_tilde(V)

def test_new():
    vv = np.random.random(size=V.shape)
    game.u_tilde_new(V)

%timeit test_old()
%timeit test_new()

from dsgamesolver.homotopy._qre_ct import u_tilde, u_tilde_new

%timeit u_tilde(game.payoffs, V, game.transitions)
%timeit u_tilde_new(game.payoffs, V, game.undiscounted_transitions, game.discount_factors)
%timeit u_tilde_newest(game.payoffs, V, game.undiscounted_transitions, game.discount_factors)

# %timeit np.allclose(game.u_tilde(V), game.u_tilde_new(V))

qre = dsg.QRE_ct(game)
qre.initialize(store_path=False)
qre.solver.verbose = 1
qre.solver.max_steps = 15
qre.solver.solve()


y = qre.solver.y
import timeit

t =timeit.timeit('qre.J(y)', globals=dict(y=y, qre=qre), number=100)
print(t)

t_2 =timeit.timeit('qre.J_2(y)', globals=dict(y=y, qre=qre), number=100)
print(t)
