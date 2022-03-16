import dsgamesolver as dsg

game = dsg.SGame.random_game(6, 6, 6, seed=1)

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
