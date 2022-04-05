import sgamesolver
import numpy as np

game = sgamesolver.SGame.random_game(5, 5, 5, seed=123)

qre = sgamesolver.homotopy.QRE(game)
qre.solver_setup()
qre.solve()
qre_np = sgamesolver.homotopy.QRE(game, implementation="numpy")
qre_np.solver_setup()
qre_np.solve()

tracing = sgamesolver.homotopy.LogTracing(game)
tracing.solver_setup()
tracing.solve()
tracing_np = sgamesolver.homotopy.LogTracing(game, implementation="numpy")
tracing_np.solver_setup()
tracing_np.solve()

tracing_f = sgamesolver.homotopy.LogTracing(game)
tracing_f.eta_fix = True
tracing_f.solver_setup()
tracing_f.solve()
tracing_f_np = sgamesolver.homotopy.LogTracing(game, implementation="numpy")
tracing_f_np.eta_fix = True
tracing_f_np.solver_setup()
tracing_f_np.solve()



loggame = sgamesolver.homotopy.LogGame(game)
loggame.solver_setup()
loggame.solve()
loggame_np = sgamesolver.homotopy.LogGame(game, implementation="numpy")
loggame_np.solver_setup()
loggame_np.solve()
