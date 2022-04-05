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

tracing_fix = sgamesolver.homotopy.LogTracingFixedEta(game)
tracing_fix.solver_setup()
tracing_fix.solve()
tracing_fix_np = sgamesolver.homotopy.LogTracingFixedEta(game, implementation="numpy")
tracing_fix_np.solver_setup()
tracing_fix_np.solve()

loggame = sgamesolver.homotopy.LogGame(game)
loggame.solver_setup()
loggame.solve()
loggame_np = sgamesolver.homotopy.LogGame(game, implementation="numpy")
loggame_np.solver_setup()
loggame_np.solve()
