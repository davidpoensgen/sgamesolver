import sgamesolver
import numpy as np
import matplotlib.pyplot as plt

game = sgamesolver.SGame.random_game(6, 6, 6, seed=123)
# game = sgamesolver.SGame.random_game(5, 5, 5, seed=123)

tracing = sgamesolver.homotopy.LogTracing(game)
tracing.solver_setup()
tracing.solver.max_steps = 100
tracing.solve()

tracing = sgamesolver.homotopy.LogTracing(game)
tracing.solver_setup()
tracing.cache = None
tracing.solver.max_steps = 100
tracing.solve()

tracing = sgamesolver.homotopy.LogTracingFixedEta(game)
tracing.solver_setup()
tracing.solver.max_steps = 100
tracing.solve()

tracing = sgamesolver.homotopy.LogTracingFixedEta(game)
tracing.solver_setup()
tracing.cache = None
tracing.solver.max_steps = 100
tracing.solve()

# 666, 123, 100 steps: 1:28 -> 1:09