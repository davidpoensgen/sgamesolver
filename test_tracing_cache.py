import sgamesolver
from sgamesolver.homotopy._tracing import Tracing_Cache
import numpy as np
import matplotlib.pyplot as plt

# game = sgamesolver.SGame.random_game(6, 6, 6, seed=123)
game = sgamesolver.SGame.random_game(4, 4, 4, seed=123)

tracing_cache = Tracing_Cache(game)
tracing_cache.solver_setup()
# tracing_cache.ct_cache = None
tracing_cache.solver.max_steps = 1000
tracing_cache.solve()

tracing = sgamesolver.homotopy.Tracing(game)
tracing.solver_setup()
tracing.solver.max_steps = 100
tracing.solve()

# 666, 123, 100 steps: 1:28 -> 1:09