import sgamesolver
from sgamesolver.homotopy._tracing import Tracing_Cache
import numpy as np
import matplotlib.pyplot as plt

game = sgamesolver.SGame.random_game(6, 6, 6, seed=123)

tracing_cache = Tracing_Cache(game)
tracing_cache.solver_setup()
tracing_cache.solver.max_steps = 66
tracing_cache.solve()

tracing = sgamesolver.homotopy.Tracing(game)
tracing.solver_setup()
tracing.solver.max_steps = 66
tracing.solve()
