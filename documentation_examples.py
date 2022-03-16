import dsgamesolver as dsg

my_first_game = dsg.SGame.random_game(num_states=4, num_players=4, num_actions=4, seed=1)

qre_homotopy = dsg.QRE_np(game=my_first_game)
qre_homotopy.initialize()
# qre_homotopy.solver.verbose = 2
qre_homotopy.solver.solve()
