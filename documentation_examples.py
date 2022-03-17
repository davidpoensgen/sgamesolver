import dsgamesolver as dsg

my_first_game = dsg.SGame.random_game(num_states=6, num_players=6, num_actions=6, seed=1)

qre_homotopy = dsg.QRE_ct(game=my_first_game)
qre_homotopy.initialize()
# qre_homotopy.solver.verbose = 2
qre_homotopy.solver.solve()
