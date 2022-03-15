import dsgamesolver as dsg

game = dsg.SGame.random_game(4, 4, 4)

qre = dsg.QRE_ct(game)
qre.initialize()
qre.solver.solve()

