import pandas as pd
import numpy as np
import sgamesolver
from datetime import datetime

name = "test3"
Homotopy = sgamesolver.homotopy.LogTracing

default_count = 100
counts = {
    (20, 5, 10): 3,
    (10, 5, 10): 3,
}

def make_file(load=True):
    if load:
        try:
            timings = pd.read_csv(f'{name}.csv', index_col='index')
        except FileNotFoundError:
            load = False
    if not load:
        timings = pd.DataFrame(columns=['S', 'I', 'A', 'no', 'seed', 'run',
                                        'date', 'success', 'seconds', 'steps', 'exception'])

    for S in [1, 2, 5, 10, 20]:
        for I in [1, 2, 3, 4, 5]:
            for A in [2, 5, 10]:
                count = counts.get((S, I, A), default=default_count)
                for no in range(count):
                    if not (timings[timings.columns[:4]] == (S, I, A, no)).all(1).any():
                        timings.loc[len(timings)] = (S, I, A, no, int(f'{S}{I}{A}{no}'), False,
                                                     "-", False, -1., -1, "-")

    timings.sort_values(['S', 'I', 'A', 'no'], inplace=True)
    timings.to_csv(f'{name}.csv', index_label='index')


def run_file(count=1000):
    timings = pd.read_csv(f'{name}.csv', index_col='index')

    for index, row in timings[timings.run == False].head(count).iterrows():
        S, I, A, no, seed = row[['S', 'I', 'A', 'no', 'seed']]
        print(S, I, A, no)
        game = sgamesolver.SGame.random_game(S, I, A, delta=.95, seed=seed)

        tracing = Homotopy(game)
        timings.loc[index, ['run', 'date']] = (True, datetime.now().strftime("%Y-%m-%d, %H:%M"))
        timings.to_csv(f'{name}.csv', index_label='index')

        try:
            tracing.solver_setup()
            tracing.solver.verbose = 0
            tracing.solver.max_steps = 1e4
            result = tracing.solver.start()
            timings.loc[index, ['success', 'seconds', 'steps']] = (result['success'], result['time'], result['steps'])

        except KeyboardInterrupt:
            timings.loc[index, ['exception']] = ('keyboard interrupt',)
            timings.to_csv(f'{name}.csv', index=False)
            raise
        except Exception as exception:
            timings.loc[index, ['exception']] = (str(exception),)
            current_max = timings.loc[(timings['S'] == S) & (timings['I'] == I) & (timings['A'] == A)]['no'].max()
            print(exception, f'adding {S, I, A, current_max + 1}')
            timings.loc[len(timings)] = (S, I, A, current_max + 1, int(f'{S}{I}{A}{current_max + 1}'), False,
                                         "-", False, -1., -1, "-")

        timings.to_csv(f'{name}.csv', index_label='index')

    timings.sort_values(['S', 'I', 'A', 'no'], inplace=True)
    timings.to_csv(f'{name}.csv', index_label='index')

