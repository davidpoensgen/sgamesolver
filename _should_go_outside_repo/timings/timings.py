"""Perform systematic timings of each homotopy to evaluate performance."""

# TODO: changed homotopy.tracking_parameters['normal'] to homotopy.default_parameters and similarly
# 'robust' to robust_parameters; not updating this file as I assume its outdated anyways
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

import sgamesolver


HOMOTOPIES = {
    'QRE': sgamesolver.homotopy.QRE,
    'LogTracing': sgamesolver.homotopy.LogTracing,
}


# %% class to perform timings


class HomotopyTimer:
    """Class to run timings on homotopy implementations."""

    def __init__(self, filename: str = 'timings') -> None:
        self.filename = filename

    def run(self, verbose: bool = True) -> HomotopyTimer:
        """
        Run all timings using specs from csv file.
        If csv file does not exist, make new empty one and run all timings.
        """

        timings = self.load_file()
        timings_to_be_done = timings.loc[~timings['done']]

        tic = datetime.now()

        for idx, row in timings_to_be_done.iterrows():
            homotopy, num_s, num_p, num_a, rep, seed = row[['homotopy', 'states', 'players', 'actions', 'rep', 'seed']]

            if verbose:
                toc = datetime.now()
                sys.stdout.write(f"\r{homotopy} - (S,I,A) = ({num_s},{num_p},{num_a}): Starting timing #{rep+1},"
                                 f" elapsed = {str(toc-tic).split('.')[0]}")
                sys.stdout.flush()

            timings.loc[idx, 'timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            timings.loc[idx, 'success'] = False
            timings.loc[idx, 'seconds'] = np.nan
            timings.loc[idx, 'steps'] = np.nan
            timings.loc[idx, 'path_len'] = np.nan
            timings.loc[idx, 'hom_par'] = np.nan
            timings.loc[idx, 'failure_reason'] = None
            self.write_to_file(frame=timings)

            try:
                sol = self.solve_random_game(homotopy=homotopy, num_s=num_s, num_p=num_p, num_a=num_a, seed=seed)
                timings.loc[idx, 'done'] = True
                timings.loc[idx, 'success'] = sol['success']
                timings.loc[idx, 'seconds'] = sol['seconds']
                timings.loc[idx, 'steps'] = sol['steps']
                timings.loc[idx, 'path_len'] = sol['path_len']
                timings.loc[idx, 'hom_par'] = sol['hom_par']
                timings.loc[idx, 'failure_reason'] = sol['failure_reason']
                timings.loc[idx, 'exception'] = 'none'
                self.write_to_file(frame=timings)

            except Exception as exception:
                timings.loc[idx, 'done'] = True
                timings.loc[idx, 'exception'] = str(exception)
                self.write_to_file(frame=timings)

            except KeyboardInterrupt:
                raise

        overall_success_rate = timings.loc[timings_to_be_done.index, 'success'].mean()
        print(f"\nOverall success rate = {100*overall_success_rate:.2f} %")

        return self

    def make_latex_tables(self) -> dict[str, str]:
        """
        Generate LaTeX code for timings tables.
        Use timings from file.
        Return one table for each homotopy in file.
        """

        timings = self.load_file()
        finished_timings = timings.loc[timings['done']]

        latex_tables = dict()

        for homotopy in np.unique(finished_timings['homotopy']):
            homotopy_timings = finished_timings.loc[finished_timings['homotopy'] == homotopy]

            nums_s = np.unique(homotopy_timings['states'])
            nums_p = np.unique(homotopy_timings['players'])
            nums_a = np.unique(homotopy_timings['actions'])

            seconds = dict()
            for num_s in nums_s:
                for num_p in nums_p:
                    for num_a in nums_a:
                        sub_df = homotopy_timings.loc[homotopy_timings['success']
                                                      & (homotopy_timings['states'] == num_s)
                                                      & (homotopy_timings['players'] == num_p)
                                                      & (homotopy_timings['actions'] == num_a)]

                        seconds[(num_s, num_p, num_a)] = {'mean': sub_df['seconds'].mean(),
                                                          'std': sub_df['seconds'].std()}

            code_lines = ['\\begin{tabular}{lc' + "c".join(['rl']*len(nums_p)) + '}',
                          '\\hline',
                          '$|A|$ / $|I|$ &  & ' + ' &  & '.join(str(num_p) + ' & ' for num_p in nums_p) + ' \\\\',
                          '\\hline']

            for k, num_s in enumerate(nums_s):
                if k > 0:
                    code_lines.append(' &  & ' + ' &  & '.join([' & ']*len(nums_p)) + ' \\\\')

                code_lines.append('$|S|$ = ' + str(num_s) + ' &  & ' + ''.join([' &  & ']*len(nums_p)) + ' \\\\')

                for num_a in nums_a:
                    code_lines.append(str(num_a) + ' &  & '
                                      + ' &  & '.join(f"{seconds[(num_s, num_p, num_a)]['mean']:0.2f}"
                                                      + ' & \\hspace{-0.6em} ('
                                                      + f"{seconds[(num_s, num_p, num_a)]['std']:0.2f})"
                                                      for num_p in nums_p)
                                      + ' \\\\')

            code_lines.append('\\hline')
            code_lines.append('\\end{tabular}')

            latex_table = '\n'.join(code_lines)
            latex_tables[homotopy] = latex_table

            print(f"\n{homotopy}:\n\n{latex_table}\n")

        return latex_table

    def plot_spec_timings(self, homotopy: str, num_s: int, num_p: int, num_a: int) -> Optional[Any]:
        """Plot histograms of timing results for given homotopy and game specs (S,I,A)."""

        timings = self.load_file()
        timings_for_spec = timings.loc[timings['done'] & (timings['homotopy'] == homotopy)
                                       & (timings['states'] == num_s) & (timings['players'] == num_p)
                                       & (timings['actions'] == num_a)]

        if timings_for_spec.empty:
            print(f"No timings for {homotopy} and (S,I,A) = ({num_s},{num_p},{num_a}).")
            return

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Requires python package 'matplotlib'.")

        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(221)
        ax1.set_title('Time')
        ax1.set_xlabel('time [sec]')
        ax1.hist(timings_for_spec['seconds'])
        ax1.grid()
        ax2 = fig.add_subplot(222)
        ax2.set_title('Steps')
        ax2.set_xlabel(r'$n$')
        ax2.hist(timings_for_spec['steps'])
        ax2.grid()
        ax3 = fig.add_subplot(223)
        ax3.set_title('Path Length')
        ax3.set_xlabel(r'$s$')
        ax3.hist(timings_for_spec['path_len'])
        ax3.grid()
        ax4 = fig.add_subplot(224)
        ax4.set_title('Homotopy Parameter')
        ax4.set_xlabel(r'$t$')
        ax4.hist(timings_for_spec['hom_par'])
        ax4.grid()
        plt.tight_layout()
        plt.show()

        return fig

    def make_file(self, homotopy: str = 'QRE', num_states: list[int] = [1, 2, 5, 10, 20],
                  num_players: list[int] = [1, 2, 3, 4, 5], num_actions: list[int] = [2, 5, 10], num_reps: int = 100,
                  special_num_reps: dict[tuple[int, int, int], int] = {(20, 5, 10): 3, (10, 5, 10): 5, (5, 5, 10): 20},
                  override: bool = False) -> pd.DataFrame:
        """Make empty csv file for timing results."""

        timings = pd.DataFrame([{'homotopy': homotopy,
                                 'states': num_s,
                                 'players': num_p,
                                 'actions': num_a,
                                 'rep': rep,
                                 'seed': self.generate_seed(num_s=num_s, num_p=num_p, num_a=num_a, rep=rep),
                                 'done': False,
                                 'timestamp': None,
                                 'success': False,
                                 'seconds': np.nan,
                                 'steps': np.nan,
                                 'path_len': np.nan,
                                 'hom_par': np.nan,
                                 'failure_reason': None,
                                 'exception': None}
                                for num_s in num_states for num_p in num_players for num_a in num_actions
                                for rep in range(special_num_reps.get((num_s, num_p, num_a), num_reps))])

        self.write_to_file(frame=timings, override=override)
        return timings

    def load_file(self) -> pd.DataFrame:
        """Load csv file for timing results. If file does not exist, make new empty one."""

        try:
            timings = self.read_from_file()

        except FileNotFoundError:
            return self.make_file()

        except Exception:
            raise

        # check loaded file:

        columns = ['homotopy', 'states', 'players', 'actions', 'rep', 'seed', 'done', 'timestamp',
                   'success', 'seconds', 'steps', 'path_len', 'hom_par', 'failure_reason', 'exception']

        # required columns
        if timings[['homotopy', 'states', 'players', 'actions']].isna().any().any():
            raise ValueError("Columns ['homotopy', 'states', 'players', 'actions'] must not contain NaN.")

        for homotopy in np.unique(timings['homotopy']):
            if homotopy not in HOMOTOPIES.keys():
                raise ValueError(f"Homotopy '{homotopy}' not in {list(HOMOTOPIES.keys())}.")

        # fill missing columns and sort
        for col in columns:
            if col not in timings.columns:
                timings[col] = np.nan

        timings = timings[columns]

        # fill rep, seed and done
        reps: dict[tuple, set] = dict()

        for idx, row in timings.iterrows():
            num_s, num_p, num_a = row[['states', 'players', 'actions']]
            dim = (num_s, num_p, num_a)

            if dim not in reps.keys():
                reps[dim] = set()

            if pd.isna(row['rep']):
                smallest_nonexisting_id = next(iter(k for k in range(len(timings)) if k not in reps[dim]))
                timings.loc[idx, 'rep'] = smallest_nonexisting_id

            if timings.loc[idx, 'rep'] in reps[dim]:
                raise ValueError("Column 'reps' is required to be unique for each specification (S,I,A).")

            reps[dim].add(timings.loc[idx, 'rep'])

            if pd.isna(row['seed']):
                timings.loc[idx, 'seed'] = self.generate_seed(s=num_s, num_p=num_p, num_a=num_a,
                                                              rep=timings.loc[idx, 'rep'])

            if pd.isna(row['done']):
                timings.loc[idx, 'done'] = False

        return timings.sort_values(['homotopy', 'states', 'players', 'actions', 'rep']).reset_index(drop=True)

    def solve_random_game(self, homotopy: str, num_s: int, num_p: int, num_a: int, seed: int,
                          delta: float = 0.95, max_steps: int = 1e5, method: str = "normal") -> dict:

        game = sgamesolver.SGame.random_game(num_states=num_s, num_players=num_p, num_actions=num_a,
                                             delta=delta, seed=seed)

        homotopy = HOMOTOPIES[homotopy](game=game)

        if method not in homotopy.tracking_parameters.keys():
            raise ValueError(f"Tracking method '{method}' not in {list(homotopy.tracking_parameters.keys())}.")

        parameters = homotopy.tracking_parameters[method]

        homotopy.solver_setup()
        homotopy.solver.max_steps = max_steps
        homotopy.solver.set_parameters(parameters)
        homotopy.solver.verbose = 0
        homotopy.solver.store_path = False

        sol = homotopy.solver.start()

        return {'success': sol['success'],
                'seconds': sol['time'],
                'steps': sol['steps'],
                'path_len': sol['s'],
                'hom_par': sol['y'][-1],
                'failure_reason': 'none' if sol['failure reason'] is None else sol['failure reason']}

    def write_to_file(self, frame: pd.DataFrame, override: bool = True) -> None:
        if not override and os.path.isfile(f"{self.filename}.csv"):
            answer = input(f"'{self.filename}.csv' already exists. Overwrite [y/N]?")

            if answer == '' or answer[0].lower() != 'y':
                print(f"Canceled saving '{self.filename}.csv'.")
                return

        frame.to_csv(f"{self.filename}.csv", index_label='index')

    def read_from_file(self) -> pd.DataFrame:
        return pd.read_csv(f"{self.filename}.csv", index_col='index')

    @staticmethod
    def generate_seed(num_s: int, num_p: int, num_a: int, rep: int) -> int:
        return f"{num_s:02d}{num_p:02d}{num_a:02d}{rep:04d}"


# %% test and run


if __name__ == '__main__':

    # small test run:

    timer = HomotopyTimer()
    timer.make_file(num_states=[5], num_players=[5], num_actions=[5], num_reps=20, override=True)

    timer.run()

    timer.plot_spec_timings(homotopy='QRE', num_s=5, num_p=5, num_a=5)
    timer.make_latex_tables()

    # full run:

    # timer = HomotopyTimer().run()
