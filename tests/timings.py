﻿"""Perform systematic timings of each homotopy to evaluate performance."""

import datetime
import itertools
import sys

import numpy as np
import matplotlib.pyplot as plt

from dsgamesolver.sgame import SGame
from dsgamesolver.qre import QRE_np, QRE_ct
# from dsgamesolver.tracing import Tracing_np, Tracing_ct
from tests.random_game import create_random_game


HOMOTOPIES = {
    'QRE_np': QRE_np,
    'QRE_ct': QRE_ct,
    # 'Tracing_np': Tracing_np,
    # 'Tracing_ct': Tracing_cp,
}


# %% class to perform timings


class HomotopyTimer:
    """Class to run timings on homotopy implementations."""

    def __init__(self, homotopy: str = 'QRE_np', max_steps: int = 1e4, tracking: str = 'normal') -> None:
        self.homotopy = homotopy
        self.Homotopy = HOMOTOPIES[self.homotopy]
        self.game_memory = SGameMemory()
        self.max_steps = max_steps
        self.tracking = tracking
        self.last_timing = None

    def batch_timings(self, nums_s: list = [2, 3, 4, 5], nums_p: list = [2, 3, 4, 5], nums_a: list = [2, 3, 4, 5],
                      delta: float = 0.95, reps: int = 100,
                      verbose: bool = True, printout: bool = True) -> np.ndarray:

        timings = {}
        tic = datetime.datetime.now()

        for (num_s, num_p, num_a) in itertools.product(nums_s, nums_p, nums_a):
            timings[(num_s, num_p, num_a)] = self.timing(num_s, num_p, num_a, delta, reps, verbose, False, False)

            if verbose:
                print(f"Time elapsed = {datetime.datetime.now() - tic}")

        if verbose:
            success_rate = np.mean([timing['success'] for spec, timing in timings.items()])
            print(f"\nOverall success rate: {success_rate:0.1f} %")

        if printout:
            self.print_latex_table(timings)

        return timings

    def timing(self, num_s: int = 3, num_p: int = 3, num_a: int = 3, delta: float = 0.95,
               reps: int = 10, verbose: bool = True, printout: bool = True, plot: bool = True) -> np.ndarray:

        result = self.batch_solve_random_games(num_s, num_p, num_a, delta, reps, verbose)
        timing = self.compute_timing(result)

        if printout:
            self.print_timing(timing)

        if plot:
            self.plot_result(result)

        return timing

    def batch_solve_random_games(self, num_s: int = 3, num_p: int = 3, num_a: int = 3, delta: float = 0.95,
                                 reps: int = 10, verbose: bool = True) -> np.ndarray:

        if verbose:
            print(f"\n(S,P,A) = ({num_s},{num_p},{num_a})")

        result = np.nan * np.ones((reps, 5))

        for rep in range(reps):
            sol = self.solve_random_game(num_s, num_p, num_a, delta)

            if sol['success']:
                result[rep] = sol['success'], sol['seconds'], sol['steps'], sol['path'], sol['parameter']

            if verbose:
                sys.stdout.write(f"\rTiming {rep+1} / {reps}. Avg time = {np.nanmean(result[:rep+1,1]):0.2f} sec   ")
                sys.stdout.flush()

        return result

    def solve_random_game(self, num_s: int = 3, num_p: int = 3, num_a: int = 3, delta: float = 0.95) -> dict:
        """Create random game with common number of actions and common discount factor, and solve it."""

        game = SGame(*create_random_game(num_s, num_p, num_a, num_a, delta, delta))

        self.game_memory.replace_current_game(game)

        hom = self.Homotopy(game)
        hom.initialize()
        hom.solver.max_steps = self.max_steps
        hom.solver.verbose = 0
        hom.solver.store_path = False
        hom.solver.set_parameters(hom.tracking_parameters[self.tracking])

        sol = hom.solver.solve()

        if not sol['success']:
            self.game_memory.add_unsolved_game(game)

        return {'success': sol['success'],
                'seconds': sol['time'],
                'steps': sol['steps'],
                'path': sol['s'],
                'parameter': sol['y'][-1]}

    @staticmethod
    def plot_result(result: np.ndarray, show: bool = True) -> plt.figure:
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(221)
        ax1.set_title('Time')
        ax1.set_xlabel('time [sec]')
        ax1.set_ylabel('frequency')
        ax1.hist(result[:, 1], density=True)
        ax1.grid()
        ax2 = fig.add_subplot(222)
        ax2.set_title('Steps')
        ax2.set_xlabel(r'$n$')
        ax2.set_ylabel('frequency')
        ax2.hist(result[:, 2], density=True)
        ax2.grid()
        ax3 = fig.add_subplot(223)
        ax3.set_title('Path Length')
        ax3.set_xlabel(r'$s$')
        ax3.set_ylabel('frequency')
        ax3.hist(result[:, 3], density=True)
        ax3.grid()
        ax4 = fig.add_subplot(224)
        ax4.set_title('Homotopy Parameter')
        ax4.set_xlabel(r'$t$')
        ax4.set_ylabel('frequency')
        ax4.hist(result[:, 4], density=True)
        ax4.grid()
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    @ staticmethod
    def compute_timing(result: np.ndarray) -> dict:
        return {'success': 100*np.mean(result[:, 0]),
                'seconds': {'mean': np.nanmean(result[:, 1]),
                            'median': np.nanmedian(result[:, 1]),
                            'std': np.nanstd(result[:, 1])},
                'steps': {'mean': np.nanmean(result[:, 2]),
                          'median': np.nanmedian(result[:, 2]),
                          'std': np.nanstd(result[:, 2])},
                'path': {'mean': np.nanmean(result[:, 3]),
                         'median': np.nanmedian(result[:, 3]),
                         'std': np.nanstd(result[:, 3])},
                'parameter': {'mean': np.nanmean(result[:, 4]),
                              'median': np.nanmedian(result[:, 4]),
                              'std': np.nanstd(result[:, 4])}}

    @staticmethod
    def print_timing(timing: dict) -> None:

        seconds = timing['seconds']
        steps = timing['steps']
        path = timing['path']
        parameter = timing['parameter']

        print(f"\nsuccess: {timing['success']:0.0f} %")
        print("              mean / median /   std")
        print(f"seconds:    {seconds['mean']:6.2f} / {seconds['median']:6.2f} / {seconds['std']:5.2f}")
        print(f"steps:      {steps['mean']:6.0f} / {steps['median']:6.0f} / {steps['std']:5.0f}")
        print(f"path:       {path['mean']:6.0f} / {path['median']:6.0f} / {path['std']:5.0f}")
        print(f"parameter:  {parameter['mean']:6.0f} / {parameter['median']:6.0f} / {parameter['std']:5.0f}")

        return

    @staticmethod
    def print_latex_table(timings: dict) -> None:

        SPA = np.zeros((len(timings), 3), dtype=np.int32)
        for k, (s, p, a) in enumerate(timings.keys()):
            SPA[k] = (s, p, a)

        nums_s = np.unique(SPA[:, 0])
        nums_p = np.unique(SPA[:, 1])
        nums_a = np.unique(SPA[:, 2])

        print("")
        print("\\begin{tabular}{lc" + "c".join(['rl']*len(nums_p)) + "}")
        print("\\hline")
        print("$|A|$ / $|I|$ &  & " + " &  & ".join(str(p) + " & " for p in nums_p) + " \\\\")
        print("\\hline")
        for k, s in enumerate(nums_s):
            if k > 0:
                print(" &  & " + " &  & ".join([" & "]*len(nums_p)) + " \\\\")
            print("$|S|$ = " + str(s) + " &  & " + "".join([" &  & "]*len(nums_p)) + " \\\\")
            for a in nums_a:
                print(str(a) + " &  & " + " &  & ".join(f"{timings[(s, p, a)]['seconds']['mean']:0.2f}"
                                                        + " & \\hspace{-0.6em} ("
                                                        + f"{timings[(s, p, a)]['seconds']['std']:0.2f})"
                                                        for p in nums_p) + " \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("")

        return


class SGameMemory:
    """Class to collect unsolved games. Survives abortion."""

    def __init__(self):
        self.unsolved_games = []

    def replace_current_game(self, game):
        self.current_game = game

    def add_unsolved_game(self, game):
        self.unsolved_games.append(game)


# %% test and run


if __name__ == '__main__':

    # timings for default specification

    timer = HomotopyTimer()
    timer.timing()

    # check unsolved games

    game = timer.game_memory.current_game
    # game = homotopy_timer.game_memory.unsolved_games[0]

    hom = timer.Homotopy(game)
    hom.initialize()
    hom.solver.solve()

    # batch timings

    np.random.seed(42)
    timer = HomotopyTimer('QRE_np')

    # timer.batch_timings(nums_s=[2,3], nums_p=[2], nums_a=[2,3], reps=2)
    # timer.batch_timings()
