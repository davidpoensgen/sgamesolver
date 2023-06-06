"""Classes for stochastic game and corresponding homotopy."""
from typing import Union, List, Tuple, Optional

import numpy as np
import pandas as pd

from .homcont import HomContSolver

ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


class SGame:
    """A stochastic game."""

    def __init__(self, payoff_matrices: List[np.ndarray], transition_matrices: List[np.ndarray],
                 discount_factors: Union[np.ndarray, float, int]) -> None:
        """Inputs:

        payoff_matrices:      list of array-like, one for each state: payoff_matrices[s][p,A]

        transition_matrices:  list of array-like, one for each from_state: transition_matrices[s][A,s']

        discount_factors:     array-like: discount_factors[p]
                              or numeric (-> common discount factor)

        Different numbers of actions across states and players are allowed.
        Inputs should be of relevant dimension and should NOT contain nan.
        """

        # bring payoff_matrices to list of np.ndarray, one array for each state
        payoff_matrices = [np.array(payoff_matrices[s], dtype=np.float64) for s in range(len(payoff_matrices))]

        # read out game shape
        self.num_states = len(payoff_matrices)
        self.num_players = payoff_matrices[0].shape[0]

        self.nums_actions = np.zeros((self.num_states, self.num_players), dtype=np.int32)
        for s in range(self.num_states):
            for p in range(self.num_players):
                self.nums_actions[s, p] = payoff_matrices[s].shape[1 + p]

        self.num_actions_max = self.nums_actions.max()
        self.num_actions_total = self.nums_actions.sum()

        # action_mask allows to convert between a jagged and a flat array containing a strategy profile (or similar)
        self.action_mask = np.zeros((self.num_states, self.num_players, self.num_actions_max), dtype=bool)
        for s in range(self.num_states):
            for p in range(self.num_players):
                self.action_mask[s, p, 0:self.nums_actions[s, p]] = 1

        # u_mask allows to normalize and de-normalize u
        self.u_mask = np.zeros((self.num_states, self.num_players, *[self.num_actions_max] * self.num_players),
                               dtype=bool)
        for s in range(self.num_states):
            for p in range(self.num_players):
                for A in np.ndindex(*self.nums_actions[s]):
                    self.u_mask[(s, p) + A] = 1

        # generate array representing u [s,p,A]
        self.u = np.zeros((self.num_states, self.num_players, *[self.num_actions_max] * self.num_players))
        for s in range(self.num_states):
            for p in range(self.num_players):
                for A in np.ndindex(*self.nums_actions[s]):
                    self.u[(s, p) + A] = payoff_matrices[s][(p,) + A]

        # generate array representing discount factors [p]
        if isinstance(discount_factors, (list, tuple, np.ndarray)):
            self.delta = np.array(discount_factors, dtype=np.float64)
        else:
            self.delta = discount_factors * np.ones(self.num_players)

        # bring transition_matrices to list of np.ndarray, one array for each state
        transition_matrices = [np.array(transition_matrices[s], dtype=np.float64) for s in range(self.num_states)]

        # build big transition matrix [s,A,s'] from list of small transition matrices [A,s'] for each s
        transition_matrix = np.zeros((self.num_states, *[self.num_actions_max] * self.num_players, self.num_states))
        for s0 in range(self.num_states):
            for A in np.ndindex(*self.nums_actions[s0]):
                for s1 in range(self.num_states):
                    transition_matrix[(s0,) + A + (s1,)] = transition_matrices[s0][A + (s1,)]

        self.phi = transition_matrix

        self.u_ravel = self.u.ravel()
        self.phi_ravel = self.phi.ravel()

        self.transitions = None  # type: Optional[np.ndarray]
        # To support older versions of homotopies. See method make_transitions.

        digits = len(str(self.num_states))
        self.state_labels = [f'state{s:0{digits}}' for s in range(self.num_states)]
        digits = len(str(self.num_players))
        self.player_labels = [f'player{s:0{digits}}' for s in range(self.num_players)]
        digits = len(str(self.num_actions_max))
        self.action_labels = [f'a{a:0{digits}}' for a in range(self.num_actions_max)]

    def _make_transitions(self) -> None:
        """Method to support the transition from game.transitions (with indices [s,p,a0,a1,...S] and discount
         factors multiplied in) to game.phi (indices [s,a0,a1,....], no discount factors).
         Old versions of homotopies that require transitions should call this method during __init__()
        """
        # generate array representing transitions, including discounting: delta * phi [s,p,A,s']
        # (player index due to potentially player-specific discount factors)
        self.transitions = np.zeros((self.num_states, self.num_players, *[self.num_actions_max] * self.num_players,
                                     self.num_states))
        for p in range(self.num_players):
            self.transitions[:, p] = self.delta[p] * self.phi

    @classmethod
    def random_game(cls, num_states, num_players, num_actions, delta=0.95, seed=None) -> 'SGame':
        """Creates an SGame of given size, with random payoff- and transition arrays.
        num_actions can be specified in the following ways:
        - integer: all agents have this same fixed number of actions
        - list/tuple of 2 integers: number of actions is randomized, the input determining [min, max]
        - array of dimension [num_states, num_actions]: number of actions for each agent
        Similarly for delta:
        - float: value used for all players
        - tuple/list of 2 floats: randomized per player with (delta_min, delta_max)
        - list or array of length num_players: values used for all players

        Passing a seed to the random number generator ensures that the game can be recreated at a
        later occasion or by other users.
        """
        rng = np.random.default_rng(seed=seed)

        # if num_actions passed as int -> fixed number for all agents:
        if isinstance(num_actions, (int, float)):
            num_actions = np.ones((num_states, num_players), dtype=int) * num_actions
        # if given as (min, max) -> randomize accordingly
        elif isinstance(num_actions, (list, tuple)) and np.array(num_actions).shape == (2,):
            num_actions = rng.integers(low=num_actions[0], high=num_actions[1],
                                       size=(num_states, num_players), endpoint=True)
        # else, assume it is an array that fully specifies the game size
        num_actions = np.array(num_actions, dtype=np.int32)

        u = [rng.random((num_players, *num_actions[s, :])) for s in range(num_states)]

        phi = [rng.exponential(scale=1, size=(*num_actions[s, :], num_states)) for s in range(num_states)]
        for s in range(num_states):
            for index, value in np.ndenumerate(np.sum(phi[s], axis=-1)):
                phi[s][index] *= 1 / value

        if isinstance(delta, (int, float)):
            delta = np.ones(num_players) * delta
        elif isinstance(delta, (list, tuple)) and len(delta) == 2:
            delta = rng.uniform(delta[0], delta[1], size=num_players)

        return cls(u, phi, delta)

    @classmethod
    def one_shot_game(cls, payoff_matrix: np.ndarray) -> 'SGame':
        """Create a one-shot (=single-state/simultaneous) game from a payoff array."""
        # phi: zeros with shape like u, but dropping first dimension (player)
        # and appending a len-1-dimension for to-state
        phi = np.zeros((*payoff_matrix.shape[1:], 1))
        return cls([payoff_matrix], [phi], 0)

    @classmethod
    def from_table(cls, table: Union[pd.DataFrame, str]) -> 'SGame':
        """Create a game from the tabular format.
        Input can be either a pandas.DataFrame, or a string containing the path to an excel file (.xlsx, .xls), a
        stata file (.dta), or a plain text file with comma separated values (.csv, .txt).
        For reference on how the table should be formatted, please refer to the online manual.
        """
        from sgamesolver.utility.sgame_conversion import game_from_table
        return game_from_table(table)

    def to_table(self) -> pd.DataFrame:
        """Convert the game to the tabular format. Returns a pandas.DataFrame that can then be saved in the format
        of choice. Note that this might take quite long for large games.
        """
        from sgamesolver.utility.sgame_conversion import game_to_table
        return game_to_table(self)

    @property
    def action_labels(self):
        return self._action_labels

    @action_labels.setter
    def action_labels(self, value):
        if isinstance(value[0], (str, int, float)):
            # just a single list which applies to all state/players
            self._action_labels = [[value for _ in range(self.num_players)] for _ in range(self.num_states)]
        else:
            self._action_labels = value

    def random_strategy(self, zeros=False, seed=None) -> np.ndarray:
        """Generate a random strategy profile. Padded with NaNs, or zeros under the respective option."""
        rng = np.random.default_rng(seed=seed)

        strategy_profile = np.full((self.num_states, self.num_players, self.num_actions_max), 0.0 if zeros else np.NaN)
        for s in range(self.num_states):
            for p in range(self.num_players):
                sigma = rng.exponential(scale=1, size=self.nums_actions[s, p])
                strategy_profile[s, p, :self.nums_actions[s, p]] = sigma / sigma.sum()
        return strategy_profile

    def centroid_strategy(self, zeros=False) -> np.ndarray:
        """Generate the centroid strategy profile. Padded with NaNs, or zeros under the respective option."""

        strategy_profile = np.full((self.num_states, self.num_players, self.num_actions_max), 0.0 if zeros else np.NaN)
        for s in range(self.num_states):
            for p in range(self.num_players):
                strategy_profile[s, p, :self.nums_actions[s, p]] = 1 / self.nums_actions[s, p]
        return strategy_profile

    def weighted_centroid_strategy(self, weights: np.ndarray, zeros=False) -> np.ndarray:
        """Generate a weighted centroid strategy profile. Padded with NaNs, or zeros under the respective option."""

        strategy_profile = np.full((self.num_states, self.num_players, self.num_actions_max), 0.0 if zeros else np.NaN)
        for s in range(self.num_states):
            for p in range(self.num_players):
                strategy_profile[s, p, :self.nums_actions[s, p]] = (weights[s, p, :self.nums_actions[s, p]]
                                                                    / np.sum(weights[s, p, :self.nums_actions[s, p]]))
        return strategy_profile

    def random_weights(self, low: float = 0, high: float = 1, zeros=False, seed: Optional[int] = None) -> np.ndarray:
        """Generate a randomized set of weights of the same shape as a strategy profile (i.e. one weight per action).
        Distribution is uniform on the interval [low, high]."""
        rng = np.random.default_rng(seed)
        nu_flat = rng.uniform(low, high, self.num_actions_total)
        return self.unflatten_strategies(nu_flat, zeros=zeros)

    def flatten_strategies(self, strategies: np.ndarray) -> np.ndarray:
        """Convert a jagged array of shape (num_states, num_players, num_actions_max), e.g. strategy profile,
        to a flat array, removing all NaNs.
        """
        return np.extract(self.action_mask, strategies)

    def unflatten_strategies(self, strategies_flat: np.ndarray, zeros: bool = False) -> np.ndarray:
        """Convert a flat array containing a strategy profile (or parameters of same shape) to an array
        with shape (num_states, num_players, num_actions_max), padded with NaNs (or zeros under the respective option.)
        """
        strategies = np.full((self.num_states, self.num_players, self.num_actions_max), 0.0 if zeros else np.NaN)
        np.place(strategies, self.action_mask, strategies_flat)
        return strategies

    def get_values(self, strategy_profile: np.ndarray) -> np.ndarray:
        """Calculate state-player values for a given strategy profile."""

        sigma = np.nan_to_num(strategy_profile)
        sigma_list = [sigma[:, p, :] for p in range(self.num_players)]

        # einsum eqs: u: 'spABC...,sA,sB,sC,...->sp' ; phi: 'sABC...t,sA,sB,sC,...->st'
        einsum_eq_u = f'sp{ABC[0:self.num_players]},s{",s".join(ABC[p] for p in range(self.num_players))}->sp'
        einsum_eq_phi = f's{ABC[0:self.num_players]}t,s{",s".join(ABC[p] for p in range(self.num_players))}->st'

        u = np.einsum(einsum_eq_u, self.u, *sigma_list)
        phi = np.einsum(einsum_eq_phi, self.phi, *sigma_list)

        values = np.empty((self.num_states, self.num_players))
        try:
            for p in range(self.num_players):
                A = np.eye(self.num_states) - phi * self.delta[p]
                values[:, p] = np.linalg.solve(A, u[:, p])
        except np.linalg.LinAlgError:
            raise "Failed to solve for state-player values: Transition matrix not invertible."
        return values

    @staticmethod
    def flatten_values(values: np.ndarray) -> np.ndarray:
        """Flatten an array with shape (num_states, num_players), e.g. state-player values."""
        return np.array(values).reshape(-1)

    def unflatten_values(self, values_flat: np.ndarray) -> np.ndarray:
        """Convert a flat array to shape (num_states, num_players), e.g. state-player values."""
        return np.array(values_flat).reshape((self.num_states, self.num_players))

    def check_equilibrium(self, strategy_profile: np.ndarray) -> np.ndarray:
        """Calculate "epsilon-equilibriumness" (maximum total utility each agent could gain by a one-shot deviation)
        of a given strategy profile.
        """
        values = self.get_values(strategy_profile)
        sigma = np.nan_to_num(strategy_profile)

        # u_tilde: u of normal form games including continuation values.
        dV = values * self.delta
        u_tilde = self.u + np.einsum('s...S,Sp->sp...', self.phi, dV)

        losses = np.empty((self.num_states, self.num_players))

        for p in range(self.num_players):
            others = [q for q in range(self.num_players) if q != p]
            einsum_eq = ('s' + ABC[0:self.num_players] + ',s' + ',s'.join(ABC[q] for q in others) + '->s' + ABC[p])
            action_values = np.einsum(einsum_eq, u_tilde[:, p, :], *[sigma[:, q, :] for q in others])

            losses[:, p] = action_values.max(axis=-1) - values[:, p]

        return losses


class StrategyProfile:
    """Container for equilibria and other strategy profiles."""

    def __init__(self, game: SGame, sigma, V=None, t=None):
        self.game = game
        self.strategies = sigma
        if V is None:
            self.values = game.get_values(sigma)
        else:
            self.values = V
        self.homotopy_parameter = t

    def to_string(self, decimals=3, v_decimals=2, action_labels=True) -> str:
        """Renders the strategy profile and associated values as human-readable string."""
        string = ""
        player_len = len(max(self.game.player_labels, key=len))
        state_len = len(max(self.game.state_labels, key=len))
        for state in range(self.game.num_states):
            string += f'+++++++++{" " + self.game.state_labels[state] + " ":+^{state_len + 2}}+++++++++\n'
            V_s = [f'{self.values[state, i]:.{v_decimals}f}' for i in range(self.game.num_players)]
            V_width = len(max(V_s, key=len))
            for player in range(self.game.num_players):
                sigma_si = self.strategies[state, player, :self.game.nums_actions[state, player]]
                row = f'{self.game.player_labels[player]: <{player_len+1}}: v={V_s[player]:>{V_width}}, ' \
                      f'σ={np.array2string(sigma_si, formatter={"float_kind": lambda x: f"%.{decimals}f" % x})}\n'
                if action_labels and (player == 0 or self.game.action_labels[state][player]
                                      != self.game.action_labels[state][player-1]):
                    labels = self.game.action_labels[state][player]
                    if len(labels) > self.game.nums_actions[state, :].max():
                        # shorten list if more labels than actions exist
                        labels = labels[:self.game.nums_actions[state, :].max()]
                    leading_space = " " * (len(row.split("σ=")[0]) + 3)
                    action_width = decimals + 2
                    label_string = " ".join([f'{a[:action_width]:<{action_width}}' for a in labels])
                    string += leading_space + label_string + "\n"
                string += row

        return string

    def to_list(self, decimals: int = None) -> list:
        if decimals is None:
            sigma = self.strategies
        else:
            sigma = np.round(self.strategies, decimals)
        list_ = [[sigma[s, p, :self.game.nums_actions[s, p]].tolist() for p in range(self.game.num_players)]
                 for s in range(self.game.num_states)]
        return list_

    def __str__(self):
        return self.to_string()

    def simulate(self, initial_state=None, max_periods=100, runs: int = 1,
                 labels: bool = True, seed: int = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed=seed)
        """Simulates the strategy profile. Inputs are
        initial_state: The starting state of each simulation run, or a distribution over it. Can be an integer 
            (which represents the 0-based index of the state), the label of the state as string, or a numpy array 
            representing a probability distribution over all states.
        max_periods: The number of periods simulated per run.
        runs: The number of independent runs performed.
        labels:if True (default) the result will contain state- and action labels; if False, their integer index will 
            be used.
        seed: allows to set a seed for the random number generator, allowing results to be reproducible.
        
        Returns a Pandas DataFrame with the following columns:
        run: indicates to which run the row belongs
        period: the period to which it refers
        state: current state 
        a_[player], one for each player: the action chosen by this player
        u_[player], one for each player: the instantaneous utility for this player
        V_[player], total discounted utility up until and including the current period, from the perspective of period 0
        """

        # process initial state
        if initial_state is None:
            initial_state_distribution = np.ones(self.game.num_states)/self.game.num_states
        elif isinstance(initial_state, str):
            try:
                state_no = self.game.state_labels.index(initial_state)
            except ValueError:
                raise ValueError(f'State "{initial_state}" is not in the state label list of the game.')
            # get state number here
            initial_state_distribution = np.zeros(self.game.num_states)
            initial_state_distribution[state_no] = 1
            pass
        elif isinstance(initial_state, (int, float)):
            initial_state_distribution = np.zeros(self.game.num_states)
            initial_state_distribution[initial_state] = 1
        elif isinstance(initial_state, (list, tuple, np.ndarray)):
            if len(initial_state) != self.game.num_states:
                raise ValueError(f'Argument initial_state has {len(initial_state)} entries, '
                                 f'but the game has {self.game.num_states} states.')
            initial_state_distribution = np.ndarray(initial_state)

        sigma = np.nan_to_num(self.strategies)
        # copy phi and extend the last dimension (=to_state) with an entry of 1-sum of the other probabilities
        # the result can be used with random.choice, with the last entry representing chance of termination
        phi = np.append(self.game.phi, 1 - self.game.phi.sum(axis=-1, keepdims=True), axis=-1)
        # due to rounding error, 1-sum can be slightly negative, which would throw an error in random.choice.
        # set those entries to 0. (Careful: does not check for any other violations etc., will miss phi summing to > 1)
        phi[..., -1] = np.maximum(phi[..., -1], 0)

        player_strings = [self.game.player_labels[player] if labels else str(player)
                          for player in range(self.game.num_players)]

        # will create a dictionary for each row, and only convert to pd.Dataframe at the end,
        # as final length is unknown -> when appending, memory would need to be reallocated all the time
        rows = []

        for run in range(runs):
            next_state_distribution = np.append(initial_state_distribution, [0])
            total_utility = np.zeros(self.game.num_players)
            delta_powers = np.ones(self.game.num_players)
            period = 0
            while period < max_periods:
                state = rng.choice(self.game.num_states + 1, p=next_state_distribution)
                if state == self.game.num_states:
                    # this additional "state" corresponds to termination of the game, see above
                    break
                # No need to truncate sigma to num_actions: additional entries are 0 anyway
                action_profile = tuple(rng.choice(self.game.num_actions_max, p=sigma[state, player, :])
                                       for player in range(self.game.num_players))
                u = self.game.u[(state, slice(None)) + action_profile]
                total_utility += np.multiply(u, delta_powers)
                delta_powers = delta_powers * self.game.delta
                next_state_distribution = phi[(state,) + action_profile + (slice(None),)]

                row = {
                    'run': run,
                    'period': period,
                    'state': self.game.state_labels[state] if labels else state,
                }
                for player in range(self.game.num_players):
                    row['action_' + player_strings[player]] = \
                        self.game.action_labels[state][player][action_profile[player]] if labels \
                        else action_profile[player]
                    row['u_' + player_strings[player]] = u[player]
                    row['V_' + player_strings[player]] = total_utility[player]
                rows.append(row)

                period += 1
        columns = ['run', 'period', 'state']
        columns += ['action_' + player_strings[player] for player in range(self.game.num_players)]
        columns += ['u_' + player_strings[player] for player in range(self.game.num_players)]
        columns += ['V_' + player_strings[player] for player in range(self.game.num_players)]
        output = pd.DataFrame(rows, columns=columns)

        return output


class SGameHomotopy:
    """General homotopy class for stochastic games."""

    def __init__(self, game: SGame) -> None:
        self.game = game
        self.y0 = None
        self.solver = None  # type: Optional[HomContSolver]
        self.equilibrium = None  # type: Optional[StrategyProfile]

    default_parameters = {}
    robust_parameters = {}

    def solver_setup(self) -> None:
        """Any steps in preparation to start solver:
        - set priors, weights etc. if needed
        - set starting point y0
        - set up homCont to solve the game
        """
        pass

    def solve(self) -> None:
        """Start the solver and store the equilibrium if solver is successful."""
        if not self.solver:
            print('Please run .solver_setup() first to set up the solver.')
            return
        self.solver.start()

    def notify(self, result):
        """Takes continuation result from HomContSolver and in case of success, stores it as equilibrium."""
        if result['success']:
            sigma, V, t = self.y_to_sigma_V_t(result['y'])
            self.equilibrium = StrategyProfile(self.game, sigma, V, t)
            if self.solver.verbose:
                print('An equilibrium was found via homotopy continuation.')
        else:
            if self.solver.verbose:
                print('The solver failed to find an equilibrium. Please refer to the manual'
                      ' for suggestions how to proceed.')  # TODO: link manual perhaps?

    def find_y0(self) -> np.ndarray:
        """Calculate starting point y0."""
        pass

    def H(self, y: np.ndarray) -> np.ndarray:
        """Homotopy function evaluated at y."""
        pass

    def J(self, y: np.ndarray) -> np.ndarray:
        """Jacobian of homotopy function evaluated at y."""
        pass

    def sigma_V_t_to_y(self, sigma: np.ndarray, V: np.ndarray, t: Union[float, int]) -> np.ndarray:
        """Generate vector y from arrays representing strategies sigma, values V, and homotopy parameter t.
        """
        sigma_flat = self.game.flatten_strategies(sigma)
        V_flat = self.game.flatten_values(V)
        return np.concatenate([sigma_flat, V_flat, [t]])

    def y_to_sigma_V_t(self, y: np.ndarray, zeros: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """Translate a vector y to arrays representing strategies sigma, values V and homotopy parameter t.
        """
        sigma = self.game.unflatten_strategies(y[0:self.game.num_actions_total], zeros=zeros)
        V = self.game.unflatten_values(y[self.game.num_actions_total:-1])
        t = y[-1]
        return sigma, V, t

    def plot_path(self, x_axis="s", s_range=None, step_range=None):
        """Plots the path the solver has followed.
        Requires that path storing was enabled before starting the solver.
        If an s_range, i.e. a tuple (s_min, s_max), is passed, only steps for which s_min < s < s_max will be plotted.
        Likewise, passing a step_range, i.e. (first_step, last_step) also allows to plot a subset of steps only."""
        if not self.solver or not self.solver.path:
            raise ValueError('No solver or no stored path.')
        import matplotlib.pyplot as plt
        import matplotlib.lines

        path = self.solver.path
        if s_range is not None:
            rows = np.nonzero((s_range[0] <= path.s) & (path.s <= s_range[1]))[0]
        elif step_range is not None:
            rows = np.nonzero((step_range[0] <= path.step) & (path.step <= step_range[1]))[0]
        else:
            rows = np.arange(0, path.index)
        if len(rows) == 0:
            raise ValueError("No data for the given range.")

        if x_axis == "s":
            x_plot = path.s[rows]
            x_label = "path length s"
        elif x_axis == "t":
            x_plot = path.y[rows, -1]
            x_label = "homotopy parameter t"
        elif x_axis == "step":
            x_plot = path.step[rows]
            x_label = "step number"
        else:
            raise ValueError(f'"{x_axis}" is not a valid value for parameter x_axis. Allowed are "s", "t", or "step".')

        # get sigma from y
        num_rows = len(x_plot)
        sigma_plot = np.empty((num_rows, self.game.num_states, self.game.num_players, self.game.num_actions_max))
        for idx, row in np.ndenumerate(rows):
            sigma_plot[idx, :] = self.y_to_sigma_V_t(path.y[row])[0]
        figure, axis = plt.subplots(nrows=self.game.num_states, ncols=self.game.num_players,
                                    figsize=(self.game.num_players * 2.66, self.game.num_states * 2),
                                    squeeze=False)

        for state in range(self.game.num_states):
            for player in range(self.game.num_players):
                ax = axis[state, player]
                ax.plot(x_plot, sigma_plot[:, state, player, :])
                ax.set_ylim((-.05, 1.05))
                ax.label_outer()
                if player == 0:
                    ax.set_ylabel(f'state{state}', rotation=90, size='large')
                if state == 0:
                    ax.set_title(f'player{player}')
                if state == self.game.num_states - 1:
                    ax.set_xlabel(x_label)

        figure.tight_layout()

        # add some padding at bottom, and place action legend there:
        figure.subplots_adjust(bottom=0.4 / self.game.num_states)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        legend_elements = [matplotlib.lines.Line2D([0], [0], color=colors[i], label=f'action{i}')
                           for i in range(self.game.num_actions_max)]
        figure.legend(handles=legend_elements, ncol=self.game.num_actions_max,
                      loc='lower center', bbox_to_anchor=(0.5, 0))

        figure.show()
        return figure


class LogStratHomotopy(SGameHomotopy):
    """Base class for homotopies using logarithmized strategies
    (where y contains beta := log(sigma), rather than sigma).
    """

    def sigma_V_t_to_y(self, sigma: np.ndarray, V: np.ndarray, t: float) -> np.ndarray:
        """Translate arrays representing strategies sigma, values V and homotopy parameter t to a vector y.
        (Version for homotopies operating on logarithmized strategies.)
        """
        beta_flat = np.log(self.game.flatten_strategies(sigma))
        V_flat = self.game.flatten_values(V)
        return np.concatenate([beta_flat, V_flat, [t]])

    def y_to_sigma_V_t(self, y: np.ndarray, zeros: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """Translate a vector y to arrays representing strategies sigma, values V and homotopy parameter t.
        (Version for homotopies operating on logarithmized strategies.)
        """
        sigma = self.game.unflatten_strategies(np.exp(y[:self.game.num_actions_total]), zeros=zeros)
        V = self.game.unflatten_values(y[self.game.num_actions_total:-1])
        t = y[-1]
        return sigma, V, t

    def sigma_distance(self, y_new: np.ndarray, y_old: np.ndarray) -> float:
        """Calculates the distance in strategies (sigma) between y_old and y_new,
        in the maximum norm, normalized by distance in homotopy parameter t. Used as convergence criterion."""
        sigma_difference = np.exp(y_new[:self.game.num_actions_total]) - np.exp(y_old[:self.game.num_actions_total])
        sigma_distance = np.max(np.abs(sigma_difference))
        return sigma_distance / np.abs(y_new[-1] - y_old[-1])
