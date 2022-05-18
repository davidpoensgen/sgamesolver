"""Classes for stochastic game and corresponding homotopy."""


from typing import Union, List, Tuple, Optional
import numpy as np
from .homcont import HomCont

ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# %% game class


class SGame:
    """A stochastic game."""

    def __init__(self, payoff_matrices: List[np.ndarray], transition_matrices: List[np.ndarray],
                 discount_factors: Union[np.ndarray, float, int] = 0.0) -> None:
        """Inputs:

        payoff_matrices:      list of array-like, one for each state: payoff_matrices[s][p,A]

        transition_matrices:  list of array-like, one for each from_state: transition_matrices[s][A,s']
                              or None (-> separated repeated game)

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

        # payoff_mask allows to normalize and de-normalize payoffs
        self.payoff_mask = np.zeros((self.num_states, self.num_players, *[self.num_actions_max] * self.num_players),
                                    dtype=bool)
        for s in range(self.num_states):
            for p in range(self.num_players):
                for A in np.ndindex(*self.nums_actions[s]):
                    self.payoff_mask[(s, p) + A] = 1

        # generate array representing payoffs [s,p,A]
        self.payoffs = np.zeros((self.num_states, self.num_players, *[self.num_actions_max] * self.num_players))
        for s in range(self.num_states):
            for p in range(self.num_players):
                for A in np.ndindex(*self.nums_actions[s]):
                    self.payoffs[(s, p) + A] = payoff_matrices[s][(p,) + A]

        # generate array representing discount factors [p]
        if isinstance(discount_factors, (list, tuple, np.ndarray)):
            self.discount_factors = np.array(discount_factors, dtype=np.float64)
        else:
            self.discount_factors = discount_factors * np.ones(self.num_players)

        # scale potentially useful for adjusting tracking parameters
        self.payoff_min = self.payoffs[self.payoff_mask].min()
        self.payoff_max = self.payoffs[self.payoff_mask].max()

        # bring transition_matrices to list of np.ndarray, one array for each state
        transition_matrices = [np.array(transition_matrices[s], dtype=np.float64) for s in range(self.num_states)]

        # build big transition matrix [s,A,s'] from list of small transition matrices [A,s'] for each s
        transition_matrix = np.zeros((self.num_states, *[self.num_actions_max] * self.num_players, self.num_states))
        for s0 in range(self.num_states):
            for A in np.ndindex(*self.nums_actions[s0]):
                for s1 in range(self.num_states):
                    transition_matrix[(s0,) + A + (s1,)] = transition_matrices[s0][A + (s1,)]

        self.phi = transition_matrix  # this is the not-per-player version. TODO: clean up

        self.u_ravel = self.payoffs.ravel()
        self.phi_ravel = self.phi.ravel()

        self.transitions = None  # type: Optional[np.ndarray]
        # To support older versions of homotopies. See method make_transitions.

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
            self.transitions[:, p] = self.discount_factors[p] * self.phi

    @classmethod
    def random_game(cls, num_states, num_players, num_actions, delta=0.95, seed=None):
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
    def one_shot_game(cls, payoff_matrix: np.ndarray):
        """Creates a one-shot (=single-state/simultaneous) game from a payoff array."""
        # phi: zeros with shape like u, but dropping first dimension (player)
        # and appending a len-1-dimension for to-state
        phi = np.zeros((*payoff_matrix.shape[1:], 1))
        return cls([payoff_matrix], [phi], 0)

    def detect_symmetries(self) -> None:
        """Detect symmetries between agents."""
        pass

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
        """Randomize a set of weights of the same shape as a strategy profile.
        Distribution is uniform on the interval [low, high]."""
        if low < 0:
            raise ValueError("Lower bound for weights must be non-negative.")
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

        u = np.einsum(einsum_eq_u, self.payoffs, *sigma_list)
        phi = np.einsum(einsum_eq_phi, self.phi, *sigma_list)

        values = np.empty((self.num_states, self.num_players))
        try:
            for p in range(self.num_players):
                A = np.eye(self.num_states) - phi * self.discount_factors[p]
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

        # u_tilde: payoffs of normal form games including continuation values.
        dV = values * self.discount_factors
        u_tilde = self.payoffs + np.einsum('s...S,Sp->sp...', self.phi, dV)

        losses = np.empty((self.num_states, self.num_players))

        for p in range(self.num_players):
            others = [q for q in range(self.num_players) if q != p]
            einsum_eq = ('s' + ABC[0:self.num_players] + ',s' + ',s'.join(ABC[q] for q in others) + '->s' + ABC[p])
            action_values = np.einsum(einsum_eq, u_tilde[:, p, :], *[sigma[:, q, :] for q in others])

            losses[:, p] = action_values.max(axis=-1) - values[:, p]

        return losses


class StrategyProfile:
    """Container for equilibria and other strategy profiles."""

    def __init__(self, game, sigma, V, t=None):
        self.game = game
        self.strategies = sigma
        self.values = V
        self.homotopy_parameter = t

    def to_string(self, decimals=3) -> str:
        """Renders the strategy profile and associated values as human-readable string."""
        string = ""
        for state in range(self.game.num_states):
            string += f'+++++++ state{state} +++++++\n'
            for player in range(self.game.num_players):
                V_si = self.values[state, player]
                sigma_si = self.strategies[state, player, :self.game.nums_actions[state, player]]
                string += f'player{player}: v={V_si:#5.2f}, ' \
                          f's={np.array2string(sigma_si, formatter={"float_kind": lambda x: f"%.{decimals}f" % x})}\n'
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


class SGameHomotopy:
    """General homotopy class for stochastic games."""

    def __init__(self, game: SGame) -> None:
        self.game = game
        self.y0 = None
        self.tracking_parameters = {}
        self.solver = None  # type: Optional[HomCont]
        self.equilibrium = None  # type: Optional[StrategyProfile]

    def solver_setup(self) -> None:
        """Any steps in preparation to start solver:
        - set priors, weights etc. if needed
        - set starting point y0
        - prepare symmetry helpers
            + (make sure priors and other parameters are in accordance with symmetries)
        - set up homCont to solve the game
        """
        pass

    def solve(self) -> None:
        """Start the solver and store the equilibrium if solver is successful."""
        if not self.solver:
            print('Please run .solver_setup() first to set up the solver.')
            return
        solution = self.solver.start()
        if solution['success']:
            sigma, V, t = self.y_to_sigma_V_t(solution['y'])
            self.equilibrium = StrategyProfile(self.game, sigma, V, t)
            print('An equilibrium was found via homotopy continuation.')
        else:
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

    def H_reduced(self, y: np.ndarray) -> np.ndarray:
        """H evaluated at y, reduced by exploiting symmetries."""
        pass

    def J_reduced(self, y: np.ndarray) -> np.ndarray:
        """J evaluated at y, reduced by exploiting symmetries."""
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
        try:
            import matplotlib.pyplot as plt
            import matplotlib.lines
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('Package matplotlib is required for plotting.') from e

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

    def sigma_V_t_to_y(self, sigma: np.ndarray, V: np.ndarray, t: Union[float, int]) -> np.ndarray:
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

    def sigma_distance(self, y_new, y_old):
        """Calculates the distance in strategies (sigma) between y_old and y_new,
        in the maximum norm, normalized by distance in homotopy parameter t. Used as convergence criterion."""
        sigma_difference = np.exp(y_new[:self.game.num_actions_total]) - np.exp(y_old[:self.game.num_actions_total])
        sigma_distance = np.max(np.abs(sigma_difference))
        return sigma_distance / np.abs(y_new[-1] - y_old[-1])
