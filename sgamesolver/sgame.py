"""Classes for stochastic game and corresponding homotopy."""

# TODO: define scale of game for adjusting tracking parameters
# TODO: finalize method check_equilibriumness
# TODO: document ordering of variables in y and equations in H and J
# TODO: symmetry


from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike

ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# %% game class


class SGame:
    """A stochastic game."""

    def __init__(self, payoff_matrices: list[ArrayLike], transition_matrices: list[ArrayLike],
                 discount_factors: Union[ArrayLike, float, int] = 0.0) -> None:
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

        # define scale for adjusting tracking parameters
        self.payoff_min = self.payoffs[self.payoff_mask].min()
        self.payoff_max = self.payoffs[self.payoff_mask].max()
        # TODO

        # bring transition_matrices to list of np.ndarray, one array for each state
        transition_matrices = [np.array(transition_matrices[s], dtype=np.float64) for s in range(self.num_states)]

        # build big transition matrix [s,A,s'] from list of small transition matrices [A,s'] for each s
        transition_matrix = np.zeros((self.num_states, *[self.num_actions_max] * self.num_players, self.num_states))
        for s0 in range(self.num_states):
            for A in np.ndindex(*self.nums_actions[s0]):
                for s1 in range(self.num_states):
                    transition_matrix[(s0,) + A + (s1,)] = transition_matrices[s0][A + (s1,)]

        # generate array representing transitions, including discounting: delta * phi [s,p,A,s']
        # (player index due to potentially player-specific discount factors)
        self.transitions = np.zeros((self.num_states, self.num_players, *[self.num_actions_max] * self.num_players,
                                     self.num_states))
        for p in range(self.num_players):
            self.transitions[:, p] = self.discount_factors[p] * transition_matrix

    @classmethod
    def random_game(cls, num_states, num_players, num_actions, delta=0.95, seed=None):
        """Creates an SGame of given size, with random payoff- and transition arrays.
        num_actions can be specified in the following ways:
        - integer: all agents have this same fixed number of actions
        - list/tuple of 2 integers: number of actions is randomized, the input determining [min, max]
        - array of dimension [num_states, num_actions]: number of actions for each agent

        A seed can be passed to the random number generator, ensuring that the game can be recreated later or by others.
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
        num_actions = np.array(num_actions, dtype=int)

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
    def one_shot_game(cls, payoff_matrix: ArrayLike):
        """Creates a one-shot (=single-state/simultaneous) game from a payoff array."""
        # phi: zeros with shape like u, but dropping first dimension (player)
        # and appending a len-1-dimension for to-state
        phi = np.zeros((*payoff_matrix.shape[1:], 1))
        return cls([payoff_matrix], [phi], 0)

    def detect_symmetries(self) -> None:
        """Detect symmetries between agents."""
        # TODO
        pass

    def random_strategy(self) -> np.ndarray:
        """Generate a random strategy profile."""

        strategy_profile = np.nan * np.empty((self.num_states, self.num_players, self.num_actions_max))
        for s in range(self.num_states):
            for p in range(self.num_players):
                sigma = np.random.exponential(scale=1, size=self.nums_actions[s, p])
                strategy_profile[s, p, :self.nums_actions[s, p]] = sigma / sigma.sum()

        return strategy_profile

    def centroid_strategy(self, weights: Optional[ArrayLike] = None) -> np.ndarray:
        """Generate the (weighted) centroid strategy profile."""

        if weights is None:
            weights = np.ones((self.num_states, self.num_players, self.num_actions_max))
        else:
            weights = np.array(weights)

        strategy_profile = np.nan * np.empty((self.num_states, self.num_players, self.num_actions_max))
        for s in range(self.num_states):
            for p in range(self.num_players):
                strategy_profile[s, p, :self.nums_actions[s, p]] = (weights[s, p, :self.nums_actions[s, p]]
                                                                    / np.sum(weights[s, p, :self.nums_actions[s, p]]))

        return strategy_profile

    def flatten_strategies(self, strategies: ArrayLike) -> np.ndarray:
        """Convert a jagged array of shape (num_states, num_players, num_actions_max), e.g. strategy profile,
        to a flat array, removing all NaNs.
        """
        return np.extract(self.action_mask, strategies)

    def unflatten_strategies(self, strategies_flat: ArrayLike, zeros: bool = False) -> np.ndarray:
        """Convert a flat array containing a strategy profile or similar to an array
        with shape (num_states, num_players, num_actions_max), padded with NaNs (or zeros under the respective option.)
        """
        if zeros:
            strategies = np.zeros((self.num_states, self.num_players, self.num_actions_max))
        else:
            strategies = np.nan * np.empty((self.num_states, self.num_players, self.num_actions_max))
        np.place(strategies, self.action_mask, strategies_flat)
        return strategies

    def get_values(self, strategy_profile: ArrayLike) -> np.ndarray:
        """Calculate state-player values for a given strategy profile."""

        sigma = np.nan_to_num(strategy_profile)
        sigma_list = [sigma[:, p, :] for p in range(self.num_players)]

        # einsum eqs: u: 'spABC...,sA,sB,sC,...->sp' ; phi: 'spAB...t,sA,sB,sC,...->spt'
        einsum_eq_u = f'sp{ABC[0:self.num_players]},s{",s".join(ABC[p] for p in range(self.num_players))}->sp'
        einsum_eq_phi = f'sp{ABC[0:self.num_players]}t,s{",s".join(ABC[p] for p in range(self.num_players))}->spt'

        u = np.einsum(einsum_eq_u, self.payoffs, *sigma_list)
        phi = np.einsum(einsum_eq_phi, self.transitions, *sigma_list)

        values = np.empty((self.num_states, self.num_players))
        try:
            for p in range(self.num_players):
                A = np.eye(self.num_states) - phi[:, p, :]
                values[:, p] = np.linalg.solve(A, u[:, p])
        except np.linalg.LinAlgError:
            raise "Failed to solve for state-player values: Transition matrix not invertible."
        return values

    @staticmethod
    def flatten_values(values: ArrayLike) -> np.ndarray:
        """Flatten an array with shape (num_states, num_players), e.g. state-player values."""
        return np.array(values).reshape(-1)

    def unflatten_values(self, values_flat: ArrayLike) -> np.ndarray:
        """Convert a flat array to shape (num_states, num_players), e.g. state-player values."""
        return np.array(values_flat).reshape((self.num_states, self.num_players))

    def check_equilibrium(self, strategy_profile: ArrayLike) -> np.ndarray:
        """Calculate "epsilon-equilibriumness" (max total utility any agent could gain by deviating)
        of a given strategy profile.
        """
        values = self.get_values(strategy_profile)
        sigma = np.nan_to_num(strategy_profile)

        # u_tilde: payoffs of normal form games that include continuation values.
        u_tilde = self.payoffs + np.einsum('sp...S,Sp->sp...', self.transitions, values)

        losses = np.empty((self.num_states, self.num_players))

        for p in range(self.num_players):
            others = [q for q in range(self.num_players) if q != p]
            einsum_eq = ('s' + ABC[0:self.num_players] + ',s' + ',s'.join(ABC[q] for q in others) + '->s' + ABC[p])
            action_values = np.einsum(einsum_eq, u_tilde[:, p, :], *[sigma[:, q, :] for q in others])

            losses[:, p] = action_values.max(axis=-1) - values[:, p]

        # TODO: absolute losses mean different things when games are scaled differently.
        # TODO: should we use percentages? or the min of some percentage and some absolute deviation?

        # TODO: decide whether losses should be aggregated (player/agent/...?)
        # TODO: Should this function report? Return? etc.
        return losses


# %% blueprint for homotopy classes


class SGameHomotopy:
    """General homotopy class for some SGame.

    TODO: document order of (beta, V, T) in y
    TODO: document order of equations in H (and thus J)
    """

    def __init__(self, game: SGame) -> None:
        self.game = game
        self.y0 = None
        self.tracking_parameters = {}
        self.solver = None
        self.equilibrium = None

    def initialize(self) -> None:
        """Any steps in preparation to start solver:
        - set priors, weights etc. if needed
        - set starting point y0
        - prepare symmetry helpers
            + (make sure priors and other parameters are in accordance with symmetries)
        - set up homCont to solve the game
        """
        pass

    def solve(self) -> None:
        """TODO: just playing with ideas to make things more easily usable
        """
        if not self.solver:
            print('Please run .initialize() first to set up the solver.')
            return
        solution = self.solver.loop()
        if solution['success']:
            sigma, V, t = self.y_to_sigma_V_t(solution['y'])
            self.equilibrium = {'strategies': sigma,
                                'values': V,
                                'homotopy_parameter': t,
                                }
            print(f'An equilibrium was found via homotopy continuation.')
        else:
            print(f'The solver failed to find an equilibrium. Please refer to the manual'
                  f' for suggestions how to proceed.')  # TODO: link manual perhaps?

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
        # TODO: to be implemented here
        pass

    def J_reduced(self, y: np.ndarray) -> np.ndarray:
        """J evaluated at y, reduced by exploiting symmetries."""
        # TODO: to be implemented here
        pass

    def sigma_V_t_to_y(self, sigma: np.ndarray, V: np.ndarray, t: Union[float, int]) -> np.ndarray:
        """Generate vector y from arrays representing strategies sigma, values V, and homotopy parameter t.
        """
        sigma_flat = self.game.flatten_strategies(sigma)
        V_flat = self.game.flatten_values(V)
        return np.concatenate([sigma_flat, V_flat, [t]])

    def y_to_sigma_V_t(self, y: np.ndarray, zeros: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
        """Translate a vector y to arrays representing strategies sigma, values V and homotopy parameter t.
        """
        sigma = self.game.unflatten_strategies(y[0:self.game.num_actions_total], zeros=zeros)
        V = self.game.unflatten_values(y[self.game.num_actions_total:-1])
        t = y[-1]
        return sigma, V, t

    def equilibrium_string(self):
        """Returns a (relatively) human-readable string of an equilibrium found by the solver."""
        if self.equilibrium is None:
            print('Please solve for an equilibrium first.')
            return

        string = ""
        for state in range(self.game.num_states):
            string += f'+++++++ state{state} +++++++\n'
            for player in range(self.game.num_players):
                v = self.equilibrium['values'][state, player]
                sigma = self.equilibrium['strategies'][state, player, :self.game.nums_actions[state, player]]
                string += f'player{player}: v={v:#5.2f}, ' \
                          f's={np.array2string(sigma, formatter={"float_kind": lambda x: "%.3f" % x})}\n'
        return string

    def plot_path(self, max_plotted=1000):
        if not self.solver or not self.solver.path:
            print('No solver or no stored path.')
            return
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print('Path cannot be plotted: Package matplotlib is required.')
            return None

        path = self.solver.path
        if path.index > max_plotted:
            sample_freq = int(np.ceil(max_plotted / path.index))
        else:
            sample_freq = 1
        rows = slice(0, path.index, sample_freq)

        s_plot = path.s[rows]
        t_plot = path.y[-1][rows]

        num_rows = len(s_plot)
        sigma_plot = np.empty((num_rows, self.game.num_states, self.game.num_players, self.game.num_actions_max))
        for row in range(path.index)[rows]:
            sigma_plot[row, :] = self.y_to_sigma_V_t(path.y[row])[0]


class LogStratHomotopy(SGameHomotopy):
    """Base class for homotopies using logarithmized strategies
    (i.e. y contains beta := log(sigma), rather than sigma).
    """

    def sigma_V_t_to_y(self, sigma: np.ndarray, V: np.ndarray, t: Union[float, int]) -> np.ndarray:
        """Translate arrays representing strategies sigma, values V and homotopy parameter t to a vector y.
        (Version for homotopies operating on logarithmized strategies.)
        """
        beta_flat = np.log(self.game.flatten_strategies(sigma))
        V_flat = self.game.flatten_values(V)
        return np.concatenate([beta_flat, V_flat, [t]])

    def y_to_sigma_V_t(self, y: np.ndarray, zeros: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
        """Translate a vector y to arrays representing strategies sigma, values V and homotopy parameter t.
        (Version for homotopies operating on logarithmized strategies.)
        """
        sigma = self.game.unflatten_strategies(np.exp(y[0:self.game.num_actions_total]), zeros=zeros)
        V = self.game.unflatten_values(y[self.game.num_actions_total:-1])
        t = y[-1]
        return sigma, V, t

    def distance(self, y_new, y_old):
        """Calculates the distance in strategies sigma between y_old and y_new,
        in the maximum norm, normalized by distance in homotopy parameter t."""
        sigma_difference = np.exp(y_new[:self.game.num_actions_total]) - np.exp(y_old[:self.game.num_actions_total])
        sigma_distance = np.max(np.abs(sigma_difference))
        return sigma_distance / np.abs(y_new[-1] - y_old[-1])
