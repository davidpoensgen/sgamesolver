"""CHANGELOG (to be deleted after approval)


sGame:
------

Variable names:
    - inputs, also saved as class attributes:
        payoff_matrices, transition_matrices, discount_factors
    - class attributes, for computation of homotopy:
        payoffs, payoffs_normalized, payoffs_with_nan, payoffs_normalized_with_nan, transitions, etc.
    - short notation for internal computation within functions, never used by user:
        u, phi, delta, etc.

Allow array-like inputs.

Use np.nan_to_num instead of custom function copy_without_nan, as suggested.

Attribute transitions_with_nan removed, as suggested.

Transitions include discounting, as before.
Having discount factors separately would result in unnecessary multiplications with every call of H and J.

ABC as constant outside of class.

Normalization and de-normalization of payoffs without nan placeholders, using payoff_mask.

Added type hints.

Added unit tests.



sGameHomotopy:
--------------

sGameHomotopy into this file.

Added type hints.

Question: Will y always consist of sigma, V and t?

"""

from typing import List, Tuple, Union, Optional

import numpy as np
from numpy.typing import ArrayLike

ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# %% game class


class SGame():
    """A stochastic game."""

    def __init__(self, payoff_matrices: List[ArrayLike], transition_matrices: Optional[List[ArrayLike]] = None,
                 discount_factors: Union[ArrayLike, float, int] = 0.0) -> None:
        """Inputs:

        payoff_matrices:      list of array-like, one for each state: payoff_matrices[s][p,A]

        transition_matrices:  list of array-like, one for each from_state: transition_matrices[s][p,A,s']
                              or None (-> separated repeated game)

        discount_factors:     array-like: discount_factors[p]
                              or numeric (-> common discount factor)

        Different numbers of actions across states and players are allowed.
        Inputs should be of relevant dimension and should NOT contain nan.
        """

        # bring payoff_matrices to list of np.ndarray, one array for each state
        self.payoff_matrices = [np.array(payoff_matrices[s], dtype=np.float64) for s in range(len(payoff_matrices))]

        # read out game shape
        self.num_states = len(self.payoff_matrices)
        self.num_players = self.payoff_matrices[0].shape[0]

        self.nums_actions = np.zeros((self.num_states, self.num_players), dtype=np.int32)
        for s in range(self.num_states):
            for p in range(self.num_players):
                self.nums_actions[s, p] = self.payoff_matrices[s].shape[1 + p]

        self.num_actions_max = self.nums_actions.max()
        self.num_actions_total = self.nums_actions.sum()

        # action_mask allows to convert between a jagged and a flat array containing a strategy profile (or similar)
        self.action_mask = np.zeros((self.num_states, self.num_players, self.num_actions_max), dtype=bool)
        for s in range(self.num_states):
            for p in range(self.num_players):
                self.action_mask[s, p, 0:self.nums_actions[s, p]] = 1

        # payoff_mask allows to normalize and de-normalize payoffs
        self.payoff_mask = np.zeros((self.num_states, self.num_players, *[self.num_actions_max]*self.num_players),
                                    dtype=bool)
        for s in range(self.num_states):
            for p in range(self.num_players):
                for A in np.ndindex(*self.nums_actions[s]):
                    self.payoff_mask[(s, p) + A] = 1

        # generate array representing payoffs [s,p,A]
        self.payoffs = np.zeros((self.num_states, self.num_players, *[self.num_actions_max]*self.num_players),
                                dtype=np.float64)
        for s in range(self.num_states):
            for p in range(self.num_players):
                for A in np.ndindex(*self.nums_actions[s]):
                    self.payoffs[(s, p) + A] = self.payoff_matrices[s][(p,) + A]

        # TODO: delete once unnormalized payoffs are used throughout
        # self.payoff_min = self.payoffs[self.payoff_mask].min()
        # self.payoff_max = self.payoffs[self.payoff_mask].max()
        # self.payoffs_normalized = self.normalize_payoffs(self.payoffs)

        # generate array representing discount factors [p]
        if isinstance(discount_factors, (list, tuple, np.ndarray)):
            self.discount_factors = np.array(discount_factors, dtype=np.float64)
        else:
            self.discount_factors = discount_factors * np.ones(self.num_players, dtype=np.float64)

        # define scale for adjusting tracking parameters
        # TODO

        # bring transition_matrices to list of np.ndarray, one array for each state
        if transition_matrices is not None:
            self.transition_matrices = [np.array(transition_matrices[s], dtype=np.float64)
                                        for s in range(self.num_states)]
        else:
            # If no transitions are specified, specification will default to separated repeated games:
            # phi(s,s') = 1 if s==s' and 0 else, for all action profiles.
            self.transition_matrices = []
            for s in self.num_states:
                phi_s = np.zeros((*self.nums_actions[s], self.num_states), dtype=np.float64)
                phi_s[..., s] = 1
                self.transition_matrices.append(phi_s)

        # build big transition matrix [s,A,s'] from list of small transition matrices [A,s'] for each s
        transition_matrix = np.zeros((self.num_states, *[self.num_actions_max]*self.num_players, self.num_states),
                                     dtype=np.float64)
        for s0 in range(self.num_states):
            for A in np.ndindex(*self.nums_actions[s0]):
                for s1 in range(self.num_states):
                    transition_matrix[(s0,)+A+(s1,)] = self.transition_matrices[s0][A+(s1,)]

        # generate array representing transitions, including discounting: delta * phi [s,p,A,s']
        # (player index due to potentially player-specific discount factors)
        self.transitions = np.zeros((self.num_states, self.num_players, *[self.num_actions_max]*self.num_players,
                                     self.num_states), dtype=np.float64)
        for p in range(self.num_players):
            self.transitions[:, p] = self.discount_factors[p] * transition_matrix

    def detect_symmetries(self) -> None:
        """Detect symmetries between agents."""
        # TODO: tbd
        pass

    # TODO: delete once unnormalized payoffs are used throughout
    # def normalize_payoffs(self, payoffs: ArrayLike) -> np.ndarray:
    #     """Normalize payoffs to values between 0 and 1. Keep zeros for nonexisting actions."""
    #     payoffs_normalized = (np.array(payoffs)-self.payoff_min) / (self.payoff_max-self.payoff_min)
    #     payoffs_normalized[~self.payoff_mask] = 0.0
    #     return payoffs_normalized
    #
    # def denormalize_payoffs(self, payoffs_normalized: ArrayLike) -> np.ndarray:
    #     """Calculate de-normalized payoffs. Keep zeros for nonexisting actions."""
    #     payoffs = self.payoff_min + np.array(payoffs_normalized) * (self.payoff_max-self.payoff_min)
    #     payoffs[~self.payoff_mask] = 0.0
    #     return payoffs

    def random_strategy(self) -> np.ndarray:
        """Generate a random strategy profile."""

        strategy_profile = np.nan * np.empty((self.num_states, self.num_players, self.num_actions_max),
                                             dtype=np.float64)
        for s in range(self.num_states):
            for p in range(self.num_players):
                sigma = np.random.exponential(scale=1, size=self.nums_actions[s, p])
                sigma = sigma / sigma.sum()
                strategy_profile[s, p, :self.nums_actions[s, p]] = sigma

        return strategy_profile

    def centroid_strategy(self) -> np.ndarray:
        """Generate the centroid strategy profile."""

        strategy_profile = np.nan * np.empty((self.num_states, self.num_players, self.num_actions_max),
                                             dtype=np.float64)
        for s in range(self.num_states):
            for p in range(self.num_players):
                strategy_profile[s, p, :self.nums_actions[s, p]] = 1 / self.nums_actions[s, p]

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
            strategies = np.zeros((self.num_states, self.num_players, self.num_actions_max), dtype=np.float64)
        else:
            strategies = np.nan * np.empty((self.num_states, self.num_players, self.num_actions_max), dtype=np.float64)
        np.place(strategies, self.action_mask, strategies_flat)
        return strategies

    # TODO: delete once unnormalized payoffs are used throughout
    # def get_values(self, strategy_profile: ArrayLike, normalized: bool = False) -> np.ndarray:
    def get_values(self, strategy_profile: ArrayLike) -> np.ndarray:
        """Calculate state-player values for a given strategy profile."""

        sigma = np.nan_to_num(strategy_profile)
        sigma_list = [sigma[:, p, :] for p in range(self.num_players)]

        # TODO: delete once unnormalized payoffs are used throughout
        # payoffs = self.payoffs_normalized if normalized else self.payoffs

        einsum_eq_u = ('sp' + ABC[0:self.num_players] + ',s' +
                       ',s'.join(ABC[p] for p in range(self.num_players)) + '->sp')
        einsum_eq_phi = ('sp' + ABC[0:self.num_players] + 't,s' +
                         ',s'.join(ABC[p] for p in range(self.num_players)) + '->spt')

        # TODO: delete once unnormalized payoffs are used throughout
        # u = np.einsum(einsum_eq_u, payoffs, *sigma_list)
        u = np.einsum(einsum_eq_u, self.payoffs, *sigma_list)
        phi = np.einsum(einsum_eq_phi, self.transitions, *sigma_list)

        values = np.empty((self.num_states, self.num_players), dtype=np.float64)
        try:
            for p in range(self.num_players):
                A = np.eye(self.num_states) - phi[:, p, :]
                values[:, p] = np.linalg.solve(A, u[:, p])
        except np.linalg.LinAlgError:
            raise("Failed to solve for state-player values: Transition matrix not invertible.")
        return values

    # TODO: delete once unnormalized payoffs are used throughout
    # def normalize_values(self, values: Union[ArrayLike, float, int]) -> Union[np.ndarray, float]:
    #     """Normalize values to [0,1]."""
    #     return (np.array(values)-self.payoff_min) / (self.payoff_max-self.payoff_min)
    #
    # def denormalize_values(self, values_normalized: Union[ArrayLike, float, int]) -> Union[np.ndarray, float]:
    #     """Calculate de-normalized values. Argument may be a scalar or np.ndarray."""
    #     return self.payoff_min + np.array(values_normalized) * (self.payoff_max-self.payoff_min)

    def flatten_values(self, values: ArrayLike) -> np.ndarray:
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

        losses = np.empty((self.num_states, self.num_players), dtype=np.float64)

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
    """General homotopy class for some sGame."""

    def __init__(self, game: SGame) -> None:
        self.game = game
        self.y0 = None
        self.tracking_parameters = {}
        self.solver = None

    def initialize(self) -> None:
        """Any steps in preparation to start solver:
        - set priors, weights etc. if needed
        - set starting point y0
        - prepare symmetry helpers
            + (make sure priors and other parameters are in accordance with symmetries)
        - set up homCont to solve the game
        """
        pass

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

    def x_transformer(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Transform vector y to vector x.

        Vector y is used during path tracing.
        Vector x is used to check for convergence.

        Typical use case: Strategies are relevant for convergence, but are transformed during tracing.
        Example: QRE, with uses log strategies beta=log(sigma) during tracing.

        Note: If not needed, can simply pass None to HomCont.
        """
        pass

    def sigma_V_t_to_y(self, sigma: np.ndarray, V: np.ndarray, t: Union[float, int]) -> np.ndarray:
        """Translate arrays representing strategies sigma, values V and homotopy parameter t to a vector y."""
        pass

    def y_to_sigma_V_t(self, y: np.ndarray, zeros: bool = False) -> Tuple[np.ndarray, np.ndarray, Union[float, int]]:
        """Translate a vector y to arrays representing strategies sigma, values V and homotopy parameter t."""
        pass
