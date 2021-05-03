import numpy as np


class sGame():
    """A stochastic game."""

    def __init__(self, payoff_matrices, transition_matrices=None, discount_factors=0.0):

        # read out game shape

        self.payoff_matrices = [payoff_matrices[s] for s in range(len(payoff_matrices))]

        self.num_states = len(self.payoff_matrices)
        self.num_players = self.payoff_matrices[0].shape[0]

        self.nums_actions = np.zeros((self.num_states, self.num_players), dtype=np.int32)
        for s in range(self.num_states):
            for p in range(self.num_players):
                self.nums_actions[s, p] = self.payoff_matrices[s].shape[1 + p]

        self.num_actions_max = self.nums_actions.max()
        self.num_actions_total = self.nums_actions.sum()
        # self.num_actionProfiles = np.product(self.nums_actions, axis=1).sum()
        # TODO ^ needed?

        # action_mask allows to convert between a jagged and a flat array containing a strategy profile (or similar)
        self.action_mask = np.zeros((self.num_states, self.num_players,self.num_actions_max), dtype=bool)
        for s in range(self.num_states):
            for p in range(self.num_players):
                self.action_mask[s, p, 0:self.nums_actions[s, p]] = 1

        # prepare array representing utilities
        self.u_min = min([payoffMatrix.min() for payoffMatrix in self.payoff_matrices])
        self.u_max = max([payoffMatrix.max() for payoffMatrix in self.payoff_matrices])

        self.u = np.nan * np.ones((self.num_states, self.num_players, *[self.num_actions_max] * self.num_players))
        for s in range(self.num_states):
            for p in range(self.num_players):
                for A in np.ndindex(*self.nums_actions[s]):
                    self.u[(s, p) + A] = self.payoff_matrices[s][(p,) + A]

        self.u_norm_with_nan = self.normalize(self.u)
        self.u_norm = copy_without_nan(self.u_norm_with_nan)


        # generate discount factors
        if isinstance(discount_factors, (list, tuple, np.ndarray)):
            self.discount_factors = np.array(discount_factors, dtype=np.float64)
        else:
            self.discount_factors = discount_factors * np.ones(self.num_players)

        # prepare array representing delta * phi (transition probabilities, incorporates individual discount factors)
        # generate transitionArray including discounting
        # transitionArray: [s,p,A,s']
        # (player index due to potentially different discount factors)
        # TODO: why not use discount factors*phi instead? (guess that'd need a lot of fixing)

        if transition_matrices is None:
            self.transition_matrices = [np.ones((*self.nums_actions[s], self.num_states))
                                        for s in range(self.num_states)]
            for s in range(self.num_states):
                for index, value in np.ndenumerate(np.sum(self.transition_matrices[s], axis=-1)):
                    self.transition_matrices[s][index] *= 1 / value
            # TODO: what exactly does this do? is it for (separated) repeated games?
        else:
            self.transition_matrices = []
            for s in range(self.num_states):
                self.transition_matrices.append(np.array(transition_matrices[s], dtype=np.float64))

        transitions = np.nan * np.ones(shape=(self.num_states,
                                              *[self.num_actions_max] * self.num_players, self.num_states))
        for s0 in range(self.num_states):
            for A in np.ndindex(*self.nums_actions[s0]):
                for s1 in range(self.num_states):
                    transitions[(s0,) + A + (s1,)] = self.transition_matrices[s0][A + (s1,)]

        self.transitionArray = np.nan * np.ones(
            shape=(self.num_states, self.num_players, *[self.num_actions_max] * self.num_players, self.num_states),
            dtype=np.float64)
        for p in range(self.num_players):
            self.transitionArray[:, p] = self.discount_factors[p] * transitions
        self.transitionArray_withNaN = self.transitionArray.copy()
        self.transitionArray[np.isnan(self.transitionArray)] = 0.0
        # TODO: reason to keep this both with and without NaN?
        # TODO: seems that: einsum wants 0s. find_y0_qre wants NaN
        # TODO: can now use get_values etc. for find y0. Any reason left to keep versions with NaN
        # TODO: If kept, can use copy_without_nan

        self.phi = self.transitionArray
        # TODO: unify notation: phi / u / etc, transitionArray etc

    def detect_symmetries(self):
        """Detect symmetries between agents. (TBD) """
        pass

    def normalize(self, u):
        """ Normalize u to values between 0 and 1: u_norm = (u - u_m)/(u_max - u_min).
            u may be a scalar or np.array.
        """
        return (u - self.u_min) / (self.u_max - self.u_min)

    def denormalize(self, u_norm):
        """ Calculate de-normalized utilities. u may be a scalar or np.array."""
        return u_norm * (self.u_max - self.u_min) + self.u_min

    def get_values(self, strategy_profile):
        """Calculate state-player values for a given strategy profile."""

        sigma = copy_without_nan(strategy_profile)
        sigma_list = [sigma[:, p, :] for p in range(self.num_players)]

        u = copy_without_nan(self.u)

        ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        einsum_eq_u = ('sp' + ABC[0:self.num_players] + ',s' +
                       ',s'.join([ABC[p] for p in range(self.num_players)]) + '->sp')
        einsum_eq_phi = ('sp' + ABC[0:self.num_players] + 't,s' +
                         ',s'.join([ABC[p] for p in range(self.num_players)]) + '->spt')

        u = np.einsum(einsum_eq_u, u, *sigma_list)
        phi = np.einsum(einsum_eq_phi, self.transitionArray, *sigma_list)

        values = np.empty((self.num_states, self.num_players))

        for p in range(self.num_players):
            A = np.eye(self.num_states) - phi[:, p, :]
            values[:, p] = np.linalg.solve(A, u[:, p])

        return values

    def random_strategy_profile(self):
        """Generate a random strategy profile."""

        strategy_profile = np.nan * np.empty((self.num_states, self.num_players, self.num_actions_max))
        for s in range(self.num_states):
            for p in range(self.num_players):
                sigma = np.random.exponential(scale=1, size=self.nums_actions[s, p])
                sigma = sigma / sigma.sum()
                strategy_profile[s, p, 0:self.nums_actions[s, p]] = sigma

        return strategy_profile

    def centroid_strategy_profile(self):
        """Returns the centroid strategy profile."""

        strategy_profile = np.nan * np.empty((self.num_states, self.num_players, self.num_actions_max))
        for s in range(self.num_states):
            for p in range(self.num_players):
                strategy_profile[s, p, 0:self.nums_actions[s, p]] = 1 / self.nums_actions[s, p]

        return strategy_profile

    def flatten(self, array):
        """Convert a jagged array of shape (states, players, max_actions) to a flat array, removing all Nans."""
        return np.extract(self.action_mask, array)

    def unflatten(self, array):
        """Convert a flat array containing a strategy profile or similar to an array with
        shape (states, players, max_actions), padded with NaNs where necessary.
        """
        out = np.nan * np.empty((self.num_states, self.num_players, self.num_actions_max))
        np.place(out, self.action_mask, array)
        return out

    def check_equilibrium(self, strategy_profile):
        """Calculates "epsilon-equilibriumness" (max total utility any agent could gain by deviating)
        of a given strategy profile.
         """
        sigma = copy_without_nan(strategy_profile)
        u = copy_without_nan(self.u)
        values = self.get_values(sigma)

        # u_tilde: normal form games that include continuation values.
        u_tilde = u + np.einsum('sp...S,Sp->sp...', self.transitionArray, values)

        losses = np.empty((self.num_states, self.num_players))

        ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for p in range(self.num_players):
            others = [q for q in range(self.num_players) if q != p]
            einsum_eq = ('s' + ABC[0:self.num_players] + ',s' +
                         ',s'.join([ABC[q] for q in others]) + '->s' + ABC[p])
            action_values = np.einsum(einsum_eq, u_tilde[:, p, :], *[sigma[:, q, :] for q in others])

            losses[:, p] = action_values.max(axis=-1) - values[:, p]

        # TODO: decide whether this should be aggregated (player/agent/...?)
        # TODO: Should this function report? Return? etc.
        return losses


def copy_without_nan(array: np.ndarray):
    """Copy input array and replace all NaNs with 0."""
    out = array.copy()
    out[np.isnan(out)] = 0
    return out

