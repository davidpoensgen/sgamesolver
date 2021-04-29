import numpy as np


class sGame():
    """ A stochastic game."""

    def __init__(
            self,
            payoff_matrices,
            transition_matrices=None,
            discount_factors=0.0):

        # read out game shape

        self.payoff_matrices = [payoff_matrices[s] for s in range(len(payoff_matrices))]

        self.num_states = len(self.payoff_matrices)
        self.num_players = self.payoff_matrices[0].shape[0]

        self.nums_actions = np.zeros(shape=(self.num_states, self.num_players), dtype=np.int32)
        for s in range(self.num_states):
            for p in range(self.num_players):
                self.nums_actions[s, p] = self.payoff_matrices[s].shape[1 + p]

        self.num_actions_max = self.nums_actions.max()
        self.num_actions_total = self.nums_actions.sum()
        self.num_actionProfiles = np.product(self.nums_actions, axis=1).sum()

        # prepare array representing utilities

        self.u_min = min([payoffMatrix.min() for payoffMatrix in self.payoff_matrices])
        self.u_max = max([payoffMatrix.max() for payoffMatrix in self.payoff_matrices])

        self.u = np.nan * np.ones((self.num_states, self.num_players, *[self.num_actions_max] * self.num_players),
                                  dtype=np.float64)
        for s in range(self.num_states):
            for p in range(self.num_players):
                for A in np.ndindex(*self.nums_actions[s]):
                    self.u[(s, p) + A] = self.payoff_matrices[s][(p,) + A]

        self.u_norm = self.normalize(self.u)
        self.u_norm_with_nan = self.u_norm.copy()
        self.u_norm[np.isnan(self.u_norm)] = 0.0

        # generate discount factors

        if type(discount_factors) != type([]) and type(discount_factors) != np.ndarray:
            self.discount_factors = discount_factors * np.ones(self.num_players, dtype=np.float64)
        else:
            self.discount_factors = np.array(discount_factors, dtype=np.float64)

        # prepare array representing delta * phi (transition probabilities, incorporates individual discount factors)
        # generate transitionArray including discounting
        # transitionArray: [s,p,A,s']
        # (player index due to potentially different discount factors)

        if transition_matrices is None:
            self.transition_matrices = [np.ones(shape=(*self.nums_actions[s], self.num_states), dtype=np.float64)
                                        for s in range(self.num_states)]
            for s in range(self.num_states):
                for index, value in np.ndenumerate(np.sum(self.transition_matrices[s], axis=-1)):
                    self.transition_matrices[s][index] *= 1 / value
            # TODO: what exactly does this do? is it for (separated) repeated games?
        else:
            self.transition_matrices = []
            for s in range(self.num_states):
                self.transition_matrices.append(np.array(transition_matrices[s], dtype=np.float64))

        transitions = np.nan * np.ones(
            shape=(self.num_states, *[self.num_actions_max] * self.num_players, self.num_states), dtype=np.float64)
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

        self.phi = self.transitionArray
        # TODO: unify notation: phi / u / etc

    def detect_symmetries(self):
        """ Function to detect symmetries between agents. (TBD) """
        pass

    def normalize(self, u):
        """ Normalize u to values between 0 and 1: u_norm = (u - u_m)/(u_max - u_min).
            u may be a scalar or np.array.
        """
        return (u - self.u_min) / (self.u_max - self.u_min)

    def denormalize(self, u_norm):
        """ Calculate de-normalized utilities. u may be a scalar or np.array.
        TODO: how to de-normalize values? think this is easiest by just using sigma and u ...
        """
        return u_norm * (self.u_max - self.u_min) + self.u_min

    def calculate_values(self, sigma):
        """Calculate state-player values under strategy profile sigma.
        TODO: This could be useful to de-normalize.
        TODO: Could also be used in finding y0 for several of the homotopies.
        """
        pass

    def check_equilibrium(self, sigma):
        """Calculates "epsilon-equilibriumness" (max total utility any agent could gain by deviating)
        of a given strategy profile. TODO: might be a nice feature
        """
