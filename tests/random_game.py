"""Create random game."""

import numpy as np


def create_random_game(
    num_states: int = 3,
    num_players: int = 3,
    num_actions_min: int = 2,
    num_actions_max: int = 4,
    discount_factor_min: float = 0.93,
    discount_factor_max: float = 0.97,
):
    nums_actions = np.random.randint(
        low=num_actions_min,
        high=num_actions_max + 1,
        size=(num_states, num_players),
        dtype=np.int32,
    )
    payoff_matrices = [
        np.random.random((num_players, *nums_actions[s, :])) for s in range(num_states)
    ]
    transition_matrices = [
        np.random.exponential(scale=1, size=(*nums_actions[s, :], num_states))
        for s in range(num_states)
    ]
    for s in range(num_states):
        for index, value in np.ndenumerate(np.sum(transition_matrices[s], axis=-1)):
            transition_matrices[s][index] *= 1 / value
    discount_factors = np.random.uniform(
        low=discount_factor_min, high=discount_factor_max, size=num_players
    )
    return payoff_matrices, transition_matrices, discount_factors
