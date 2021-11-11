"""Create random game."""


from typing import Optional

import numpy as np


def create_random_game(
    num_states: int = 3,
    num_players: int = 3,
    num_actions_min: int = 2,
    num_actions_max: int = 4,
    discount_factor_min: float = 0.93,
    discount_factor_max: float = 0.97,
    rng: Optional[np.random.RandomState] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:

    if rng is None:
        rng = np.random.RandomState()

    nums_actions = rng.random_integers(
        low=num_actions_min,
        high=num_actions_max,
        size=(num_states, num_players),
    )

    payoff_matrices = [
        rng.random_sample((num_players, *nums_actions[s, :])) for s in range(num_states)
    ]

    transition_matrices = [
        rng.exponential(scale=1, size=(*nums_actions[s, :], num_states))
        for s in range(num_states)
    ]
    for s in range(num_states):
        for index, value in np.ndenumerate(np.sum(transition_matrices[s], axis=-1)):
            transition_matrices[s][index] *= 1 / value

    discount_factors = rng.uniform(
        low=discount_factor_min, high=discount_factor_max, size=num_players
    )

    return payoff_matrices, transition_matrices, discount_factors


# %% testing


if __name__ == '__main__':

    u, phi, delta = create_random_game()
