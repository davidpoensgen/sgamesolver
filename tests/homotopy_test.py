"""Test implementation of homotopies."""

import numpy as np
import pytest

from dsgamesolver.sgame import sGame
from dsgamesolver.homotopy import QRE_np, QRE_ct


# %% helpers


def create_random_game(
    num_states: int = 3,
    num_players: int = 3,
    num_actions_min: int = 2,
    num_actions_max: int = 4,
    discount_factor_min: float = 0.92,
    discount_factor_max: float = 0.98,
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


# %% test parameters


@pytest.fixture(params=["qre"])
def homotopy(request):
    return request.param


@pytest.fixture(params=["np", "ct"])
def implementation(request):
    return request.param


@pytest.fixture(params=["uniform", "random"])
def starting_point(request):
    return request.param


# %% test homotopy creation


class TestQRE:

    random_game = sGame(*create_random_game())
    random_point = np.random.random(
        random_game.num_actions_total
        + random_game.num_states * random_game.num_players
        + 1
    )
    hom_np = QRE_np(random_game)
    hom_ct = QRE_ct(random_game)

    # def test_H_zero_at_starting_point(cls):

    def test_H_numpy_equal_cython(cls):
        assert np.allclose(
            cls.hom_np.H(cls.random_point), cls.hom_ct.H(cls.random_point)
        )

    def test_J_numpy_equal_cython(cls):
        assert np.allclose(
            cls.hom_np.J(cls.random_point), cls.hom_ct.J(cls.random_point)
        )
