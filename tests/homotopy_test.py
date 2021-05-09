"""Test implementation of homotopies."""

import numpy as np
import pytest

from dsgamesolver.sgame import SGame
from dsgamesolver.qre import QRE_np, QRE_ct
from tests.random_game import create_random_game


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

    random_game = SGame(*create_random_game())
    random_point = np.random.random(
        random_game.num_actions_total
        + random_game.num_states * random_game.num_players
        + 1
    )
    hom_np = QRE_np(random_game)
    hom_ct = QRE_ct(random_game)

    def test_H_zero_at_starting_point(cls):
        # TODO
        pass

    # def test_H_numpy_equal_cython(cls):
    #     assert np.allclose(
    #         cls.hom_np.H(cls.random_point), cls.hom_ct.H(cls.random_point)
    #     )

    # def test_J_numpy_equal_cython(cls):
    #     assert np.allclose(
    #         cls.hom_np.J(cls.random_point), cls.hom_ct.J(cls.random_point)
    #     )
