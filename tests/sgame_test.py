"""Test SGame class."""

import numpy as np

from sgamesolver.sgame import SGame
from tests.random_game import create_random_game


# %% test SGame class


class TestSGame:

    num_states = 3
    num_players = 3
    num_actions_min = 2
    num_actions_max = 4
    discount_factor_min = 0.93
    discount_factor_max = 0.97

    payoff_matrices, transition_matrices, discount_factors = create_random_game(
        num_states, num_players, num_actions_min, num_actions_max, discount_factor_min, discount_factor_max
    )
    game = SGame(payoff_matrices, transition_matrices, discount_factors)

    def test_shape_of_game(cls):
        assert cls.game.num_states == cls.num_states
        assert cls.game.num_players == cls.num_players
        assert cls.game.num_actions_max <= cls.num_actions_max

    def test_nums_actions(cls):
        assert cls.game.nums_actions.shape == (cls.num_states, cls.num_players)
        assert (cls.game.nums_actions >= cls.num_actions_min).all()
        assert (cls.game.nums_actions <= cls.num_actions_max).all()

    def test_payoffs(cls):
        assert isinstance(cls.game.payoffs, np.ndarray)
        assert not np.isnan(cls.game.payoffs).any()
        # dimensions:
        assert cls.game.payoffs.shape[:2] == (cls.num_states, cls.num_players)
        assert (np.array(cls.game.payoffs.shape[2:]) >= cls.num_actions_min).all()
        assert (np.array(cls.game.payoffs.shape[2:]) <= cls.num_actions_max).all()
        # sum:
        assert np.allclose(sum(matrix.sum() for matrix in cls.payoff_matrices), cls.game.payoffs.sum())

    def test_transitions(cls):
        cls.game._make_transitions()
        assert isinstance(cls.game.transitions, np.ndarray)
        assert not np.isnan(cls.game.transitions).any()
        # dimensions:
        assert cls.game.transitions.shape[:2] == (cls.num_states, cls.num_players)
        assert (np.array(cls.game.transitions.shape[2:]) >= cls.num_actions_min).all()
        assert (np.array(cls.game.transitions.shape[2:]) <= cls.num_actions_max).all()
        assert cls.game.transitions.shape[-1] == cls.num_states
        # sum and range:
        transitions_sum = sum(matrix.sum() for matrix in cls.transition_matrices) * cls.num_players
        assert cls.discount_factor_min * transitions_sum <= cls.game.transitions.sum()
        assert cls.discount_factor_max * transitions_sum >= cls.game.transitions.sum()
        assert cls.game.transitions.min() >= 0.0
        assert cls.game.transitions.max() <= 1.0

    def test_discount_factors(cls):
        assert isinstance(cls.game.discount_factors, np.ndarray)
        assert not np.isnan(cls.game.payoffs).any()
        assert cls.game.discount_factors.shape == (cls.num_players,)
        assert cls.game.discount_factors.min() >= 0.0
        assert cls.game.discount_factors.max() <= 1.0

    def test_symmetries(cls):
        # TODO
        pass

    def test_strategies(cls):
        centroid_strategy = cls.game.centroid_strategy()
        random_strategy = cls.game.random_strategy()
        strategies_flattened = cls.game.flatten_strategies(random_strategy)
        strategies_unflattened = cls.game.unflatten_strategies(strategies_flattened)
        assert centroid_strategy.shape == (cls.num_states, cls.num_players, cls.game.num_actions_max)
        assert np.allclose(np.nansum(centroid_strategy), cls.num_states*cls.num_players)
        assert random_strategy.shape == (cls.num_states, cls.num_players, cls.game.num_actions_max)
        assert np.allclose(np.nansum(random_strategy), cls.num_states*cls.num_players)
        assert strategies_flattened.shape == (cls.game.num_actions_total,)
        assert np.allclose(strategies_unflattened, random_strategy, equal_nan=True)

    def test_values(cls):
        random_strategy = cls.game.random_strategy()
        values = cls.game.get_values(random_strategy)
        values_flattened = cls.game.flatten_values(values)
        values_unflattened = cls.game.unflatten_values(values_flattened)
        assert values.shape == (cls.num_states, cls.num_players)
        assert values_flattened.shape == (cls.num_states*cls.num_players,)
        assert np.allclose(values_unflattened, values)

    def test_equilibrium_check(cls):
        random_strategy = cls.game.random_strategy()
        assert np.max(np.abs(cls.game.check_equilibrium(random_strategy))) < 1


# %% run


if __name__ == '__main__':

    test_sgame = TestSGame()

    test_sgame.test_shape_of_game()
    test_sgame.test_nums_actions()
    test_sgame.test_payoffs()
    test_sgame.test_transitions()
    test_sgame.test_discount_factors()
    test_sgame.test_symmetries()
    test_sgame.test_strategies()
    test_sgame.test_values()
    test_sgame.test_equilibrium_check()
