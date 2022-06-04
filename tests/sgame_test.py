"""Test SGame class."""


import numpy as np
import sgamesolver


# %% test SGame class


class TestSGame:

    num_states = 3
    num_players = 3
    num_actions = 3
    discount_factor = 0.95

    game = sgamesolver.SGame.random_game(num_states=num_states, num_players=num_players, num_actions=num_actions,
                                         delta=discount_factor)

    def test_shape_of_game(self):
        assert self.game.num_states == self.num_states
        assert self.game.num_players == self.num_players
        assert self.game.num_actions_max == self.num_actions
        assert self.game.num_actions_total == self.num_states * self.num_players * self.num_actions

    def test_nums_actions(self):
        assert self.game.nums_actions.shape == (self.num_states, self.num_players)
        assert (self.game.nums_actions == self.num_actions).all()

    def test_payoffs(self):
        assert isinstance(self.game.u, np.ndarray)
        assert not np.isnan(self.game.u).any()
        assert self.game.u.shape[:2] == (self.num_states, self.num_players)
        assert (np.array(self.game.u.shape[2:]) == self.num_actions).all()

    def test_transitions(self):
        self.game._make_transitions()
        assert isinstance(self.game.transitions, np.ndarray)
        assert not np.isnan(self.game.transitions).any()
        assert self.game.transitions.shape[:2] == (self.num_states, self.num_players)
        assert (np.array(self.game.transitions.shape[2:]) == self.num_actions).all()
        assert self.game.transitions.shape[-1] == self.num_states
        # sum and range:
        assert np.allclose(self.discount_factor * self.num_players * self.game.phi.sum(), self.game.transitions.sum())
        assert self.game.transitions.min() >= 0.0
        assert self.game.transitions.max() <= 1.0

    def test_discount_factors(self):
        assert isinstance(self.game.delta, np.ndarray)
        assert not np.isnan(self.game.u).any()
        assert self.game.delta.shape == (self.num_players,)
        assert self.game.delta.min() >= 0.0
        assert self.game.delta.max() <= 1.0

    def test_symmetries(self):
        # TODO
        pass

    def test_strategies(self):
        centroid_strategy = self.game.centroid_strategy()
        random_strategy = self.game.random_strategy()
        strategies_flattened = self.game.flatten_strategies(random_strategy)
        strategies_unflattened = self.game.unflatten_strategies(strategies_flattened)
        assert centroid_strategy.shape == (self.num_states, self.num_players, self.num_actions)
        assert np.allclose(np.nansum(centroid_strategy), self.num_states*self.num_players)
        assert random_strategy.shape == (self.num_states, self.num_players, self.num_actions)
        assert np.allclose(np.nansum(random_strategy), self.num_states*self.num_players)
        assert strategies_flattened.shape == (self.game.num_actions_total,)
        assert np.allclose(strategies_unflattened, random_strategy, equal_nan=True)

    def test_values(self):
        random_strategy = self.game.random_strategy()
        values = self.game.get_values(random_strategy)
        values_flattened = self.game.flatten_values(values)
        values_unflattened = self.game.unflatten_values(values_flattened)
        assert values.shape == (self.num_states, self.num_players)
        assert values_flattened.shape == (self.num_states*self.num_players,)
        assert np.allclose(values_unflattened, values)

    def test_equilibrium_check(self):
        random_strategy = self.game.random_strategy()
        assert np.max(np.abs(self.game.check_equilibrium(random_strategy))) < 1

    def test_table_conversion(self):
        game_table = self.game.to_table()
        reloaded_game = sgamesolver.SGame.from_table(table=game_table)
        assert np.allclose(reloaded_game.u, self.game.u, equal_nan=True)
        assert np.allclose(reloaded_game.phi, self.game.phi, equal_nan=True)
        assert np.allclose(reloaded_game.delta, self.game.delta)


# %% run


if __name__ == '__main__':

    test_class = TestSGame()

    method_names = [method for method in dir(test_class)
                    if callable(getattr(test_class, method))
                    if not method.startswith('__')]
    for method in method_names:
        getattr(test_class, method)()
