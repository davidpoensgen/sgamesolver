from typing import Union
import pandas as pd
import numpy as np
import itertools
from sgamesolver import SGame


def game_from_table(table: Union[str, pd.DataFrame]) -> SGame:
    """Convert files to DataFrame if needed, then parse the DataFrame to create an SGame."""
    if isinstance(table, pd.DataFrame):
        df = table
        row_offset = 0
    elif isinstance(table, str):
        if table[-5:] == '.xlsx' or table[:-4] == '.xls':
            # read header and body separately; this will preserve duplicate column names
            # otherwise, pandas renames those, leading to cryptic error messages
            cols = pd.read_excel(table, header=None, nrows=1).values[0]
            df = pd.read_excel(table, header=None, skiprows=1, keep_default_na=False)
            df.columns = cols
            row_offset = 2
        elif table[-4:] == '.dta':
            df = pd.read_stata(table)
            row_offset = 1
            pass
        elif table[-4:] in ['.csv', '.txt']:
            cols = pd.read_csv(table, header=None, nrows=1).values[0]
            df = pd.read_csv(table, header=None, skiprows=1, keep_default_na=False)
            df.columns = cols
            row_offset = 2
        else:
            raise ValueError(f'"{table}: Unknown file extension. (Use any of .xlsx/.xls, .dta, .csv/.txt).')
    else:
        raise ValueError('Table needs to be either a pandas DataFrame or a string containing a file path.')

    u, phi, delta, state_labels, player_labels, action_labels = _dataframe_to_game(df, row_offset)
    game = SGame(u, phi, delta)
    game.state_labels = state_labels
    game.player_labels = player_labels
    game.action_labels = action_labels
    return game


def _dataframe_to_game(df: pd.DataFrame, row_offset=0):
    """Function to actually parse the dataframe and convert the information to numpy.ndarrays."""
    # row_offset is to account for (i) header and (ii) 0-indexing; e.g. line 2 in .xlsx corresponds to index 0 of df
    # copy index to a column, so it is preserved over all splits etc.
    df['idx_column'] = df.index

    # read players from a_-columns
    action_col_list = [col for col in df.columns if col[:2] == 'a_']
    player_list = [col[2:] for col in df.columns if col[:2] == 'a_']
    num_p = len(player_list)
    if len(player_list) != len(set(player_list)):
        raise ValueError('"a_-"-columns contain duplicate player suffixes.')
    # assert that u_-column exists for each player extracted from a_-columns
    u_col_list = [col for col in df.columns if col[:2] == 'u_']
    u_player_list = [col[2:] for col in df.columns if col[:2] == 'u_']
    if not set(player_list) == set(u_player_list):
        # could be more specific here and list differences
        raise ValueError('Player suffixes from "a_"-columns do not match player suffixes from "u_"-columns')

    # read and remove delta-row (before states are read)
    delta_row = df[df['state'] == 'delta']
    if len(delta_row) > 1:
        raise ValueError('Table contains more than one "delta"-row.')
    elif len(delta_row) == 0:
        raise ValueError('Table contains no "delta"-row.')
    else:
        try:
            delta = np.empty(num_p) * np.nan
            delta[:] = delta_row[u_col_list]
            # numpy will raise ValueError for strings that cannot be cast to float;
            # empty cells will be NaN, so check for these too:
            if np.isnan(delta).any():
                raise ValueError
        except ValueError:
            raise ValueError('"Delta"-row has missing values or is incorrectly formatted.')
    df = df[df['state'] != 'delta']

    # read states
    if 'state' not in df.columns:
        raise ValueError('Table does not have a "state"-column.')
    # make sure the state labels are strings
    state_list = [str(state) for state in df['state'].unique()]
    state_set = set(state_list)
    num_s = len(state_list)
    # check if transitions are specified correctly
    if 'to_state' in df.columns:
        if [col[4:] for col in df.columns if col[:4] == 'phi_']:
            raise ValueError('Table contains both a "to_state"-column and "phi_"-columns, which is ambiguous.'
                             ' Please remove either.')
        to_state_format = True
        to_state_col_list = None
        # ensure that to_states are reads as strings, even if e.g. a column of integers is passed
        to_state_set = {str(state) for state in df['to_state'].unique()}
        if not to_state_set.issubset(state_set):
            missing_states = to_state_set.difference(state_set)
            raise ValueError(f'The following states appear in column "to_state", but not in "state": {missing_states}')
    else:
        to_state_format = False
        to_state_list = [col[4:] for col in df.columns if col[:4] == 'phi_']
        if len(to_state_list) != len(set(to_state_list)):
            raise ValueError('"phi_"-columns contain duplicate state suffixes.')
        # ! to_state_col_list needs to be ordered like state_list, so that the first index corresponds to state 0 etc.
        to_state_col_list = ['phi_' + state for state in state_list]
        to_state_set = set(to_state_list)
        if to_state_set != state_set:
            phi_extra_states = to_state_set.difference(state_set)
            state_extra_states = state_set.difference(to_state_set)
            message = 'Entries in "state"-column do not match "phi_"-column-suffixes:\n'
            if phi_extra_states:
                message += f'The following states have a "phi_"-column, but no entry in "state": {phi_extra_states}.\n'
            if state_extra_states:
                message += f'The following states have an entry in "state", but no "phi_"-column: {state_extra_states}.'
            raise ValueError(message)

    action_lists_list = []
    u_list = []
    phi_list = []
    error_list = []
    for state in state_list:
        df_state = df[df['state'] == state]
        action_lists = [df_state['a_' + player].unique() for player in player_list]
        action_lists_list.append(action_lists)
        nums_a = [len(action_list) for action_list in action_lists]
        u = np.full([num_p] + nums_a, np.NaN)
        phi = np.full(nums_a + [num_s], np.NaN)

        for index, action_profile in zip(np.ndindex(*nums_a), itertools.product(*action_lists)):
            # find all rows in current state with matching action profile:
            rows = df_state.merge(pd.DataFrame((action_profile,), columns=action_col_list))
            # check for any errors, but finish parsing the table (so that all errors can be reported at once)
            if len(rows) == 0:
                error_list.append(f'Missing > state: {state}, actions: {", ".join(action_profile)}')
                continue
            elif len(rows) > 1:
                row_no = ", ".join(map(str, list(rows['idx_column'] + row_offset)))
                error_list.append(f'Duplicate > state: {state}, '
                                  f'actions: {", ".join(action_profile)} > rows: {row_no}')
                continue
            try:
                u[(slice(None),) + index] = rows[u_col_list]
                # numpy will raise ValueError for strings that cannot be cast to float;
                # empty cells will be NaN, so check for these too:
                if np.isnan(u[(slice(None),) + index]).any():
                    raise ValueError
            except ValueError:
                row_no = ", ".join(map(str, list(rows['idx_column'] + row_offset)))
                error_list.append(f'Format (u) > state: {state}, '
                                  f'actions: {", ".join(action_profile)} > row: {row_no}')
            if to_state_format:
                phi[index + (slice(None),)] = 0
                to_state = state_list.index(str(rows['to_state']))
                phi[index + (to_state,)] = 1
            else:
                try:
                    phi[index + (slice(None),)] = rows[to_state_col_list]
                    if np.isnan(phi[index + (slice(None),)]).any():
                        raise ValueError
                except ValueError:
                    row_no = ", ".join(map(str, list(rows['idx_column'] + row_offset)))
                    error_list.append(f'Format (phi) > state: {state}, '
                                      f'actions: {", ".join(action_profile)} > row: {row_no}')

        u_list.append(u)
        phi_list.append(phi)

    if error_list:  # now, raise an error if any action profiles had issues:
        message = 'The table has missing or duplicate action profiles; or missing/illegal values for u or phi:'
        for error in error_list:
            message += '\n' + error
        raise ValueError(message)

    return u_list, phi_list, delta, state_list, player_list, action_lists_list


def game_to_table(game: SGame) -> pd.DataFrame:
    """Convert SGame to a DataFrame in the tabular format."""
    state_labels = game.state_labels
    player_labels = game.player_labels
    action_labels = game.action_labels
    # if action-labels is given as single list for all agents, create a full nested version:
    if isinstance(action_labels[0], (str, int, float)):
        action_labels = [[action_labels[:game.nums_actions[s, i]]
                          for i in range(game.num_players)] for s in range(game.num_states)]

    # table header:
    a_cols = [f'a_{p}' for p in player_labels]
    u_cols = [f'u_{p}' for p in player_labels]
    phi_cols = [f'phi_{s}' for s in state_labels]
    df = pd.DataFrame(columns=['state'] + a_cols + u_cols + phi_cols)
    # delta-row:
    df.loc[0] = ['delta'] + len(a_cols) * [""] + game.discount_factors.tolist() + len(phi_cols) * [np.nan]

    for s in range(game.num_states):
        for index, action_profile in zip(np.ndindex(*game.nums_actions[s]), itertools.product(*action_labels[s])):
            u = game.payoffs[(s, slice(None)) + index].tolist()
            phi = game.phi[(s,) + index + (slice(None),)].tolist()
            df.loc[len(df)] = [state_labels[s]] + list(action_profile) + u + phi

    return df
