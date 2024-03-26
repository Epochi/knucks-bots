"""Simple Q-Learning Agent with optimized state Module"""

import random
from agents.base_agent import AbstractAgent
import game.player_actions as pa


class QLearningAgentSpaceOptimized(AbstractAgent):
    """
    This agent is using simple Q learning to learn how to play the game.
    But it has state optimization, where columns in the state are sorted
        and we only keep the column selection as the action.
    """

    def __init__(
        self,
        q_table_path=None,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
    ):
        super().__init__(q_table_path=q_table_path)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Recalculate the exploration rate based on the Q-table size if a Q-table is provided
        # this is not entirely accurate, since we don't know the numbet of actions, but it's a good enough approximation
        if q_table_path:
            num_states = len(self.q_table)
            num_actions = sum(len(v) for v in self.q_table.values())
            total_size = num_states * num_actions

            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay * total_size,
            )

    def select_move(self, game_engine):
        # Get the current state
        board_state = pa.get_board_state(game_engine)
        dice_value = pa.get_dice_value(game_engine)
        state = self.convert_state(board_state, dice_value)

        available_moves = pa.get_available_moves(game_engine)
        # filter to one action per column
        seen = set()
        available_moves = [
            (x[0], x[1])
            for x in available_moves
            if not (x[1] in seen or seen.add(x[1]))
        ]

        # Decide action: explore or exploit
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(available_moves)
        if state not in self.q_table:
            action = random.choice(available_moves)
        else:
            q_values = self.q_table[state]
            available_column_moves = [x[1] for x in available_moves]
            available_moves_q_values = {
                k: v for k, v in q_values.items() if k in available_column_moves
            }

            # select the available column with the highest Q-value
            column_to_select = max(
                available_moves_q_values, key=available_moves_q_values.get
            )
            # select any row from the available moves for the selected column
            action = random.choice(
                [
                    (row, column_to_select)
                    for row, col in available_moves
                    if col == column_to_select
                ]
            )

        return action

    def update_q_table(
        self, prev_state: str, action: tuple, reward: int, new_state: str
    ):
        """update the Q-table based on the reward received"""

        # Ensure the state entries exist in the Q-table
        if prev_state not in self.q_table:
            self.q_table[prev_state] = {action[1]: 0}

        if action not in self.q_table[prev_state]:
            self.q_table[prev_state][action[1]] = 0

        # update Q-Value for the taken action in the previous state
        current_q_value = self.q_table[prev_state][action[1]]

        # to get max future reward we need to ignore the dice value from the state
        possible_new_states_prefix = new_state[:-1]

        max_future_reward = 0

        # get all possible future states by adding dice values from 1 to 6 to possible new states prefix
        possible_new_states = [possible_new_states_prefix + str(i) for i in range(1, 7)]
        for state in possible_new_states:
            if state in self.q_table:
                max_reward_for_state = max(self.q_table[state].values())
                max_future_reward = max(max_future_reward, max_reward_for_state)

        new_q_value = current_q_value + self.learning_rate * (
            reward + self.discount_factor * max_future_reward - current_q_value
        )

        self.q_table[prev_state][action[1]] = new_q_value

        # Update exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )

    def convert_state(self, board_state, dice_value):
        """
        Sort the columns in the board state for each player separately and convert the state to a string.
        """
        # TODO: this is very slow, we can optimize this further, need to explore
        sort_player_1_board_state = _sort_grid_columns(board_state[:3])
        sort_player_2_board_state = _sort_grid_columns(board_state[3:])

        sorted_board_state = sort_player_1_board_state + sort_player_2_board_state
        return "".join(str(col) for row in sorted_board_state for col in row) + str(
            dice_value
        )


def _sort_grid_columns(grid):
    """
    Sorts the columns of a 2D grid based on the values in each column without moving the rows.
    The sorting is done in descending order, with higher values positioned at the top of the grid.

    :param grid: A list of lists representing the 2D grid.
    :return: A new grid with columns sorted according to their values.
    """
    # Transpose the grid to work with columns as lists
    transposed_grid = list(zip(*grid))
    # Sort each column individually in descending order
    sorted_transposed = [sorted(column, reverse=True) for column in transposed_grid]
    # Transpose back to the original grid structure
    # we don't need to do this, we can just return the sorted_transposed
    # sorted_grid = list(map(list, zip(*sorted_transposed)))
    return sorted_transposed
