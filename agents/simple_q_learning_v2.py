"""Simple Q-Learning Agent Module"""

import random
from agents.base_agent_v2 import AbstractAgent
import game.player_actions_v2 as pa


class QLearningAgent(AbstractAgent):
    """
    An agent that learns to play the game using Q-Learning.
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
        self.nickname = "Cell, the Brain Cell"
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

        # Decide action: explore or exploit
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(available_moves)
        if state not in self.q_table:
            action = random.choice(available_moves)
        else:
            q_values = self.q_table[state]
            # filter out actions from the Q-table that are not in the available moves
            available_moves_q_values = {
                k: v for k, v in q_values.items() if k in available_moves
            }
            # select the available action with the highest Q-value
            action = max(available_moves_q_values, key=available_moves_q_values.get)

        return action

    def update_q_table(
        self, prev_state: str, action: tuple, reward: int, new_state: str
    ):
        """update the Q-table based on the reward received"""

        # Ensure the state entries exist in the Q-table
        if prev_state not in self.q_table:
            self.q_table[prev_state] = {action: 0}

        if action not in self.q_table[prev_state]:
            self.q_table[prev_state][action] = 0

        # update Q-Value for the taken action in the previous state
        current_q_value = self.q_table[prev_state][action]

        # to get max future reward we need to ignore the dice value from the state
        possible_new_states_prefix = new_state[:-1]

        max_future_reward = 0

        # # Iterate through all states in the Q-table
        # for state, actions in self.q_table.items():
        #     # Check if the state matches the base_state, ignoring the dice roll
        #     if state.startswith(possible_new_states_prefix):
        #         # Find the maximum Q-value among actions in this matching state
        #         max_reward_for_state = max(actions.values())
        #         # Update the maximum future reward if this state has a higher value
        #         max_future_reward = max(max_future_reward, max_reward_for_state)

        # performance optimization, went from 60s to 1-2s
        # get all possible future states by adding dice values from 1 to 6 to possible new states prefix
        possible_new_states = [possible_new_states_prefix + str(i) for i in range(1, 7)]
        for state in possible_new_states:
            if state in self.q_table:
                max_reward_for_state = max(self.q_table[state].values())
                max_future_reward = max(max_future_reward, max_reward_for_state)

        new_q_value = current_q_value + self.learning_rate * (
            reward + self.discount_factor * max_future_reward - current_q_value
        )

        self.q_table[prev_state][action] = new_q_value

        # Update exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )
