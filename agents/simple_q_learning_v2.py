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
        nickname="Cell, the Brain Cell",
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
    ):
        super().__init__(nickname)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

    def select_move(self, game_engine):
        # Get the current state
        board_state = pa.get_board_state(game_engine)
        dice_value = pa.get_dice_value(game_engine)
        state = self.convert_state(board_state, dice_value)

        available_moves = pa.get_available_moves(game_engine)

        # Decide action: explore or exploit
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(available_moves)
        if state not in self.model:
            action = random.choice(available_moves)
        else:
            q_values = self.model[state]
            # filter out actions from the Q-table that are not in the available moves
            available_moves_q_values = {
                k: v for k, v in q_values.items() if k in available_moves
            }
            # select the available action with the highest Q-value
            action = max(available_moves_q_values, key=available_moves_q_values.get)

        return action

    def learn(self, prev_state: str, action: tuple, reward: int, new_states: list):
        """update the Q-table based on the reward received"""
        # Ensure the state entries exist in the Q-table
        if prev_state not in self.model:
            self.model[prev_state] = {action: 0}

        if action not in self.model[prev_state]:
            self.model[prev_state][action] = 0

        # update Q-Value for the taken action in the previous state
        current_q_value = self.model[prev_state][action]

        max_future_reward = 0
        for state in new_states:
            if state in self.model:
                max_reward_for_state = max(self.model[state].values())
                max_future_reward = max(max_future_reward, max_reward_for_state)

        new_q_value = current_q_value + self.learning_rate * (
            reward + self.discount_factor * max_future_reward - current_q_value
        )

        self.model[prev_state][action] = new_q_value

        # Update exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )

    def convert_state(self, board_state, dice_value):
        """
        Converts the current board state and dice value into a string for Q-Table.
        """
        # flatten two arrays and concatinate into string
        # when calculating own score only
        # the max possible state size is = 592704 * 6 = 3,556,224
        #   with 3 lists with 3 elements each,
        #   with element value possible from 0 to 6,
        #   and dice roll value from 1 to 6

        state = "".join(
            str(col) for sublist in board_state for row in sublist for col in row
        ) + str(dice_value)
        return int(state)

    def load_model(self, path):
        super().load_model(path)

        num_states = len(self.model)
        num_actions = sum(len(v) for v in self.model.values())
        total_size = num_states * num_actions

        self.exploration_rate = max(
            self.min_exploration_rate,
            (self.exploration_rate * self.exploration_decay) ** total_size,
        )
