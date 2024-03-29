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
        should_save_model=True,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
    ):
        super().__init__(nickname, should_save_model)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.type = "QL"

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
            available_moves_q_values = [q_values[i] for i in available_moves]
            if all(q == 0 for q in available_moves_q_values):
                # if all the available actions have a Q-value of 0, then select a random action
                action = random.choice(available_moves)
            # select the available action with the highest Q-value
            action = available_moves[
                available_moves_q_values.index(max(available_moves_q_values))
            ]

        return action

    def learn(
        self,
        prev_state: str,
        action: tuple,
        reward: int,
        new_states: list,
        game_over: bool,
        winner=None,
    ):
        """update the Q-table based on the reward received"""
        # Ensure the state entries exist in the Q-table

        if prev_state not in self.model:
            self.model[prev_state] = [0.0, 0.0, 0.0]

        # update Q-Value for the taken action in the previous state
        current_q_value = self.model[prev_state][action]

        max_future_reward = 0
        for state in new_states:
            if state in self.model:
                max_reward_for_state = max(self.model[state])
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
