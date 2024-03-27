"""Module for the Abstract Agent Class"""

from abc import ABC, abstractmethod
import pickle
import numpy as np
import os


class AbstractAgent(ABC):
    """
    Abstract base class for all agents in the game.

    Agents must implement the select_move method, which decides on a move
    based on the current game board state and the value of the rolled dice.
    """

    def __init__(self, q_table_path=None):
        """in case we want to add some common attributes to all agents in the future."""
        self.nickname = "The Mysterion"
        self.q_table = {}
        if q_table_path:
            self.load_q_table(q_table_path)
            print("Q-Table loaded successfully.")
            print(f"Q-Table size: {len(self.q_table)}")

    @abstractmethod
    def select_move(self, game_engine):
        """
        Determines the move to make based on the game state and dice value.

        :param game_board_state: The current state of the game board.
        :param my_score: The current score of the agent's player.
        :param opponent_score: The current score of the opponent player.
        :param dice_value: The value of the rolled dice.
        :param available_moves: A list of tuples representing the available moves.
        :return: row, col move to make
        """

    def update_q_table(self, prev_state, action, reward, new_state):
        """
        Update the Q-Table based on the previous state, action, reward, and new state.
        """

    def convert_state(self, board_state, dice_value):
        """
        Converts the current board state and dice value into a string for Q-Table.
        """
        # flatten two arrays and concatinate into string
        state = (
            "".join(str(col) for row in board_state[0] for col in row)
            # + "".join(str(col) for row in board_state[1] for col in row)
            + str(dice_value)
        )
        return state

    def load_q_table(self, path):
        """Load the Q-Table from a file."""
        # Check if file exists
        if not os.path.exists(path):
            print(
                f"Warning: File {path} does not exist. Starting with an empty Q-table."
            )
            return

        with open(path, "rb") as file:
            self.q_table = pickle.load(file)
