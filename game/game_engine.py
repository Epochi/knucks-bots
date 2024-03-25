"""
Game Engine Module for playing the game.
"""

import random
from game.game_board import GameBoard


class GameEngine:
    """Class for running the game and interface for interacting with the game board."""

    def __init__(self, enable_print=False):
        self.game_board = GameBoard()
        # randomly select the first player
        self.current_player = random.randint(1, 2)
        self.game_over = False
        self.winner = None
        self.enable_print = enable_print
        self.dice_value = None

        self.print(f"Player {self.current_player} goes first.")

    def print(self, *args, **kwargs):
        """Print function that can be toggled on/off."""
        if self.enable_print:
            print(*args, **kwargs)

    def roll_dice(self):
        """
        Roll a dice for the current player.

        :return: The value of the rolled dice (1-6).
        """
        return self.game_board.roll_dice()

    def switch_player(self):
        """Switch the current player."""
        self.current_player = 2 if self.current_player == 1 else 1

    def check_game_over(self):
        """Check if the game is over and set the winner if it is."""
        if self.game_board.check_full():
            self.game_over = True
            scores = self.game_board.calculate_score()
            if scores[0] > scores[1]:
                self.winner = 1
            elif scores[1] > scores[0]:
                self.winner = 2
            else:
                self.winner = 0  # Draw

    def start_turn(self):
        """
        Start a new turn for the current player.

        :return: The value of the rolled dice.
        """
        if self.game_over:
            raise ValueError("The game is already over.")

        self.dice_value = self.roll_dice()
        self.print(f"Player {self.current_player} rolled a {self.dice_value}.")

    def do_move(self, row, col):
        """
        Place the dice in the specified column for the current player.

        :param row: The row where the dice should be placed.
        :param col: The column where the dice should be placed.
        :return: True if the move was successfully made, otherwise False.
        """
        if self.game_over:
            raise ValueError("The game is already over.")

        if self.dice_value is None:
            raise ValueError("A dice has not been rolled for the current turn.")

        try:
            self.game_board.place_dice(row, col, self.dice_value)
            self.print(
                f"Player {self.current_player} placed a {self.dice_value} at ({row}, {col})."
            )
            return True
        except ValueError as e:
            self.print(e)
            return False

    def end_turn(self):
        """End the current player's turn."""
        if self.game_over:
            raise ValueError("The game is already over.")

        self.check_game_over()

        self.dice_value = None
        self.switch_player()
