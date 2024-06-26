"""
Game Engine Module for playing the game.
"""

import random
from game.game_board_v2 import GameBoard


class GameEngine:
    """Class for running the game and interface for interacting with the game board."""

    def __init__(
        self,
        enable_print=False,
        max_dice_value=None,
        should_remove_opponents_dice=None,
        safe_mode=None,
    ):
        self.game_board = GameBoard(
            max_dice_value, should_remove_opponents_dice, safe_mode
        )
        # randomly select the first player
        self.current_player = random.randint(0, 1)
        self.game_over = False
        self.winner = None
        self.enable_print = enable_print
        self.dice_value = None

        self.print(f"Player {self.current_player} goes first.")

    def print(self, *args, **kwargs):
        """Print function that can be toggled on/off."""
        if self.enable_print:
            print(*args, **kwargs)

    def switch_player(self):
        """Switch the current player."""
        self.current_player = 1 if self.current_player == 0 else 0

    def check_game_over(self):
        """Check if the game is over and set the winner if it is."""
        if self.game_board.check_full():
            self.game_over = True
            scores = self.game_board.player_1_score, self.game_board.player_2_score
            if scores[0] > scores[1]:
                self.winner = 0
            elif scores[1] > scores[0]:
                self.winner = 1
            else:
                self.winner = -1  # Draw

    def start_turn(self):
        """
        Start a new turn for the current player.

        :return: The value of the rolled dice.
        """
        if self.game_over:
            raise ValueError("The game is already over.")

        self.dice_value = self.game_board.roll_dice()
        self.print(f"Player {self.current_player} rolled a {self.dice_value}.")

    def do_move(self, col):
        """
        Place the dice in the specified column for the current player.

        :param row: The row where the dice should be placed.
        :param col: The column where the dice should be placed.
        :return: True if the move was successfully made, otherwise False.
        """
        # if self.game_over:
        #     raise ValueError("The game is already over.")

        # if self.dice_value is None:
        #     raise ValueError("A dice has not been rolled for the current turn.")

        try:
            self.game_board.place_dice(self.current_player, col, self.dice_value)
            self.print(
                f"Player {self.current_player} placed a {self.dice_value} at column {col}."
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
        if not self.game_over:
            self.switch_player()
