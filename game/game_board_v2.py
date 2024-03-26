"""
Game Board Module
"""

import random
import copy
import numpy as np


class GameBoard:
    """
    Class for representing state of the game board.
    Restructured to work by columns instead to help with agent training performance
    Also accept columns as input for placing dice instead of row and column
    """

    def __init__(self):
        # Initialize 2 3x3 grids for two players.s

        # we will treat it as a 3D array with 2 layers, each layer representing a player's board
        # although the rules says that we match the dice value in the same column
        # we will treat second dimension as columns and third dimension as rows
        # and change represantation in visual layer when we draw the board
        self.board = np.zeros((2, 3, 3))

    def roll_dice(self):
        """
        Roll a dice.

        :return: The value of the dice (1-6).
        """
        return random.randint(1, 6)

    def is_valid_move(self, player, col):
        """
        Check if a move is valid (i.e., the selected spot is empty and within the board).

        :param player: The row for the intended move.
        :param col: The column for the intended move.
        :return: True if the move is valid, False otherwise.
        """
        return (
            0 <= player < 2
            and 0 <= col < 3
            and (
                self.board[player][col][0] == 0
                or self.board[player][col][1] == 0
                or self.board[player][col][2] == 0
            )
        )

    def place_dice(self, player, col, value):
        """
        Place a dice on the board.

        :param player: The row to place the dice (0-1)
        :param col: The column to place the dice (0-2)
        :param value: The value of the dice (1-6)
        :return: None
        """
        if self.is_valid_move(player, col):
            row = np.argmax(self.board[player][col] == 0)
            self.board[player][col][row] = value
            self.remove_opponents_dice(player, col, value)
            np.sort(self.board[player][col], axis=0)
        else:
            raise ValueError("Invalid move. Spot is already occupied or out of range.")

    def remove_opponents_dice(self, player, col, value):
        """
        Remove the opponent's dice with the same value in the same column from the board
        based on just placed dice.

        :param player: The player that wants its opponent destroyed.
        :param col: The column where the dice was placed.
        :param value: The value of the dice placed.
        :return: None
        """

        opponent = 1 if player == 0 else 0

        # check if the opponent has the same value in the same column
        if np.any(self.board[opponent][col] == value):
            self.board[opponent][col] = np.where(self.board[opponent][col] == value, 0, self.board[opponent][col])

    def check_full(self):
        """
        Check if either side of the board is full, indicating the end of the game.

        :return: True if either side of the board is full, False otherwise.
        """
        return np.all(self.board[0] != 0) or np.all(self.board[1] != 0)

    def calculate_score(self):
        """
        Calculate the score for each player based on the current board state.

        :return: A tuple containing the scores of player 1 and player 2 respectively.
        """
        scores = np.zeros(2, dtype=int)
        # Iterate over each player
        for player in range(2):
            # Iterate over each column
            for col in range(3):
                # Extract the column for the player, this time including zeros
                column_values = self.board[player, col, :]
                # Count the occurrence of each value in the column
                unique, counts = np.unique(column_values, return_counts=True)
                # Calculate the score for the column, skipping zeros in the scoring logic
                column_score = sum([value**count if count > 1 and value != 0 else value for value, count in zip(unique, counts) if value != 0])
                # Add the column score to the player's total score
                scores[player] += column_score

        return scores

    def get_available_moves(self):
        """
        Get all available moves on the board.

        :return: A tuple containing the available array of moves for player 1 and player 2 respectively.
        """
        moves = [[], []]
        for player in range(2):
            for col in range(3):
                if np.any(self.board[player][col] == 0):
                    moves[player].append(col)

        return moves

    def display(self):
        """
        Display the current state of the board for both players.
        """
        print("\n")  # Print a newline character before the game board
        lines = []

        # Iterate over each player's board
        for player in range(2):
            for col in range(3):
                # Transpose the column to row for display, and replace 0 with space
                line = "| " + " | ".join(str(self.board[player, row, col]) if self.board[player, row, col] != 0 else " " for row in range(3)) + " |"
                print(line)
                lines.append(line)
            # Print the separating line after each player's board except the last one
            separator = "|---|---|---|" if player == 0 else ""
            if separator:
                print(separator)
                lines.append(separator)
        
        return lines