"""
Game Board Module
"""

import random
import copy


class GameBoard:
    """Class for representing state of the game board."""

    def __init__(self):
        # Initialize a 6x3 board, representing two 3x3 grids for two players.
        # The board is initialized with undefined's indicating empty spaces.
        self.board = [[0 for _ in range(3)] for _ in range(6)]

    def roll_dice(self):
        """
        Roll a dice.

        :return: The value of the dice (1-6).
        """
        return random.randint(1, 6)

    def place_dice(self, row, col, value):
        """
        Place a dice on the board.

        :param row: The row to place the dice (0-5)
        :param col: The column to place the dice (0-2)
        :param value: The value of the dice (1-6)
        :return: None
        """
        if self.is_valid_move(row, col):
            self.board[row][col] = value
            self.remove_opponents_dice(row, col, value)
        else:
            raise ValueError("Invalid move. Spot is already occupied or out of range.")

    def is_valid_move(self, row, col):
        """
        Check if a move is valid (i.e., the selected spot is empty and within the board).

        :param row: The row for the intended move.
        :param col: The column for the intended move.
        :return: True if the move is valid, False otherwise.
        """
        return 0 <= row < 6 and 0 <= col < 3 and self.board[row][col] == 0

    def remove_opponents_dice(self, row, col, value):
        """
        Remove the opponent's dice with the same value in the same column from the board
        based on just placed dice.

        :param row: The row where the dice was placed.
        :param col: The column where the dice was placed.
        :param value: The value of the dice placed.
        :return: None
        """

        player = 1 if row < 3 else 2

        for col in range(3):
            player_one_has_value = any(
                self.board[row][col] == value for row in range(3)
            )
            player_two_has_value = any(
                self.board[row][col] == value for row in range(3, 6)
            )
            if player_one_has_value and player_two_has_value:
                if player == 1:
                    for row in range(3, 6):  # Remove dice from player 2's area
                        if self.board[row][col] == value:
                            self.board[row][col] = 0
                elif player == 2:
                    for row in range(3):  # Remove dice from player 1's area
                        if self.board[row][col] == value:
                            self.board[row][col] = 0

    def check_full(self):
        """
        Check if either side of the board is full, indicating the end of the game.

        :return: True if either side of the board is full, False otherwise.
        """
        return all(
            self.board[row][col] != 0 for row in range(3) for col in range(3)
        ) or all(self.board[row][col] != 0 for row in range(3, 6) for col in range(3))

    def calculate_score(self):
        """
        Calculate the score for each player based on the current board state.

        :return: A tuple containing the scores of player 1 and player 2 respectively.
        """

        def score_for_player(start_row):
            score = 0
            for col in range(3):
                column_values = [
                    self.board[row][col]
                    for row in range(start_row, start_row + 3)
                    if self.board[row][col] != 0
                ]
                unique_values = set(column_values)
                for value in unique_values:
                    count = column_values.count(value)
                    score += value**count
            return score

        player_one_score = score_for_player(0)
        player_two_score = score_for_player(3)
        return player_one_score, player_two_score

    def get_available_moves(self):
        """
        Get all available moves on the board.


        :return: A tuple containing the available list of moves for player 1 and player 2 respectively.
        """
        player_one_available_moves = [
            (row, col)
            for row in range(3)
            for col in range(3)
            if self.board[row][col] == 0
        ]
        player_two_available_moves = [
            (row, col)
            for row in range(3, 6)
            for col in range(3)
            if self.board[row][col] == 0
        ]
        return player_one_available_moves, player_two_available_moves

    def get_available_moves_from_1p_pov(self, player):
        """
        Get all available moves on the board as if the player is player 1.

        :param player: The player's perspective to return the available moves.
        :return: The available moves on the board from the perspective of the player.
        """
        if player == 1:
            return self.get_available_moves()[0]
        elif player == 2:
            # Return the available moves for player 2 as if they were player 1
            # so if row is 3 it becomes 0, if row is 4 it becomes 1, if row is 5 it becomes 2
            return [(row - 3, col) for row, col in self.get_available_moves()[1]]

    def get_board_from_1p_pov(self, player):
        """
        Get the current state of the board from the perspective of a player.

        :param player: The player's perspective to return the board state.
        :return: The current state of the board from the perspective of the player.
        """
        if player == 1:
            return copy.deepcopy(self.board)
        elif player == 2:
            return [row[::-1] for row in copy.deepcopy(self.board)[::-1]]

    def display(self):
        """
        Display the current state of the board.
        """
        lines = []
        print("\n", end="")  # Print a newline character before the game board
        for i, row in enumerate(self.board):
            line = (
                "| "
                + " | ".join(str(value) if value is not 0 else " " for value in row)
                + " |"
            )
            print(line)
            lines.append(line)
            if i == 2:  # Only print the separating line after the third row
                separator = "|---|---|---|"
                print(separator)
                lines.append(separator)
        return lines
