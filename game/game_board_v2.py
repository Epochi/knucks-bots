"""
Game Board Module
"""

import random


class GameBoard:
    """
    Class for representing state of the game board.
    Restructured to work by columns instead to help with agent training performance
    Also accept columns as input for placing dice instead of row and column
    """

    def __init__(self, max_dice_value, should_remove_opponents_dice, safe_mode):
        # Initialize 2 3x3 grids for two players
        self.player_1_board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.player_2_board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # since we check this often, we can keep track of how many dice are placed for cheaper check
        self.player_1_board_placed_dice = 0
        self.player_2_board_placed_dice = 0

        self.player_1_score = 0
        self.player_2_score = 0

        if max_dice_value is None:
            max_dice_value = 6
        self.max_dice_value = max_dice_value

        if should_remove_opponents_dice is None:
            should_remove_opponents_dice = True
        self.should_remove_opponents_dice = should_remove_opponents_dice

        if safe_mode is None:
            safe_mode = True
        self.safe_mode = safe_mode

    def roll_dice(self):
        """
        Roll a dice.

        :return: The value of the dice (1-6).
        """
        return random.randint(1, self.max_dice_value)

    def is_valid_move(self, player, col):
        """
        Check if a move is valid (i.e., the selected spot is empty and within the board).

        :param player: The row for the intended move.
        :param col: The column for the intended move.
        :return: True if the move is valid, False otherwise.
        """
        player_board = self.player_1_board if player == 0 else self.player_2_board
        return (
            0 <= player < 2
            and 0 <= col < 3
            and (
                player_board[col][0] == 0
                or player_board[col][1] == 0
                or player_board[col][2] == 0
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
        if self.safe_mode:
            if not self.is_valid_move(player, col):
                raise ValueError(
                    "Invalid move. Spot is already occupied or out of range."
                )

        board = self.player_1_board if player == 0 else self.player_2_board

        if board[col][0] == 0:
            board[col][0] = value
        elif board[col][1] == 0:
            board[col][1] = value
        elif board[col][2] == 0:
            board[col][2] = value
        else:
            raise ValueError("Invalid move. Spot is already occupied or out of range.")

        if player == 0:
            self.player_1_board_placed_dice += 1
        else:
            self.player_2_board_placed_dice += 1

        board[col].sort()

        if self.should_remove_opponents_dice:
            self.remove_opponents_dice(player, col, value)

        self.player_1_score, self.player_2_score = self.calculate_score()

    def remove_opponents_dice(self, player, col, value):
        """
        Remove the opponent's dice with the same value in the same column from the board
        based on just placed dice.

        :param player: The player that wants its opponent destroyed.
        :param col: The column where the dice was placed.
        :param value: The value of the dice placed.
        :return: None
        """

        opponent_board = self.player_2_board if player == 0 else self.player_1_board
        opponent_placed_dice = 0

        if opponent_board[col][0] == value:
            opponent_board[col][0] = 0
            opponent_placed_dice -= 1
        if opponent_board[col][1] == value:
            opponent_board[col][1] = 0
            opponent_placed_dice -= 1
        if opponent_board[col][2] == value:
            opponent_board[col][2] = 0
            opponent_placed_dice -= 1

        if player == 0:
            self.player_1_board_placed_dice += opponent_placed_dice
        else:
            self.player_2_board_placed_dice += opponent_placed_dice

        opponent_board[col].sort()

    def check_full(self):
        """
        Check if either side of the board is full, indicating the end of the game.

        :return: True if either side of the board is full, False otherwise.
        """
        if self.player_1_board_placed_dice == 9 or self.player_2_board_placed_dice == 9:
            return True

    def calculate_score(self, player=None):
        """
        Calculate the score for each player based on the current board state.

        :return: A tuple containing the scores of player 1 and player 2 respectively.
        """

        def score_in_column(col):
            # if all the same
            if col[0] == col[1] == col[2] and col[0] != 0:
                return col[0] ** 3

            # if two are the same
            if col[0] == col[1] and col[0] != 0:
                return col[0] ** 2 + col[2]

            if col[1] == col[2] and col[1] != 0:
                return col[1] ** 2 + col[0]

            if col[0] == col[2] and col[0] != 0:
                return col[0] ** 2 + col[1]

            # if all unique
            return col[0] + col[1] + col[2]

        if self.should_remove_opponents_dice or player is None:
            self.player_1_score = sum(
                score_in_column(col) for col in self.player_1_board
            )
            self.player_2_score = sum(
                score_in_column(col) for col in self.player_2_board
            )
        elif player is not None:
            player_board = self.player_1_board if player == 0 else self.player_2_board
            if player == 0:
                self.player_1_score = sum(score_in_column(col) for col in player_board)
            else:
                self.player_2_score = sum(score_in_column(col) for col in player_board)

        return self.player_1_score, self.player_2_score

    def get_available_moves(self, player):
        """
        Get all available moves on the board.

        :return: Available moves for the player.
        """

        def can_move_to_cols(board):
            moves = []
            for col in range(3):
                for row in range(3):
                    if board[col][row] == 0:
                        moves.append(col)
                        break
            return moves

        if player == 0:
            return can_move_to_cols(self.player_1_board)
        return can_move_to_cols(self.player_2_board)

    def display(self):
        """
        Display the current state of the board for both players.
        """
        print("\n")  # Print a newline character before the game board
        lines = []

        # Iterate over each player's board
        for player in range(2):
            if player == 0:
                board = self.player_1_board
            else:
                board = self.player_2_board
            for col in range(3):
                # Transpose the column to row for display, and replace 0 with space
                line = (
                    "| "
                    + " | ".join(
                        (str(board[row][col]) if board[row][col] != 0 else " ")
                        for row in range(3)
                    )
                    + " |"
                )
                print(line)
                lines.append(line)
            # Print the separating line after each player's board except the last one
            separator = "|---|---|---|" if player == 0 else ""
            if separator:
                print(separator)
                lines.append(separator)

        return lines
