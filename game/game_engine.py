"""
Game Engine Module for playing the game.
"""

import random
from abc import ABC, abstractmethod
from game.game_board import GameBoard


class GameEngine:
    """Class for running the game engine."""

    def __init__(self, enable_print=False):
        self.game_board = GameBoard()
        # randomly select the first player
        self.current_player = random.randint(1, 2)
        self.game_over = False
        self.winner = None
        self.enable_print = enable_print

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

    def make_move(self, player, dice_value, row, col):
        """
        Make a move on the board based on the player's decision.

        :param player: The current player (1 or 2).
        :param dice_value: The value of the rolled dice.
        :param row: The row where the dice should be placed.
        :param col: The column where the dice should be placed.
        :return: True if the move was successfully made, otherwise False.
        """
        if player != self.current_player:
            self.print(f"It's not player {player}'s turn.")
            return False

        try:
            place_dice = self.game_board.get_place_dice_from_1p_pov(player)
            place_dice(row, col, dice_value)
            self.check_game_over()
            self.switch_player()
            return True
        except ValueError as e:
            self.print(e)
            return False

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

    def play_game(self, player_one_strategy, player_two_strategy):
        """
        Main game loop for playing the game. Accepts two player strategies that control
        how players make their moves based on the rolled dice.

        :param player_one_strategy: Function representing player one's strategy.
        :param player_two_strategy: Function representing player two's strategy.
        """
        while not self.game_over:
            if self.enable_print:
                self.game_board.display()
            dice_value = self.roll_dice()
            self.print(f"Player {self.current_player} rolled a {dice_value}.")

            if self.current_player == 1:
                row, col = player_one_strategy(self.game_board, dice_value)
            else:
                row, col = player_two_strategy(self.game_board, dice_value)

            # check player available moves
            self.print(
                f"Player {self.current_player} available moves: {self.game_board.get_available_moves()[self.current_player - 1]}"
            )
            self.print(f"Player {self.current_player} selected move: ({row}, {col}).")

            if self.make_move(self.current_player, dice_value, row, col):
                self.print(
                    f"Player {self.current_player} placed a {dice_value} at ({row}, {col})."
                )
            else:
                self.print(
                    f"Invalid Move attempted. Player {self.current_player} attempted to place {dice_value} at ({row}, {col}). Please try again."
                )

            if self.game_over:
                if self.enable_print:
                    self.game_board.display()  # Show final board state
                if self.winner == 0:
                    self.print("The game is a draw!")
                else:
                    self.print(f"Player {self.winner} wins!")


class AbstractAgent(ABC):
    """
    Abstract base class for all agents in the game.

    Agents must implement the select_move method, which decides on a move
    based on the current game board state and the value of the rolled dice.
    """

    def __init__(self, player_number):
        """
        Initializes the agent with its player number.

        To simplify logic we just set player number on agent initialization
          and instead we choose random player to start the game.

        :param player_number: The player number (1 or 2).
        """
        self.player_number = player_number

    @abstractmethod
    def select_move(self, game_board, dice_value):
        """
        Determines the move to make based on the game state and dice value.

        :param game_board: The current state of the game board, providing access to game data.
        :param dice_value: The result of the dice roll for this turn.
        :return: A tuple (row, col) indicating the chosen move.
        """

    def get_player_available_moves(self, game_board):
        """
        Get the available moves for the agent's player number.

        :param game_board: The current state of the game board.
        :return: A list of tuples representing the available moves as if the player was player 1.
        """
        return game_board.get_available_moves_from_1p_pov(self.player_number)

    def get_player_board(self, game_board):
        """
        Get the game board from the perspective of the agent's player number.

        :param game_board: The current state of the game board.
        :return: The game board as if the player was player 1.
        """
        return game_board.get_board_from_1p_pov(self.player_number)

    def strategy(self, game_board, dice_value):
        """
        Alias for the select_move method, to be used in the GameEngine class.

        :param game_board: The current state of the game board, providing access to game data.
        :param dice_value: The result of the dice roll for this turn.
        :return: A tuple (row, col) indicating the chosen move.
        """
        return self.select_move(game_board, dice_value)


# def q_learning_strategy(game_board, dice_value):
#     """
#     Example strategy function for a Q-Learning agent. This should be replaced with the actual
#     decision-making logic of the agent, which decides based on the current board state and the
#     value of the rolled dice.

#     :param game_board: The current game board.
#     :param dice_value: The value of the rolled dice.
#     :return: The chosen row and column as a tuple.
#     """
#     # Placeholder logic for selecting a move; replace with your Q-learning agent's decision-making process
#     available_moves = game_board.get_available_moves()[
#         0 if game_board.current_player == 1 else 1
#     ]
#     if available_moves:
#         return available_moves[0]  # Example: choose the first available move
#     else:
#         return None, None  # No available moves


# # Example usage
# if __name__ == "__main__":
#     engine = GameEngine()
#     engine.play_game(q_learning_strategy, q_learning_strategy)
