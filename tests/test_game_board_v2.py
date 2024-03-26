"""
GameBoard class test module.
"""

import unittest
import numpy as np
from game.game_board_v2 import GameBoard



class TestGameBoardV2(unittest.TestCase):
    """Green Path tests for the GameBoard class."""

    def setUp(self):
        """Initialize a new game board for each test."""
        self.game_board = GameBoard()

    def test_init(self):
        """Test the initialization of the game board."""
        self.assertEqual(len(self.game_board.board), 2, "the board should have 2 members")
        self.assertEqual(
            len(self.game_board.board[0]), 3, "The board should have 3 columns."
        )
        self.assertEqual(
            len(self.game_board.board[0][0]), 3, "The board should have 3 rows."

        )
    def test_roll_dice(self):
        """Test rolling a dice."""
        dice_value = self.game_board.roll_dice()
        self.assertTrue(1 <= dice_value <= 6, "The dice value is out of range.")

    def test_is_valid_move_green_path(self):
        """Test checking if a move is valid."""
        self.assertTrue(self.game_board.is_valid_move(0, 0), "The move should be valid.")
        self.assertTrue(self.game_board.is_valid_move(1, 2), "The move should be valid.")
    
    def test_is_valid_move_red_path(self):
        """Test checking if a move is valid."""
        self.game_board.board = [[[1, 2, 3], [4, 5, 6], [0, 0, 0]], [[1, 2, 3], [4, 5, 6], [0, 0, 0]]]
        self.assertFalse(self.game_board.is_valid_move(0, 0), "The move should be invalid because column is full")
        self.assertFalse(self.game_board.is_valid_move(2, 2), "The move should be invalid because player is out of range")
        self.assertFalse(self.game_board.is_valid_move(1, 3), "The move should be invalid because column is out of range")

    def test_place_dice_green_path(self):
        """Test placing a dice on the board."""
        self.game_board.place_dice(0, 0, 3)
        self.assertEqual(
            self.game_board.board[0][0][0], 3, "The dice was not placed correctly on the board."
        )
        self.game_board.place_dice(0, 0, 1)
        self.assertEqual(
            np.array_equal(self.game_board.board[0][0], [3,1,0]), True, "The dice was not sorted."
        )
        self.game_board.place_dice(1, 1, 6)
        self.assertEqual(
            self.game_board.board[1][1][0], 6, "The dice was not placed correctly on the board."
        )
        
    def test_place_dice_red_path(self):
        """Test placing a dice on the board."""
        self.game_board.board = np.array([[[1, 2, 3], [4, 5, 6], [0, 0, 0]], [[1, 2, 3], [4, 5, 6], [0, 0, 0]]])
        with self.assertRaises(ValueError):
            self.game_board.place_dice(0, 0, 3)
            self.game_board.place_dice(0, 0, 3)
    
    def test_remove_opponents_dice(self):
        """Test removing the opponent's dice."""
        self.game_board.board = np.array([[[1, 2, 3], [4, 5, 6], [0, 0, 0]], [[1, 2, 3], [4, 5, 6], [0, 0, 0]]])
        self.game_board.remove_opponents_dice(0, 0, 1)
        self.assertEqual(
            np.array_equal(self.game_board.board[1][0], [0, 2, 3]), True, "The opponent's dice was not removed correctly."
        )
        self.game_board.remove_opponents_dice(1, 0, 6)
        self.assertEqual(
            np.array_equal(self.game_board.board[0][0], [1, 2, 3]), True, "The opponent's dice was not removed correctly."
        )

    def test_check_if_null(self):
        """Test if the board is full."""
        self.game_board.board = np.array([[[1, 2, 3], [4, 5, 6], [0, 0, 0]], [[1, 2, 3], [4, 5, 6], [0, 0, 0]]])
        self.assertFalse(self.game_board.check_full(), "The board should not be full.")
        self.game_board.board = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0, 0, 0], [4, 5, 6], [7, 8, 9]]])
        self.assertTrue(self.game_board.check_full(), "The board should be full.")

    def test_calculate_score(self):
        """Test calculating the score."""
        self.game_board.board = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        self.assertEqual(self.game_board.calculate_score(), (18, 0), "Should not multiple unique values in col")
        
        self.game_board.board = np.array([[[2, 2, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        self.assertEqual(self.game_board.calculate_score(), (4, 0), "Should multiple values by themselves if 2x")

        self.game_board.board = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [6, 6, 6]]])
        self.assertEqual(self.game_board.calculate_score(), (0, 216), "Should multiple values by themselves if 3x")

    def test_get_available_moves(self):
        """Test getting available moves."""
        self.game_board.board = np.array([[[1, 2, 3], [4, 5, 6], [0, 0, 0]], [[1, 2, 3], [4, 5, 6], [1, 1, 1]]])
        self.assertEqual(self.game_board.get_available_moves(), [[2], []], "The available moves should be 2 for both players.")

        self.game_board.board = np.array([[[1, 2, 3], [4, 5, 6], [0, 0, 0]], [[1, 2, 0], [4, 5, 0], [0, 0, 0]]])
        self.assertEqual(self.game_board.get_available_moves(), [[2], [0, 1, 2]], "The available moves should be 2 for player 1 and 0, 1, 2 for player 2.")

    def test_display(self):
        """Test displaying the board."""
        self.game_board.board = np.array([[[1, 0, 3], [0, 2, 1], [2, 0, 0]], [[0, 5, 6], [0, 5, 6], [4, 5, 6]]])
        expected_lines = [
            "| 1 |   | 2 |",
            "|   | 2 |   |",
            "| 3 | 1 |   |",
            "|---|---|---|",
            "|   |   | 4 |",
            "| 5 | 5 | 5 |",
            "| 6 | 6 | 6 |",
        ]
        result_lines = self.game_board.display()
        self.assertEqual(result_lines, expected_lines, "The display is incorrect.")

if __name__ == "__main__":
    unittest.main()
