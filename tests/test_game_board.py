"""
GameBoard class test module.
"""

import unittest
from game.game_board import GameBoard


class TestGameBoard(unittest.TestCase):
    """Green Path tests for the GameBoard class."""

    def setUp(self):
        """Initialize a new game board for each test."""
        self.board = GameBoard()

    def test_init(self):
        """Test the initialization of the game board."""
        self.assertEqual(len(self.board.board), 6, "The board should have 6 rows.")
        self.assertEqual(
            len(self.board.board[0]), 3, "The board should have 3 columns."
        )
        self.assertEqual(
            all(all(cell == 0 for cell in row) for row in self.board.board),
            True,
            "The board should be initialized with None.",
        )

    def test_roll_dice(self):
        """Test rolling a dice."""
        dice_value = self.board.roll_dice()
        self.assertTrue(1 <= dice_value <= 6, "The dice value is out of range.")

    def test_place_dice(self):
        """Test placing a dice on the board."""
        self.board.place_dice(
            0, 0, 3
        )  # Place a dice with value 3 at the top-left corner
        self.assertEqual(
            self.board.board[0][0], 3, "The dice was not placed correctly on the board."
        )

    def test_place_dice_remove_oppoent_dice_after_placing(self):
        """Test placing a dice on the board and removing the opponent's dice."""
        self.board.place_dice(3, 0, 3)
        self.board.place_dice(0, 0, 3)
        self.assertEqual(
            self.board.board[0][0], 3, "The dice was not placed correctly on the board."
        )
        self.assertEqual(
            self.board.board[3][0], 0, "The opponent's dice was not removed correctly."
        )

    def test_place_dice_remove_all_oppoent_dice_after_placing(self):
        """Test placing a dice on the board and removing the opponent's dice."""
        self.board.place_dice(3, 0, 3)
        self.board.place_dice(4, 0, 3)
        self.board.place_dice(5, 0, 3)
        self.board.place_dice(0, 0, 3)
        self.assertEqual(
            self.board.board[0][0], 3, "The dice was not placed correctly on the board."
        )
        self.assertEqual(
            self.board.board[3][0], 0, "The opponent's dice was not removed correctly."
        )
        self.assertEqual(
            self.board.board[4][0], 0, "The opponent's dice was not removed correctly."
        )
        self.assertEqual(
            self.board.board[5][0], 0, "The opponent's dice was not removed correctly."
        )

    def test_remove_opponents_dice_when_player_1(self):
        """Test the removal of opponent's dice with the same value in the same column."""
        self.board.board[0][0] = 3  # Player 1 places a 3
        self.board.board[1][1] = 3  # Player 1 places a 3 in a different column
        self.board.board[3][0] = 3  # Player 2 places a 3 in the same column
        self.board.remove_opponents_dice(3, 0, 3)  # Should trigger removal
        self.assertEqual(
            self.board.board[0][0], 0, "The opponent's dice was not removed correctly."
        )
        self.assertEqual(
            self.board.board[1][1], 3, "The opponent's dice was removed incorrectly."
        )
        self.assertEqual(self.board.board[3][0], 3, "Player Dice Was removed.")

    def test_remove_opponents_dice_when_player_2(self):
        """Test the removal of opponent's dice with the same value in the same column."""
        self.board.board[3][0] = 3  # Player 2 places a 3
        self.board.board[0][0] = 3
        self.board.remove_opponents_dice(0, 0, 3)  # Should trigger removal
        self.assertEqual(
            self.board.board[0][0], 3, "The opponent's dice was not removed correctly."
        )
        self.assertEqual(self.board.board[3][0], 0, "Player Dice Was removed.")

    def test_check_full(self):
        """Test the check_full function."""
        self.board.board = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [0, 0, 0],
        ]
        self.assertEqual(self.board.check_full(), True, "The board should be full.")

    def test_calculate_score_simple(self):
        """Test score calculation without any special scoring rules."""
        self.board.place_dice(0, 0, 1)
        self.board.place_dice(0, 1, 2)
        self.board.place_dice(0, 2, 3)
        player_one_score, player_two_score = self.board.calculate_score()
        self.assertEqual(player_one_score, 6, "Score calculation is incorrect.")
        self.assertEqual(player_two_score, 0, "Score calculation is incorrect.")

    def test_calculate_score_advanced(self):
        """Test score calculation with squares and cubes."""
        self.board.place_dice(0, 0, 2)  # This should be squared
        self.board.place_dice(1, 0, 2)  # This should be squared
        self.board.place_dice(3, 1, 3)  # This should be cubed
        self.board.place_dice(4, 1, 3)  # This should be cubed
        self.board.place_dice(5, 1, 3)  # This should be cubed
        player_one_score, player_two_score = self.board.calculate_score()
        self.assertEqual(
            player_one_score, 4, "Score calculation with squares is incorrect."
        )
        self.assertEqual(
            player_two_score, 27, "Score calculation with cubes is incorrect."
        )

    def test_display(self):
        """Test the display of the game board."""
        # fmt: off
        self.board.board = [
            [1, 0, 2],
            [0, 2, 0],
            [3, 1, 0],
            [0, 0, 4], 
            [5, 5, 5],
            [6, 6, 6]
        ]
        # fmt: on
        expected_lines = [
            "| 1 |   | 2 |",
            "|   | 2 |   |",
            "| 3 | 1 |   |",
            "|---|---|---|",
            "|   |   | 4 |",
            "| 5 | 5 | 5 |",
            "| 6 | 6 | 6 |",
        ]
        result_lines = self.board.display()
        self.assertEqual(result_lines, expected_lines, "The display is incorrect.")

    def test_get_available_moves(self):
        """Test the get_available_moves function."""
        # fmt: off
        self.board.board = [
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 0, 0],
        ]
        # fmt: on
        expected_moves_player_one = [(0, 2)]
        expected_moves_player_two = [(5, 1), (5, 2)]
        result_moves_player_one, result_moves_player_two = (
            self.board.get_available_moves()
        )
        self.assertEqual(result_moves_player_one, expected_moves_player_one)
        self.assertEqual(result_moves_player_two, expected_moves_player_two)


if __name__ == "__main__":
    unittest.main()
