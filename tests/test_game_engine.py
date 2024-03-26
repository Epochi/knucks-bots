"""Tests for the GameEngine class."""

import unittest
from unittest.mock import patch
from game.game_engine import GameEngine


class TestGameEngine(unittest.TestCase):
    """Tests for the GameEngine class."""

    def setUp(self):
        """Initialize the GameEngine object before each test."""
        self.game_engine = GameEngine(enable_print=False)

    def test_init(self):
        """Test the initialization of the GameEngine."""
        self.assertFalse(self.game_engine.game_over, "Game should not be over on init.")
        self.assertIsNone(self.game_engine.winner, "There should be no winner on init.")
        self.assertIn(
            self.game_engine.current_player, [1, 2], "Invalid starting player."
        )

    @patch("game.game_board.GameBoard.roll_dice")
    def test_start_turn(self, mock_roll_dice):
        """Test starting a turn."""
        mock_roll_dice.return_value = 4
        self.game_engine.start_turn()
        self.assertEqual(
            self.game_engine.dice_value, 4, "Dice value should be set after rolling."
        )

    def test_switch_player(self):
        """Test switching the current player."""
        current_player = self.game_engine.current_player
        self.game_engine.switch_player()
        self.assertNotEqual(
            self.game_engine.current_player,
            current_player,
            "The current player should have switched.",
        )

    @patch("game.game_board.GameBoard.check_full")
    def test_check_game_over(self, mock_check_full):
        """Test the check_game_over method."""
        mock_check_full.return_value = True
        self.game_engine.check_game_over()
        self.assertTrue(self.game_engine.game_over, "Game should be marked as over.")

    @patch("game.game_board.GameBoard.place_dice")
    def test_do_move(self, mock_place_dice):
        """Test making a move."""
        self.game_engine.dice_value = 5
        move_made = self.game_engine.do_move(0, 0)
        self.assertTrue(move_made, "Move should be successful.")
        mock_place_dice.assert_called_once_with(0, 0, 5)

    @patch("game.game_board.GameBoard.check_full", return_value=False)
    def test_end_turn_without_game_over(self, mock_check_full):
        """Test ending a turn without the game being over."""
        self.game_engine.end_turn()
        self.assertIsNone(
            self.game_engine.dice_value,
            "Dice value should be reset at the end of a turn.",
        )
        self.assertFalse(self.game_engine.game_over, "Game should not be over.")

    @patch("game.game_board.GameBoard.check_full", return_value=True)
    @patch("game.game_board.GameBoard.calculate_score", return_value=(10, 5))
    def test_end_turn_with_game_over(self, mock_calculate_score, mock_check_full):
        """Test ending a turn with the game being over."""
        self.game_engine.end_turn()
        self.assertTrue(self.game_engine.game_over, "Game should be over.")
        self.assertEqual(self.game_engine.winner, 1, "Winner should be player 1.")

    def test_attempt_to_start_turn_when_game_over(self):
        """Test starting a turn when the game is already over."""
        self.game_engine.game_over = True
        with self.assertRaises(ValueError):
            self.game_engine.start_turn()

    def test_attempt_to_do_move_when_game_over(self):
        """Test making a move when the game is already over."""
        self.game_engine.game_over = True
        with self.assertRaises(ValueError):
            self.game_engine.do_move(0, 0)

    def test_attempt_to_end_turn_when_game_over(self):
        """Test ending a turn when the game is already over."""
        self.game_engine.game_over = True
        with self.assertRaises(ValueError):
            self.game_engine.end_turn()


if __name__ == "__main__":
    unittest.main()
