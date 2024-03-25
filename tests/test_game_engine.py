"""Tests for the GameEngine class"""

import unittest
from unittest.mock import MagicMock
from game.game_engine import (
    GameEngine,
)


class TestGameEngine(unittest.TestCase):
    """Test cases for the GameEngine class"""

    def setUp(self):
        self.game_engine = GameEngine()
        # Mocking the GameBoard used within GameEngine
        self.game_engine.game_board = unittest.mock.MagicMock()
        self.game_engine.game_board.roll_dice.return_value = 4
        self.game_engine.game_board.calculate_score.return_value = [15, 20]

    def test_roll_dice(self):
        """Test if roll_dice method returns a value"""
        dice_value = self.game_engine.roll_dice()
        self.game_engine.game_board.roll_dice.assert_called_once()
        self.assertTrue(1 <= dice_value <= 6)

    def test_make_move_wrong_player(self):
        """Test making a move for the wrong player"""
        result = self.game_engine.make_move(
            2, 4, 0, 0
        )  # It's player 1's turn, not player 2
        self.assertFalse(result)

    def test_make_move_correct_player(self):
        """Test making a valid move"""
        self.game_engine.game_board.place_dice.side_effect = (
            lambda row, col, dice_value: None
        )
        result = self.game_engine.make_move(1, 4, 0, 0)  # It's player 1's turn
        self.assertTrue(result)
        self.game_engine.game_board.place_dice.assert_called_once_with(0, 0, 4)

    def test_switch_player(self):
        """Test if the current player switches correctly"""
        self.game_engine.switch_player()
        self.assertEqual(self.game_engine.current_player, 2)
        self.game_engine.switch_player()
        self.assertEqual(self.game_engine.current_player, 1)

    def test_check_game_over(self):
        """Test if the game over condition and winner are set correctly"""
        self.game_engine.game_board.check_full.return_value = True
        self.game_engine.check_game_over()
        self.assertTrue(self.game_engine.game_over)
        self.assertEqual(
            self.game_engine.winner, 2
        )  # Assuming player 2 has higher score


if __name__ == "__main__":
    unittest.main()
