"""Tests for game player actions."""

from unittest.mock import MagicMock
import unittest
import game.player_actions as pa


class TestGameUtilityFunctions(unittest.TestCase):
    """Tests for game utility functions that manipulate exposed state."""

    def setUp(self):
        """Initialize a mock game engine before each test."""
        self.mock_engine = MagicMock()

    def test_do_move_player_1(self):
        """Test making a move as player 1."""
        self.mock_engine.current_player = 1
        result = pa.do_move(self.mock_engine, 0, 1)
        self.mock_engine.do_move.assert_called_once_with(0, 1)
        self.assertTrue(result)

    def test_do_move_player_2(self):
        """Test making a move as player 2, with board perspective flipped."""
        self.mock_engine.current_player = 2
        result = pa.do_move(self.mock_engine, 0, 1)
        self.mock_engine.do_move.assert_called_once_with(3, 1)
        self.assertTrue(result)

    def test_get_score_player_1_perspective(self):
        """Test getting the score from player 1's perspective."""
        # Setup the mock to return specific scores
        self.mock_engine.game_board.calculate_score.return_value = (10, 5)
        self.mock_engine.current_player = 1

        # Call the function under test
        player_score, opponent_score = pa.get_score(self.mock_engine)

        # Verify the result is as expected for player 1
        self.assertEqual(player_score, 10, "Player score should be 10 for player 1.")
        self.assertEqual(opponent_score, 5, "Opponent score should be 5 for player 1.")

    def test_get_score_player_2_perspective(self):
        """Test getting the score from player 2's perspective."""
        # Setup the mock to return specific scores
        self.mock_engine.game_board.calculate_score.return_value = (10, 5)
        self.mock_engine.current_player = 2

        # Call the function under test
        player_score, opponent_score = pa.get_score(self.mock_engine)

        # Verify the result is as expected for player 2
        self.assertEqual(player_score, 5, "Player score should be 5 for player 2.")
        self.assertEqual(
            opponent_score, 10, "Opponent score should be 10 for player 2."
        )


if __name__ == "__main__":
    unittest.main()
