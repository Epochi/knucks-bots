"""Test cases for the simple_q_learning module"""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
from agents.simple_q_learning import QLearningAgent


class TestGameEngine(unittest.TestCase):
    """Test cases for the GameEngine class"""

    def setUp(self):
        """Set up the test case"""
        self.q_agent = QLearningAgent()

    def test_state_to_string(self):
        """Test if roll_dice method returns a value"""
        board_state = [[1, 2, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]]
        dice_value = 4
        state = self.q_agent.state_to_string(board_state, dice_value)
        self.assertEqual(state, "1200000000000000014")

    def test_update_q_table(self):
        """Test if the Q-table is updated correctly"""
        q_agent = QLearningAgent(
            learning_rate=1,
            discount_factor=0,
            exploration_rate=0,
            exploration_decay=0,
            min_exploration_rate=0,
        )

        prev_state = "1200000000000000014"
        action = (0, 2)
        reward = 1
        new_state = "1240000000000000010"

        q_agent.update_q_table(prev_state, action, reward, new_state)
        self.assertEqual(q_agent.q_table[prev_state][action], reward)

    @patch("agents.simple_q_learning.pa")
    def test_select_move_random(self, mock_pa):
        """Test if the agent selects a move correctly"""
        q_agent = QLearningAgent(
            learning_rate=1,
            discount_factor=0,
            exploration_rate=1,
            exploration_decay=0,
            min_exploration_rate=0,
        )

        board_state = [
            [1, 2, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]

        game_engine = MagicMock()
        mock_pa.get_board_state.return_value = board_state
        mock_pa.get_dice_value.return_value = 4
        # game_engine.get_board_state.return_value = board_state
        # game_engine.get_dice_value.return_value = 4
        mock_pa.get_available_moves.return_value = [
            (1, 0),
            (1, 1),
            (1, 2),
        ]

        action = q_agent.select_move(game_engine)
        self.assertIn(action, [(1, 0), (1, 1), (1, 2)])

    @patch("agents.simple_q_learning.pa")
    def test_select_move_max_q(self, mock_pa):
        """Test if the agent selects a move correctly"""
        q_agent = QLearningAgent(
            learning_rate=1,
            discount_factor=0,
            exploration_rate=0,
            exploration_decay=0,
            min_exploration_rate=0,
        )

        board_state = [
            [1, 2, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]

        game_engine = MagicMock()
        mock_pa.get_board_state.return_value = board_state
        mock_pa.get_dice_value.return_value = 4
        # game_engine.get_board_state.return_value = board_state
        # game_engine.get_dice_value.return_value = 4
        mock_pa.get_available_moves.return_value = [
            (1, 0),
            (1, 1),
            (1, 2),
        ]

        q_agent.q_table["1200000000000000014"] = {(1, 0): 1, (1, 1): 2, (1, 2): 3}

        action = q_agent.select_move(game_engine)
        print("Actionnnn: ", action)
        self.assertEqual(action, (1, 2))
