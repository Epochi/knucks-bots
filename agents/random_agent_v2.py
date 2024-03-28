"""Random Agent Module"""

import random
from agents.base_agent_v2 import AbstractAgent
import game.player_actions_v2 as pa


class RandomAgent(AbstractAgent):
    """
    A simple agent that selects its move randomly from available moves.
    """

    def __init__(self):
        super().__init__()
        self.nickname = "Wild Card"

    def select_move(self, game_engine):
        available_moves = pa.get_available_moves(game_engine)
        if available_moves is None or len(available_moves) == 0:
            # throw error, invalid path
            print("Available Moves", available_moves)
            print("Current Player", pa.get_current_player(game_engine))
            print("is game over", pa.get_game_over(game_engine))
            print("winner", pa.get_winner(game_engine))
            print("board state", pa.get_board_state(game_engine))
            print("player_1_dice", game_engine.game_board.player_1_board_placed_dice)
            print("player_2_dice", game_engine.game_board.player_2_board_placed_dice)

            raise ValueError("No available moves, Invalid State.")

        return random.choice(available_moves)
