"""Random Agent Module"""

import random
from game.game_engine import AbstractAgent


class RandomAgent(AbstractAgent):
    """
    A simple agent that selects its move randomly from available moves.
    """

    def select_move(self, game_board, dice_value):
        available_moves = self.get_player_available_moves(game_board)
        if not available_moves:
            # throw error, invalid path
            raise ValueError("No available moves, Invalid path")
        return random.choice(available_moves)
