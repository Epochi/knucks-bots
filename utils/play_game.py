"""Utility to play and train agents against each other."""

import signal
import game.player_actions_v2 as pa
from agents.base_agent_v2 import AbstractAgent
from game.game_engine_v2 import GameEngine

interrupted = False


class PlayingAgent:
    def __init__(self, agent: AbstractAgent, reward_func=None, model_name: str = None):
        self.agent = agent
        if model_name is not None:
            self.agent.load_model(f"./models/{model_name}.pkl")
        self.reward_func = reward_func
        self.model_name = model_name


class GameRules:
    def __init__(
        self, max_dice_value: int = 6, should_remove_opponents_dice: bool = False
    ):
        self.max_dice_value = max_dice_value
        self.should_remove_opponents_dice = should_remove_opponents_dice
        self.safe_mode = False


def player_move(game_engine: GameEngine, agent: PlayingAgent):
    pa.start_turn(game_engine)

    action = agent.agent.select_move(game_engine)
    pa.do_move(game_engine, action)

    pa.end_turn(game_engine)
