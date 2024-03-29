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


def player_move(game_engine: GameEngine, agent: PlayingAgent, game_rules=GameRules()):
    pa.start_turn(game_engine)
    dice_value = pa.get_dice_value(game_engine)

    # agent always as player 1
    if agent.reward_func is not None:
        pre_move_state = agent.agent.convert_state(
            pa.get_board_state(game_engine, game_rules.should_remove_opponents_dice),
            dice_value,
        )
        pre_move_scores = pa.get_score(game_engine)

    action = agent.agent.select_move(game_engine)
    pa.do_move(game_engine, action)

    if agent.reward_func is not None:
        post_move_board_state = pa.get_board_state(
            game_engine, game_rules.should_remove_opponents_dice
        )
        # only show player side of the board to the agent if we're not removing opponents dice
        # since opponent dice have no impact on the agent's decision
        post_move_states = [
            agent.agent.convert_state(post_move_board_state, i) for i in range(1, 7)
        ]
        post_move_score = pa.get_score(game_engine)

        reward = agent.reward_func(
            game_engine=game_engine,
            prev_state=pre_move_state,
            prev_scores=pre_move_scores,
            action=action,
            new_state=post_move_states,
            new_scores=post_move_score,
            dice_placed=dice_value,
        )

        agent.agent.learn(
            prev_state=pre_move_state,
            action=action,
            reward=reward,
            new_states=post_move_states,
            game_over=pa.get_game_over(game_engine),
            winner=pa.did_i_win(game_engine),
        )

    pa.end_turn(game_engine)
