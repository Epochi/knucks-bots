"""Utiltiy module for various reward models."""

import game.player_actions_v2 as pa
from game.game_engine_v2 import GameEngine


def calculate_reward_template(
    game_engine: GameEngine,
    prev_state: str,
    prev_scores: tuple,
    action: tuple,
    new_state: str,
    new_scores: tuple,
    dice_placed: int,
):
    """
    Calculate the reward based on the game state.
    Adjust this function based on your reward strategy.

    This function is a template and should not be called directly. Instead,
    create a new function that accepts the same arguments and implements the
    desired behavior.

    Args:
    game_engine: The current game engine.
    prev_state: The previous state.
    prev_scores: The previous scores.
    action: The action taken.
    new_state: The new state after the action.
    new_scores: The new scores after the action.
    dice_placed: The dice value placed.

    Returns:
    The calculated reward.
    """
    raise NotImplementedError(
        "This function is a template and should not be called directly."
    )


def calculate_for_score(
    game_engine: GameEngine,
    prev_state: str,
    prev_scores: tuple,
    action: tuple,
    new_state: str,
    new_scores: tuple,
    dice_placed: int,
):
    """
    Calculate the reward based on score movement.
    """

    reward = 0
    player_score_diff = new_scores[0] - prev_scores[0]
    opponent_score_diff = new_scores[1] - prev_scores[1]

    # if player increased the score, reward very well
    if player_score_diff > 0:
        reward += player_score_diff

    # if player decreased the score of the opponent, reward very well
    if opponent_score_diff < 0:
        reward += abs(opponent_score_diff)

    if pa.get_game_over(game_engine):
        if game_engine.winner == pa.did_i_win(game_engine):
            reward += 1000
    return reward


def calculate_for_own_score_only(
    game_engine: GameEngine,
    prev_state: str,
    prev_scores: tuple,
    action: tuple,
    new_state: str,
    new_scores: tuple,
    dice_placed: int,
):
    """
    Calculate the reward based on score movement.
    """

    reward = 0
    player_score_diff = new_scores[0] - prev_scores[0]

    # if player increased the score, reward very well
    if player_score_diff > 0:
        reward += player_score_diff

    if pa.get_game_over(game_engine):
        if game_engine.winner == pa.did_i_win(game_engine):
            reward += 1000
    return reward


def one_side_for_multiply_and_win_only(
    game_engine: GameEngine,
    prev_state: str,
    prev_scores: tuple,
    action: tuple,
    new_state: str,
    new_scores: tuple,
    dice_placed: int,
):
    """
    Only give rewards if agent manages to multiply the dice and win the game.
    """
    reward = 0
    player_score_diff = new_scores[0] - prev_scores[0]
    if player_score_diff > dice_placed:
        reward += player_score_diff

    if pa.get_game_over(game_engine):
        if game_engine.winner == pa.did_i_win(game_engine):
            reward += 50
    return reward
