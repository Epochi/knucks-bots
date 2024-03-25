"""Utiltiy module for various reward models."""

import game.player_actions as pa
from game.game_engine import GameEngine


def calculate_reward_template(
    game_engine: GameEngine,
    prev_state: str,
    prev_scores: tuple,
    action: tuple,
    new_state: str,
    new_scores: tuple,
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

    Returns:
    The calculated reward.
    """
    raise NotImplementedError(
        "This function is a template and should not be called directly."
    )


def calculate_reward_per_win(
    game_engine: GameEngine,
    prev_state: str,
    prev_scores: tuple,
    action: tuple,
    new_state: str,
    new_scores: tuple,
):
    """
    Calculate the reward based on the game state.
    Adjust this function based on your reward strategy.
    """
    if pa.get_game_over(game_engine):
        if game_engine.winner == 1:
            return 100
        elif game_engine.winner == 2:
            return -100
        else:
            return 0  # Draw
    return -1  # Penalize long games


def calculate_for_score(
    game_engine: GameEngine,
    prev_state: str,
    prev_scores: tuple,
    action: tuple,
    new_state: str,
    new_scores: tuple,
):
    """
    Calculate the reward based on score movement.
    """

    reward = 0
    player_score_diff = new_scores[0] - prev_scores[0]
    player_vs_opponent_score_diff = new_scores[0] - new_scores[1]
    opponent_score_diff = new_scores[1] - prev_scores[1]

    # if player increased the score, reward well
    if player_score_diff > 0:
        reward += player_score_diff / 2

    # if player decreased the score of the opponent, reward very well
    if opponent_score_diff < 0:
        reward += abs(opponent_score_diff)

    # if player has more points than the opponent, reward ok
    if player_vs_opponent_score_diff > 0:
        reward += player_vs_opponent_score_diff / 3

    if pa.get_game_over(game_engine):
        if game_engine.winner == 1:
            reward += 1000
        elif game_engine.winner == 2:
            reward -= 1000
    return reward
