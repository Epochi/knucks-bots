"""Utiltiy module for various reward models."""

import game.player_actions_v2 as pa
from game.game_engine_v2 import GameEngine
from abc import ABC, abstractmethod


class AbstractRewardModel(ABC):
    """
    Abstract base class for reward models.
    """

    def __init__(self):
        pass

    @abstractmethod
    def calculate_reward(
        self,
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


class ParametrizedRewardModel(AbstractRewardModel):
    """
    A reward model that accepts parameters for customization.
    """

    def __init__(
        self,
        reward_score_increase_multiplier=0,
        reward_score_incerease_multiplier_multiplier=0,
        reward_opponent_score_decrease_multiplier=0,
        reward_opponent_score_decrease_multiplier_multiplier=0,
        reward_score_diff_multiplier=0,
        reward_win_amount=0,
        reward_loss_amount=0,
    ):
        self.reward_score_increase_multiplier = reward_score_increase_multiplier
        self.reward_multiplier_multiplier = reward_score_incerease_multiplier_multiplier
        self.reward_opponent_score_decrease_multiplier = (
            reward_opponent_score_decrease_multiplier
        )
        self.reward_opponent_score_decrease_multiplier_multiplier = (
            reward_opponent_score_decrease_multiplier_multiplier
        )
        self.reward_win_multiplier = reward_win_amount
        self.reward_loss_multiplier = reward_loss_amount
        self.reward_score_diff_multiplier = reward_score_diff_multiplier

    def calculate_reward(
        self,
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
        reward = 0
        player_score_diff = new_scores[0] - prev_scores[0]
        opponent_score_diff = new_scores[1] - prev_scores[1]
        score_diff = player_score_diff - opponent_score_diff

        # reward player for increasing score
        reward += player_score_diff * self.reward_score_increase_multiplier

        # reward total score increase, including opponent score decrease
        reward += score_diff * self.reward_score_diff_multiplier

        # reward player for increasing score by placing dice on the same column as another dice of the same value
        if player_score_diff > dice_placed:
            reward += player_score_diff * self.reward_multiplier_multiplier

        if opponent_score_diff < 0:
            reward += (
                abs(opponent_score_diff)
                * self.reward_opponent_score_decrease_multiplier
            )

        if abs(opponent_score_diff) > dice_placed:
            reward += (
                abs(opponent_score_diff)
                * self.reward_opponent_score_decrease_multiplier_multiplier
            )

        if pa.get_game_over(game_engine):
            if game_engine.winner == pa.did_i_win(game_engine):
                reward += self.reward_win_multiplier
            else:
                reward += self.reward_loss_multiplier
        return reward


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


def calculate_for_multiples_and_removals_score(
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

    if player_score_diff > dice_placed:
        reward += player_score_diff

    if abs(opponent_score_diff) > dice_placed:
        reward += abs(opponent_score_diff)

    if pa.get_game_over(game_engine):
        if game_engine.winner == pa.did_i_win(game_engine):
            reward += 100
    return reward


def calculate_score_parametrized(
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

    if player_score_diff > dice_placed:
        reward += player_score_diff

    if opponent_score_diff < 0:
        reward += abs(opponent_score_diff)
    if abs(opponent_score_diff) > dice_placed:
        reward += abs(opponent_score_diff)

    if pa.get_game_over(game_engine):
        if game_engine.winner == pa.did_i_win(game_engine):
            reward += 100
    return reward


def calculate_for_multiples_and_any_removals_score(
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

    if player_score_diff > dice_placed:
        reward += player_score_diff

    if opponent_score_diff < 0:
        reward += abs(opponent_score_diff)
    if abs(opponent_score_diff) > dice_placed:
        reward += abs(opponent_score_diff)

    if pa.get_game_over(game_engine):
        if game_engine.winner == pa.did_i_win(game_engine):
            reward += 100
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
