"""Run a game of Knucklebones between two players"""

import game.player_actions_v2 as pa
from utils.play_game import PlayingAgent, GameRules, player_move


def play_game(
    player_1: PlayingAgent,
    player_2: PlayingAgent,
    game_rules: GameRules,
    verbose: bool = False,
):
    """
    Play a game between two players.

    :param player1: The first player.
    :param player2: The second player.
    :param verbose: Whether to print game state at each turn.
    :return: The winner of the game.
    """
    game_engine = pa.start_game(
        enable_print=False,
        max_dice_value=game_rules.max_dice_value,
        should_remove_opponents_dice=game_rules.should_remove_opponents_dice,
        safe_mode=False,
    )

    while not pa.get_game_over(game_engine):
        current_player = (
            player_1 if pa.get_current_player(game_engine) == 0 else player_2
        )

        player_move(game_engine, current_player, game_rules)

    if pa.get_game_over(game_engine) is True:
        if verbose:
            game_engine.game_board.display()
            if game_engine.winner == 0:
                print("Player 1 - {player1_with_reward.nickname} has won the game!")
            elif game_engine.winner == 1:
                print("Player 2 - {player2_with_reward.nickname} has won the game!")
            else:
                print(f"Draw! No one won the game.")

    return game_engine.winner
