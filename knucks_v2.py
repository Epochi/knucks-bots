"""Run a game of Knucklebones between two players"""

import game.player_actions_v2 as pa


def play_game(
    player1, player2, verbose=False, player1_reward_func=None, player2_reward_func=None
):
    """
    Play a game between two players.

    :param player1: The first player.
    :param player2: The second player.
    :param verbose: Whether to print game state at each turn.
    :return: The winner of the game.
    """
    engine = pa.start_game(verbose)

    player1_with_reward = {"player": player1, "reward_func": player1_reward_func}
    player2_with_reward = {"player": player2, "reward_func": player2_reward_func}

    while not pa.get_game_over(engine):
        if verbose:
            engine.game_board.display()

        pa.start_turn(engine)

        if engine.current_player == 0:
            current_player = player1_with_reward
        else:
            current_player = player2_with_reward

        if current_player["reward_func"] is not None:
            prev_state = current_player["player"].convert_state(
                pa.get_board_state(engine), pa.get_dice_value(engine)
            )
            prev_scores = pa.get_score(engine)

        action = current_player["player"].select_move(engine)

        if verbose:
            print(
                f"Player {engine.current_player} selected move: {action})."
            )

        try:
            pa.do_move(engine, action)

            if current_player["reward_func"] is not None:
                next_state = current_player["player"].convert_state(
                    pa.get_board_state(engine), 0
                )
                next_scores = pa.get_score(engine)
                reward = current_player["reward_func"](
                    engine, prev_state, prev_scores, action, next_state, next_scores
                )
                current_player["player"].update_q_table(
                    prev_state, action, reward, next_state
                )

        except ValueError as e:
            print(e)
            break

        pa.end_turn(engine)

    if pa.get_game_over(engine) is True:
        if verbose:
            engine.game_board.display()
            if engine.winner == 0:
                print("Player 1 - {player1_with_reward.nickname} has won the game!")
            elif engine.winner == 1:
                print("Player 2 - {player2_with_reward.nickname} has won the game!")
            else:
                print(f"Draw! No one won the game.")

    return engine.winner
