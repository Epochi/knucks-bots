"""
Utility functions for player inetracting with the game engine.
Players should not need to interact with the game engine class directly.

We also flip player board perspective here for player 2 to make it easier to train the model
since we randomly assign which player goes first instead.

For simplicity, we let game engine manage current player instead of exposing it to agent or player.
"""

from game.game_engine_v2 import GameEngine


def start_game(
    enable_print=None,
    max_dice_value=None,
    should_remove_opponents_dice=None,
    safe_mode=None,
):
    """
    Starts a new game

    :param enable_print: enable print
    :return: game engine
    """
    return GameEngine(
        enable_print, max_dice_value, should_remove_opponents_dice, safe_mode
    )


def start_turn(engine: GameEngine):
    """
    Start Player Turn

    :param engine: game engine
    """
    engine.start_turn()


def do_move(engine: GameEngine, col: int):
    """
    make a dice placement
    (if player 2, flip the board perspective and call engine do move)

    :param engine: game engine
    :param col: column to place dice
    :return: True if move success False otherwise
    """
    return engine.do_move(col)


def end_turn(engine: GameEngine):
    """
    End player turn
    """
    engine.end_turn()


def get_board_state(engine: GameEngine):
    """
    Get picture of the current board
    (if player 2, flip the board perspective)

    :param engine: game engine
    :return: player board state
    """
    if engine.current_player == 1:
        return engine.game_board.player_2_board, engine.game_board.player_1_board
    return engine.game_board.player_1_board, engine.game_board.player_2_board


def get_dice_value(engine: GameEngine):
    """
    get rolled dice value for current turn

    :param engine: game engine
    :return: dice value
    """
    return engine.dice_value


def get_score(engine: GameEngine):
    """
    get current game score

    :param engine: game engine
    :return: player score, opponent score
    """
    if engine.current_player == 0:
        return engine.game_board.player_1_score, engine.game_board.player_2_score
    else:
        return engine.game_board.player_2_score, engine.game_board.player_1_score


def get_available_moves(engine: GameEngine):
    """
    get available moves for player

    :param engine: game engine
    :return: available moves
    """
    return engine.game_board.get_available_moves(engine.current_player)


def did_i_win(engine: GameEngine):
    """
    check if you won the game

    :param engine: game engine
    :return: return if you won or not
    """
    if engine.game_over:
        if engine.winner == engine.current_player:
            return 1
        if engine.winner == -1:
            return 0
        return -1


def get_winner(engine: GameEngine):
    """
    check if you won the game

    :param engine: game engine
    :return: return if you won or not
    """
    return engine.winner


def get_game_over(engine: GameEngine):
    """
    get game over status

    :param engine: game engine
    :return: game over status
    """
    return engine.game_over


def get_current_player(engine: GameEngine):
    """
    get current player

    :param engine: game engine
    :return: current player
    """
    return engine.current_player


def display_board(engine: GameEngine):
    """
    Display the current game board
    """
    engine.game_board.display()
