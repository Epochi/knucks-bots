"""
Utility functions for player inetracting with the game engine.
Players should not need to interact with the game engine class directly.

We also flip player board perspective here for player 2 to make it easier to train the model
since we randomly assign which player goes first instead.

For simplicity, we let game engine manage current player instead of exposing it to agent or player.
"""

from game.game_engine import GameEngine


def start_game(enable_print=False):
    """
    Starts a new game

    :param enable_print: enable print
    :return: game engine
    """
    return GameEngine(enable_print)


def start_turn(engine: GameEngine):
    """
    Start Player Turn

    :param engine: game engine
    """
    engine.start_turn()


def do_move(engine: GameEngine, row: int, col: int):
    """
    make a dice placement
    (if player 2, flip the board perspective and call engine do move)

    :param engine: game engine
    :param row: row to place dice
    :param col: column to place dice
    :return: True if move success False otherwise
    """
    if engine.current_player == 2:
        row = row + 3
    return engine.do_move(row, col)


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
    return engine.game_board.get_board_from_1p_pov(engine.current_player)


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
    player_1_score, player_2_score = engine.game_board.calculate_score()
    if engine.current_player == 1:
        player_score = player_1_score
        opponent_score = player_2_score
    else:
        player_score = player_2_score
        opponent_score = player_1_score
    return player_score, opponent_score


def get_available_moves(engine: GameEngine):
    """
    get available moves for player

    :param engine: game engine
    :return: available moves
    """
    return engine.game_board.get_available_moves_from_1p_pov(engine.current_player)


def get_winner(engine: GameEngine):
    """
    get winner of the game

    :param engine: game engine
    :return: winner
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
