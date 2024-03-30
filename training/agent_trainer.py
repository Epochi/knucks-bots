"""Utility to train agents against each other."""

import signal
import pickle
import time
import copy
from math import comb
import game.player_actions_v2 as pa
from utils.play_game import PlayingAgent, GameRules, player_move
from game.game_engine_v2 import GameEngine
import datetime

interrupted = False


# Signal handler function
def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("CTRL+C detected. Finishing the current episode...")


signal.signal(signal.SIGINT, signal_handler)


def train_agents(
    player_1: PlayingAgent,
    player_2: PlayingAgent,
    game_rules: GameRules,
    episodes=1000,
    write_result_history=False,
):
    """train Q-Learning agent against a random agent"""
    wins = 0
    losses = 0
    draws = 0
    if write_result_history:
        result_history = []

    heartbeat = min(10 / 100 * episodes, 10000)
    perf_timer_total_run = time.time()
    perf_timer = time.time()
    global interrupted

    n = game_rules.max_dice_value + 1  # dice size + empty
    r = 3  # column size
    perf_timer = time.time()
    total_combinations = comb(n + r - 1, r)
    one_side = total_combinations**3 * n
    two_sides = total_combinations**6 * n
    # default rules
    # Total combinations: 84
    # Total combinations for one side: 4,148,928
    # Total combinations for two sides: 2,459,086,221,312
    # Simple Q table would need to use around 1,036 terabytes of memory to store all possible states

    average_moves_per_game = 0
    total_moves = 0

    for episode in range(episodes):
        game_engine = pa.start_game(
            enable_print=False,
            max_dice_value=game_rules.max_dice_value,
            should_remove_opponents_dice=game_rules.should_remove_opponents_dice,
            safe_mode=False,
        )

        is_heartbeat = episode % heartbeat == 0

        move_counter = 0

        previous_move = {
            player_1: {
                "state": None,
                "scores": None,
                "dice": None,
                "action": None,
            },
            player_2: {
                "state": None,
                "scores": None,
                "dice": None,
                "action": None,
            },
        }
        while not pa.get_game_over(game_engine):
            if pa.get_current_player(game_engine) == 0:
                current_player = player_1
            else:
                current_player = player_2

            pa.start_turn(game_engine)
            dice_value = pa.get_dice_value(game_engine)

            # we're selecting the action before we calculated reward for the previous move :thinking:

            if current_player.reward_func is not None:
                next_state = current_player.agent.convert_state(
                    pa.get_board_state(game_engine),
                    dice_value,
                )
                next_state_scores = pa.get_score(game_engine)

                if move_counter > 1:
                    delayed_reward(
                        game_engine=game_engine,
                        player=current_player,
                        prev_state=previous_move[current_player]["state"],
                        prev_scores=previous_move[current_player]["scores"],
                        prev_dice=previous_move[current_player]["dice"],
                        prev_action=previous_move[current_player]["action"],
                        next_state=next_state,
                        next_scores=next_state_scores,
                        game_over=pa.get_game_over(game_engine),
                        winner=pa.get_winner(game_engine),
                    )
                previous_move[current_player]["state"] = copy.deepcopy(next_state)
                previous_move[current_player]["scores"] = copy.deepcopy(
                    next_state_scores
                )
                previous_move[current_player]["dice"] = dice_value

            action = current_player.agent.select_move(game_engine)
            pa.do_move(game_engine, action)
            pa.end_turn(game_engine)

            if current_player.reward_func is not None:
                previous_move[current_player]["action"] = action

            move_counter += 1

        # run delayed reward after the game is over
        # we know that game engine current player is the one that did the last move, since we do not switch players after end_turn
        if pa.get_current_player(game_engine) == 0:
            last_player_index = 0
            last_player = player_1
            before_last_player = player_2
            before_last_player_index = 1
        else:
            last_player_index = 1
            last_player = player_2
            before_last_player = player_1
            before_last_player_index = 0

        if last_player.reward_func is not None:
            # for the last state we always pass dice value 0, since it's not relevant
            delayed_reward(
                game_engine=game_engine,
                player=last_player,
                prev_state=previous_move[last_player]["state"],
                prev_scores=previous_move[last_player]["scores"],
                prev_dice=previous_move[last_player]["dice"],
                prev_action=previous_move[last_player]["action"],
                next_state=last_player.agent.convert_state(
                    pa.get_board_state(
                        engine=game_engine,
                        player=last_player_index,
                    ),
                    dice_value,
                ),
                next_scores=pa.get_score(engine=game_engine, player=last_player_index),
                game_over=True,
                winner=pa.get_winner(game_engine),
            )

        if before_last_player.reward_func is not None:
            # for the last state we always pass dice value 0, since it's not relevant
            delayed_reward(
                game_engine=game_engine,
                player=before_last_player,
                prev_state=previous_move[before_last_player]["state"],
                prev_scores=previous_move[before_last_player]["scores"],
                prev_dice=previous_move[before_last_player]["dice"],
                prev_action=previous_move[before_last_player]["action"],
                next_state=before_last_player.agent.convert_state(
                    pa.get_board_state(
                        engine=game_engine,
                        player=last_player_index,
                    ),
                    dice_value,
                ),
                next_scores=pa.get_score(
                    engine=game_engine, player=before_last_player_index
                ),
                game_over=True,
                winner=pa.get_winner(game_engine),
            )

        winner = pa.get_winner(game_engine)
        if winner == 0:
            wins += 1
            if write_result_history:
                result_history.append(1)
        elif winner == 1:
            losses += 1
            if write_result_history:
                result_history.append(-1)
        else:
            draws += 1
            if write_result_history:
                result_history.append(0)

        total_moves += move_counter
        average_moves_per_game = total_moves / (episode + 1)
        move_counter = 0

        # sanity check, print game every 10% of the episodes, but not more rarely than 1000
        if is_heartbeat:
            print(f"Episode {episode:,}/{episodes:,}")

            print("\nPerformance stats:")
            if perf_timer:
                print(
                    f"Time taken for {heartbeat:,} episodes: {str(datetime.timedelta(seconds=int(time.time() - perf_timer)))}"
                )
            # objgraph.show_most_common_types()

            if (
                hasattr(player_1.agent, "model")
                and isinstance(player_1.agent.model, dict)
                and len(player_1.agent.model) > 0
            ):
                print(
                    f"keys in {player_1.agent.nickname} model: {len(player_1.agent.model):,}"
                )
                print(
                    f"which is {len(player_1.agent.model) / two_sides if game_rules.should_remove_opponents_dice else one_side:.2%} of all possible states"
                )
            if (
                hasattr(player_2.agent, "model")
                and isinstance(player_2.agent.model, dict)
                and len(player_2.agent.model) > 0
            ):
                print(
                    f"keys in {player_2.agent.nickname} model: {len(player_2.agent.model):,}"
                )
                print(
                    f"which is {len(player_2.agent.model) / two_sides if game_rules.should_remove_opponents_dice else one_side:.2%} of all possible states"
                )
            print(f"Average moves per game: {average_moves_per_game:.2f}")

            # if player_1.agent.modelType == "DQ":
            #     for name, param in player_1.agent.model.named_parameters():
            #         print(
            #             f"Layer Name: {name}, Parameter Size: {param.size()}, Param Data: {param.data}"
            #         )

            # if player_2.agent.modelType == "DQ":
            #     for name, param in player_2.agent.model.named_parameters():
            #         print(
            #             f"Layer Name: {name}, Parameter Size: {param.size()}, Param Data Len: {param.data}"
            #         )

            print("\nMotivational Stats:")
            print(
                f"{player_1.agent.nickname} Wins: {wins:,}, {player_2.agent.nickname} Wins: {losses:,}, Draws: {draws:,}"
            )
            # print diff in wins and % from mean
            print(
                f"\nWin difference: {abs(wins - losses):,} - % from mean: {abs(wins - losses) / (wins + losses + draws) * 100:.2f}% - in the lead: {player_1.agent.nickname if wins > losses else player_2.agent.nickname}"
            )
            print(
                "\nWe just won!"
                if pa.get_winner(game_engine) == 0
                else "\nWe just lost!"
            )
            print(
                f"Scores: {player_1.agent.nickname}: {pa.get_score(game_engine)[0]},  {player_2.agent.nickname}: {pa.get_score(game_engine)[1]}"
            )

            if hasattr(player_1.agent, "learning_rate"):
                print("\nLearning Stats:")
                print(f"learning rate: {player_1.agent.learning_rate}")
                print(f"exploration rate: {player_1.agent.exploration_rate}")
            if hasattr(player_2.agent, "learning_rate"):
                print("\nLearning Stats:")
                print(f"learning rate: {player_2.agent.learning_rate}")
                print(f"exploration rate: {player_2.agent.exploration_rate}")

            pa.display_board(game_engine)
            print("\n")

        if interrupted:
            print("Stop requested. Exiting training.")
            break

    print(
        f"Total time taken: {time.time() - perf_timer_total_run} for {wins+draws+losses:,} episodes"
    )
    if player_1.model_name is not None:
        player_1.agent.save_model(f"./models/{player_1.model_name}.pkl")

    if player_2.model_name is not None:
        player_2.agent.save_model(f"./models/{player_2.model_name}.pkl")

    if write_result_history:
        save_list(
            result_history,
            f"./models/{player_1.agent.nickname}_vs_{player_2.agent.nickname}_result_history_{time.time()}.pkl",
        )

    print(
        f"Training completed. {player_1.agent.nickname} Wins: {wins:,}, {player_2.agent.nickname} Wins: {losses:,}, Draws: {draws:,}"
    )

    return wins, losses, draws


def delayed_reward(
    game_engine: GameEngine,
    player: PlayingAgent,
    prev_state: str,
    prev_scores: tuple,
    prev_action: tuple,
    prev_dice: int,
    next_state: str,
    next_scores: tuple,
    game_over: bool,
    winner: int,
):
    reward = player.reward_func(
        game_engine=game_engine,
        prev_state=prev_state,
        prev_scores=prev_scores,
        action=prev_action,
        # we're calculating score for the previous turn
        dice_placed=prev_dice,
        # this is really before the new move, but from perspective of the previous turn, it's the new state
        next_state=next_state,
        next_scores=next_scores,
    )
    # print(
    #     f"prev_state: {prev_state}, prev_scores: {prev_scores}, prev_action: {prev_action}, prev_dice: {prev_dice}, next_state: {next_state}, next_scores: {next_scores}, game_over: {game_over}, winner: {winner}, reward: {reward}"
    # )

    player.agent.learn(
        prev_state=prev_state,
        action=prev_action,
        reward=reward,
        next_state=next_state,
        game_over=game_over,
        winner=winner,
    )


def save_list(list, save_path):
    """Save the List to a file."""
    with open(save_path, "wb") as file:
        pickle.dump(list, file)
