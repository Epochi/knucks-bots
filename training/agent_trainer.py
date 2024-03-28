"""Utility to train agents against each other."""

import signal
import pickle
import time
import game.player_actions_v2 as pa
from utils.play_game import PlayingAgent, GameRules, player_move

interrupted = False


# Signal handler function
def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("CTRL+C detected. Finishing the current episode...")


signal.signal(signal.SIGINT, signal_handler)


def train_q_learning_agent(
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
    perf_timer = None
    global interrupted

    for episode in range(episodes):
        game_engine = pa.start_game(
            enable_print=False,
            max_dice_value=game_rules.max_dice_value,
            should_remove_opponents_dice=game_rules.should_remove_opponents_dice,
            safe_mode=False,
        )

        is_heartbeat = episode % heartbeat == 0

        while not pa.get_game_over(game_engine):
            current_player = (
                player_1 if pa.get_current_player(game_engine) == 0 else player_2
            )

            player_move(game_engine, current_player, game_rules)
        # Check game outcome
        if pa.get_game_over(game_engine):
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

        # sanity check, print game every 10% of the episodes, but not more rarely than 1000
        if is_heartbeat:
            print(f"Episode {episode:,}/{episodes:,}")

            print("\nPerformance stats:")
            if perf_timer:
                print(
                    f"Time taken for {heartbeat} episodes: {time.time() - perf_timer}"
                )
            # objgraph.show_most_common_types()
            perf_timer = time.time()
            if (
                hasattr(player_1.agent, "model")
                and isinstance(player_1.agent.model, dict)
                and len(player_1.agent.model) > 0
            ):
                print(
                    f"keys in {player_1.agent.nickname} model: {len(player_1.agent.model)}"
                )
            if (
                hasattr(player_2.agent, "model")
                and isinstance(player_2.agent.model, dict)
                and len(player_1.agent.model) > 0
            ):
                print(
                    f"keys in {player_2.agent.nickname} model: {len(player_2.agent.model)}"
                )

            print("\nMotivation Stats:")
            print(f"Wins: {wins:,}, Losses: {losses:,}, Draws: {draws:,}")
            print(
                "We just won!" if pa.get_winner(game_engine) == 0 else "We just lost!"
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

        if interrupted:
            print("Stop requested. Exiting training.")
            break

    print(
        f"Total time taken: {time.time() - perf_timer_total_run} for {episodes:,} episodes"
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

    print(f"Training completed. Wins: {wins}, Losses: {losses}, Draws: {draws}")


def save_list(list, save_path):
    """Save the List to a file."""
    with open(save_path, "wb") as file:
        pickle.dump(list, file)
