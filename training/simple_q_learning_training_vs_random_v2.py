"""Train a Q-Learning agent against a random agent"""

import os
import pickle
import time
import atexit
import objgraph
import game.player_actions_v2 as pa
import training.reward_models as rm
from agents.simple_q_learning_v2 import QLearningAgent
from agents.simple_q_learning_state_optimized import QLearningAgentSpaceOptimized

from agents.random_agent import RandomAgent


def train_q_learning_agent(
    agent=None,
    reward_func=None,
    episodes=1000,
    save_path="./models/q_learning_model.pkl",
    resume_model_from_path=None,
):
    """train Q-Learning agent against a random agent"""
    q_agent = agent(resume_model_from_path)
    random_agent = RandomAgent()
    wins = 0
    losses = 0
    draws = 0

    heartbeat = min(10 / 100 * episodes, 10000)
    perf_timer = None
    file_size_limit = 10000000000  # around 10GB
    atexit.register(lambda: save_q_table(q_agent.q_table, save_path))

    for episode in range(episodes):
        game_engine = pa.start_game()

        is_heartbeat = episode % heartbeat == 0

        while not pa.get_game_over(game_engine):
            pa.start_turn(game_engine)
            dice_value = pa.get_dice_value(game_engine)

            # agent always as player 1
            if pa.get_current_player(game_engine) == 0:
                pre_move_state = q_agent.convert_state(
                    pa.get_board_state(game_engine), dice_value
                )
                pre_move_scores = pa.get_score(game_engine)
                # Q-Learning Agent makes a move
                action = q_agent.select_move(game_engine)
                pa.do_move(game_engine, action)
                post_move_state = q_agent.convert_state(
                    pa.get_board_state(game_engine), 0
                )
                post_move_score = pa.get_score(game_engine)

                reward = reward_func(
                    game_engine,
                    pre_move_state,
                    pre_move_scores,
                    action,
                    post_move_state,
                    post_move_score,
                )

                q_agent.update_q_table(pre_move_state, action, reward, post_move_state)
            else:
                # Random Agent makes a move
                action = random_agent.select_move(game_engine)
                pa.do_move(game_engine, action)

            pa.end_turn(game_engine)

        # Check game outcome
        if pa.get_game_over(game_engine):
            winner = pa.get_winner(game_engine)
            if winner == 0:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1

        # sanity check, print game every 10% of the episodes, but not more rarely than 1000
        if is_heartbeat:
            print(f"Episode {episode}/{episodes}")

            print("\nPerformance stats:")
            if perf_timer:
                print(
                    f"Time taken for {heartbeat} episodes: {time.time() - perf_timer}"
                )
            objgraph.show_most_common_types()
            perf_timer = time.time()
            print(f"keys in q_table: {len(q_agent.q_table)}")

            print("\nMotivation Stats:")
            print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
            print(
                "We just won!" if pa.get_winner(game_engine) == 0 else "We just lost!"
            )
            print(
                f"Scores: Player 1: {pa.get_score(game_engine)[0]}, Player 2: {pa.get_score(game_engine)[1]}"
            )
            pa.display_board(game_engine)

            if os.path.exists(save_path) & os.path.getsize(save_path) > file_size_limit:
                print(f"Q-Table size exceeds {file_size_limit}, saving to {save_path}")
                # exit early
                break

    save_q_table(q_agent.q_table, save_path)

    print(f"Training completed. Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Model saved to {save_path}")


def save_q_table(q_table, save_path):
    """Save the Q-Table to a file."""
    with open(save_path, "wb") as file:
        pickle.dump(q_table, file)


# python training/simple_q_learning_training_vs_random.py

if __name__ == "__main__":
    train_q_learning_agent(
        QLearningAgent,
        rm.calculate_for_score,
        episodes=20 * 1000 * 1000,
        save_path="./models/q_learning_model_by_score.pkl",
        resume_model_from_path="./models/q_learning_model_by_score.pkl",
    )
# if __name__ == "__main__":
#     train_q_learning_agent(
#         QLearningAgentSpaceOptimized,
#         rm.calculate_for_score,
#         episodes=20 * 1000 * 1000,
#         save_path="./models/q_learning_model_by_score_space_optmized.pkl",
#         resume_model_from_path="./models/q_learning_model_by_score_space_optmized.pkl",
#     )
