"""Train a Q-Learning agent against a random agent"""

import pickle
import time
import game.player_actions as pa
import training.reward_models as rm
from agents.simple_q_learning import QLearningAgent
from agents.random_agent import RandomAgent


def train_q_learning_agent(
    reward_func=None,
    episodes=1000,
    save_path="./models/q_learning_model.pkl",
    resume_model_from_path=None,
):
    """train Q-Learning agent against a random agent"""
    q_agent = QLearningAgent(resume_model_from_path)
    random_agent = RandomAgent()
    wins = 0
    losses = 0
    draws = 0

    perf_timer = None

    for episode in range(episodes):
        game_engine = pa.start_game()

        if episode % 1000 == 0:
            if perf_timer:
                print(f"Time taken for 1000 episodes: {time.time() - perf_timer}")
            perf_timer = time.time()

        while not pa.get_game_over(game_engine):
            pa.start_turn(game_engine)
            dice_value = pa.get_dice_value(game_engine)

            # agent always as player 1
            if pa.get_current_player(game_engine) == 1:
                pre_move_state = q_agent.convert_state(
                    pa.get_board_state(game_engine), dice_value
                )
                pre_move_scores = pa.get_score(game_engine)
                # Q-Learning Agent makes a move
                action = q_agent.select_move(game_engine)
                pa.do_move(game_engine, *action)
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
                pa.do_move(game_engine, *action)

            pa.end_turn(game_engine)

        # Check game outcome
        if pa.get_game_over(game_engine):
            winner = pa.get_winner(game_engine)
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1

        # sanity check, print game every 10% of the episodes, but not more rarely than 1000
        if episode % min(episodes // 10, 1000) == 0:
            print(f"Episode {episode}/{episodes}")
            print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

        # Save the trained Q-Table every 1000 games
        # (Nightly run security)
        if episode % 1000 == 0:
            with open(f"{save_path}", "wb") as f:
                pickle.dump(q_agent.q_table, f)

    # Save the trained Q-Table
    with open(save_path, "wb") as f:
        pickle.dump(q_agent.q_table, f)

    print(f"Training completed. Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_q_learning_agent(
        rm.calculate_for_score,
        episodes=100000,
        save_path="./models/q_learning_model_by_score.pkl",
        resume_model_from_path="./models/q_learning_model_by_score.pkl",
    )
