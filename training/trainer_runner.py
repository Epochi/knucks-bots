import training.agent_trainer as agent_trainer
import training.reward_models_v2 as rm
from agents.random_agent_v2 import RandomAgent
from agents.simple_q_learning_v2 import QLearningAgent


def train_simple_vs_random():
    player_1 = agent_trainer.TrainingAgent(
        QLearningAgent,
        rm.calculate_for_own_score_only,
        "simple_q_by_score_vs_random_game_no_removal",
    )
    player_2 = agent_trainer.TrainingAgent(RandomAgent, None)
    game_rules = agent_trainer.TrainingGameRules(
        max_dice_value=6, should_remove_opponents_dice=False
    )
    agent_trainer.train_q_learning_agent(
        player_1, player_2, game_rules, episodes=100 * 1000 * 1000
    )


# python training/trainer_runner.py

train_simple_vs_random()
