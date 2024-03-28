import training.agent_trainer as agent_trainer
import training.reward_models_v2 as rm
from agents.random_agent_v2 import RandomAgent
from agents.simple_q_learning_v2 import QLearningAgent
from agents.deep_q_learning import DeepQLearningAgent
from utils.play_game import PlayingAgent, GameRules


# python -c 'from training import trainer_runner; trainer_runner.train_simple_vs_random()'
def train_simple_vs_random():
    player_1 = PlayingAgent(
        QLearningAgent(
            nickname="Quickly Learns, quickly forgets",
            learning_rate=0.2,
            discount_factor=0.95,
            exploration_rate=1.0,
            exploration_decay=0.9999999999999,
            min_exploration_rate=0.1,
        ),
        rm.calculate_for_own_score_only,
        "simple_q_by_score_vs_random_game_no_removal",
    )
    player_2 = PlayingAgent(RandomAgent(), None)
    game_rules = GameRules(max_dice_value=6, should_remove_opponents_dice=False)
    agent_trainer.train_agents(
        player_1, player_2, game_rules, episodes=100 * 1000 * 1000
    )


# python -c 'from training import trainer_runner; trainer_runner.train_simple_vs_random_multiply_only()'
def train_simple_vs_random_multiply_only():
    player_1 = PlayingAgent(
        QLearningAgent(nickname="Doesn't get out from the bed for less than 10k"),
        rm.one_side_for_multiply_and_win_only,
        "simple_q_by_score_vs_random_game_reward_for_multiply_only",
    )
    player_2 = PlayingAgent(RandomAgent(), None)
    game_rules = GameRules(max_dice_value=6, should_remove_opponents_dice=False)
    agent_trainer.train_agents(
        player_1, player_2, game_rules, episodes=100 * 1000 * 1000
    )


# python -c 'from training import trainer_runner; trainer_runner.deep_q_vs_random()'
def deep_q_vs_random():
    player_1 = PlayingAgent(
        DeepQLearningAgent(
            state_size=63 + 6,
            nickname="The Brain that thinks that it plays with itself",
        ),
        rm.one_side_for_multiply_and_win_only,
        "deep_q_by_score_vs_random_game_no_removal",
    )
    player_2 = PlayingAgent(RandomAgent(), None)
    game_rules = GameRules(max_dice_value=6, should_remove_opponents_dice=False)
    agent_trainer.train_agents(
        player_1, player_2, game_rules, episodes=100 * 1000 * 1000
    )


# python -c 'from training import trainer_runner; trainer_runner.random_vs_random()'
def random_vs_random():
    # this is a control group to see if the agents are learning anything
    player_1 = PlayingAgent(RandomAgent(), None)
    player_2 = PlayingAgent(RandomAgent("Mad Contender"), None)
    game_rules = GameRules(max_dice_value=6, should_remove_opponents_dice=False)
    agent_trainer.train_agents(
        player_1, player_2, game_rules, episodes=100 * 1000 * 1000
    )
