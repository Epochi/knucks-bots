import training.agent_trainer as agent_trainer
import training.reward_models_v2 as rm
from agents.random_agent_v2 import RandomAgent
from agents.simple_q_learning_v2 import QLearningAgent
from agents.deep_q_learning import DeepQLearningAgent
from agents.simple_q_win_reinforcment import SimpleQWinReinforcementAgent
from utils.play_game import PlayingAgent, GameRules


# python -c 'from training import trainer_runner; trainer_runner.train_simple_vs_random()'
# Quickly Learns, quickly forgets Wins: 48,944,442, Wild Card Wins: 48,948,708, Draws: 2,096,851
def train_simple_vs_random():
    player_1 = PlayingAgent(
        QLearningAgent(
            nickname="Quickly Learns, quickly forgets",
            learning_rate=0.2,
            discount_factor=0.95,
            exploration_rate=1.0,
            exploration_decay=0.9999,
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
# Training completed. Doesn't get out from the bed for less than 10k Wins: 48,951,522, Wild Card Wins: 48,949,752, Draws: 2,098,726
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
# The Brain that thinks that it plays with itself Wins: 48,947,680, Wild Card Wins: 48,944,365, Draws: 2,097,956
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


# python -c 'from training import trainer_runner; trainer_runner.deep_q_full_game_vs_random()'
# Training completed. AlphaMinusOne Wins: 24,606,330, Wild Card Wins: 24,603,382, Draws: 790,288
def deep_q_full_game_vs_random():
    player_1 = PlayingAgent(
        DeepQLearningAgent(
            state_size=63 + 63 + 6,
            nickname="AlphaMinusOne",
        ),
        rm.calculate_for_multiples_and_removals_score,
        "deep_q_full_game_vs_random",
    )
    player_2 = PlayingAgent(RandomAgent(), None)
    game_rules = GameRules(should_remove_opponents_dice=True)
    agent_trainer.train_agents(
        player_1, player_2, game_rules, episodes=50 * 1000 * 1000
    )


# python -c 'from training import trainer_runner; trainer_runner.deep_q_full_game_vs_random_double_state()'
def deep_q_full_game_vs_random_double_state():
    player_1 = PlayingAgent(
        DeepQLearningAgent(
            state_size=(63 + 63 + 6) * 2,
            nickname="Two Brain Cells",
        ),
        rm.calculate_for_multiples_and_removals_score,
        "deep_q_vs_random_double_state",
    )
    player_2 = PlayingAgent(RandomAgent(), None)
    game_rules = GameRules(should_remove_opponents_dice=True)
    agent_trainer.train_agents(
        player_1, player_2, game_rules, episodes=50 * 1000 * 1000
    )


# python -c 'from training import trainer_runner; trainer_runner.deep_q_tuned_vs_random()'
def deep_q_tuned_vs_random():
    parametrized_reward_model = rm.ParametrizedRewardModel(
        reward_loss_amount=-100,
        reward_win_amount=100,
        reward_score_diff_multiplier=0.01,
        reward_score_incerease_multiplier_multiplier=0.001,
        reward_opponent_score_decrease_multiplier_multiplier=0.001,
    )
    player_1 = PlayingAgent(
        DeepQLearningAgent(
            learning_rate=0.01,
            discount_factor=0.99,
            exploration_rate=1.0,
            exploration_decay=0.99,
            min_exploration_rate=0.1,
            batch_size=64,
            memory_size=5000,
            target_update=10,
            state_size=63 + 63 + 6,
            nickname="AlphaNegativeOne",
        ),
        parametrized_reward_model.calculate_reward,
        # "deep_q_tuned_vs_random",
    )
    player_2 = PlayingAgent(RandomAgent(), None)
    game_rules = GameRules(should_remove_opponents_dice=True)
    agent_trainer.train_agents(player_1, player_2, game_rules, episodes=1 * 1000 * 1000)


# python -c 'from training import trainer_runner; trainer_runner.deep_q_full_game_vs_random_qucik_learner()'
def deep_q_full_game_vs_random_qucik_learner():
    player_1 = PlayingAgent(
        DeepQLearningAgent(
            state_size=(63 + 63 + 6),
            nickname="Only need two brain cells if they're quick",
            exploration_rate=1.0,
            exploration_decay=0.9999,
            min_exploration_rate=0.02,
            batch_size=32,
            memory_size=5000,
            learning_rate=0.1,
            discount_factor=0.95,
            target_update=50,
        ),
        rm.calculate_for_multiples_and_removals_score,
        "deep_q_full_game_vs_random_qucik_learner",
    )
    player_2 = PlayingAgent(RandomAgent(), None)
    game_rules = GameRules(should_remove_opponents_dice=True)
    agent_trainer.train_agents(
        player_1, player_2, game_rules, episodes=50 * 1000 * 1000
    )


# python -c 'from training import trainer_runner; trainer_runner.simple_high_explore_vs_random_no_removal_reward_multiply_only()'
# Dora the explorer Wins: 48,944,533, Wild Card Wins: 48,947,809, Draws: 2,097,659
def simple_high_explore_vs_random_no_removal_reward_multiply_only():
    player_1 = PlayingAgent(
        QLearningAgent(
            nickname="Dora the explorer",
            learning_rate=0.3,
            discount_factor=0.95,
            exploration_rate=1.0,
            exploration_decay=0.9999,
            min_exploration_rate=0.25,
        ),
        rm.one_side_for_multiply_and_win_only,
        "simple_high_explore_vs_random_no_removal_reward_multiply_only",
    )
    player_2 = PlayingAgent(RandomAgent(), None)
    game_rules = GameRules(max_dice_value=6, should_remove_opponents_dice=False)
    agent_trainer.train_agents(
        player_1, player_2, game_rules, episodes=100 * 1000 * 1000
    )


# python -c 'from training import trainer_runner; trainer_runner.train_simple_vs_random_full_game_multiply_only()'
# Simple, but Brave Wins: 4,987,622, Wild Card Wins: 4,845,584, Draws: 156,795
def train_simple_vs_random_full_game_multiply_only():
    parametrized_reward_model = rm.ParametrizedRewardModel(
        reward_loss_amount=-100,
        reward_win_amount=100,
        reward_score_diff_multiplier=0.1,
        reward_score_incerease_multiplier_multiplier=0.1,
        reward_opponent_score_decrease_multiplier_multiplier=0.01,
    )
    player_1 = PlayingAgent(
        QLearningAgent(
            nickname="Simple, but Brave",
            min_exploration_rate=0.001,
            exploration_decay=0.9995,
            learning_rate=0.01,
            discount_factor=0.99,
        ),
        parametrized_reward_model.calculate_reward,
        # "train_simple_vs_random_full_game_multiply_only",
    )
    player_2 = PlayingAgent(RandomAgent(), None)
    game_rules = GameRules(should_remove_opponents_dice=True)
    agent_trainer.train_agents(player_1, player_2, game_rules, episodes=1 * 1000 * 1000)


#  python -c 'from training import trainer_runner; trainer_runner.train_simple_selective_memory_vs_random()'
def train_simple_selective_memory_vs_random():
    parametrized_reward_model = rm.ParametrizedRewardModel(
        reward_win_amount=100,
        reward_score_increase_multiplier=0.1,
        reward_score_diff_multiplier=0.1,
        reward_score_incerease_multiplier_multiplier=0.1,
        reward_opponent_score_decrease_multiplier_multiplier=0.01,
    )
    player_1 = PlayingAgent(
        SimpleQWinReinforcementAgent(
            nickname="Selective Memory",
            min_exploration_rate=0.05,
            exploration_decay=0.9995,
            learning_rate=0.01,
            discount_factor=0.99,
        ),
        parametrized_reward_model.calculate_reward,
        # "train_simple_vs_random_full_game_multiply_only",
    )
    player_2 = PlayingAgent(RandomAgent(), None)
    game_rules = GameRules(should_remove_opponents_dice=True)
    agent_trainer.train_agents(
        player_1, player_2, game_rules, episodes=10 * 1000 * 1000
    )


# python -c 'from training import trainer_runner; trainer_runner.train_simple_vs_deep_q_full_game_multiply_only()'
def train_simple_vs_deep_q_full_game_multiply_only():
    player_1 = PlayingAgent(
        QLearningAgent(
            nickname="Simple, but Brave",
            min_exploration_rate=0.05,
            exploration_decay=0.9999,
            should_save_model=False,
        ),
        rm.calculate_for_multiples_and_removals_score,
        model_name="train_simple_vs_random_full_game_multiply_only",
    )
    player_2 = PlayingAgent(
        DeepQLearningAgent(
            state_size=63 + 63 + 6,
            nickname="AlphaMinusOne",
            should_save_model=False,
        ),
        None,
        model_name="deep_q_full_game_vs_random",
    )
    game_rules = GameRules(should_remove_opponents_dice=True)
    agent_trainer.train_agents(
        player_1, player_2, game_rules, episodes=10 * 1000 * 1000
    )


# python -c 'from training import trainer_runner; trainer_runner.random_vs_random()'
def random_vs_random():
    # this is a control group to see if the agents are learning anything
    # Results from 100M episodes: Training completed. Wild Card Wins: 48,959,336, Mad Contender Wins: 48,942,009, Draws: 2,098,655
    player_1 = PlayingAgent(RandomAgent(), None)
    player_2 = PlayingAgent(RandomAgent("Mad Contender"), None)
    game_rules = GameRules(max_dice_value=6, should_remove_opponents_dice=False)
    agent_trainer.train_agents(
        player_1, player_2, game_rules, episodes=100 * 1000 * 1000
    )
