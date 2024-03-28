# Overview

In this project different agents play board game knucklebones.

The agents tested here are:
1. Random
2. Q-Learning
3. Deep Q-Learning
4. Human (Not implemented)

# Game Overview

Knucklebones is a two player dice game from the videya game Cult of the Lamb.

# Game Rules

* Game board consists of two symetrical 3x3 grids
* A player is randomly assigned if they go first or second
* Player rolls the dice and then puts the dice in any spot on their side of the board.
* If there is any number of dice of the same value on the same column in opponent side of the board, the opponents dice with the same value on the same column are removed from the game.
* Game ends when one of the players board is filled and they no longer have where to put dice.
* Player with the higher score wins

## Scoring:

* Player score is sum of all of the die values on their side of the board
* If there are two dice of the same value on the same column in player side of the board, then the score for those die is squared. I.e. 3x3 will give 9 score instead of 6
* If there are three dice of the same value on the same column in player side of the board, then the score for those die are cubed. I.e. 3x3x3 will give 27 instead of 9
* If player dice is removed from the game by the other player, then the score is also removed


# Project Structure

* /game - files for controlling the game
* /agents - reinforment learning agents that are able to play the game
* /training - scripts for training the agents versus eachother and saving models
* /models - models of the trained agents
* /notebooks - jupyter notebooks comparing different model win rates
* /tests - unit tests
* /utils - shared utilty functions for the above

# Running the project

Requires pyhon 3.11.
I recommend using .venv and setting export PYTHONPATH="$PYTHONPATH:full-path-to-directory/knucks-bots" in activate file


## Running training

To train models go to training/trainer_runner.py and run one of the training configurations defined. 

i.e. 
```bash
python -c 'from training import trainer_runner; trainer_runner.deep_q_vs_random()'
```

will run a game with the following configuration

```python
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
```

## Training Monitoring

So we don't get bored and can monitor progress of our training, we run a hearbreat every 10% of games up to every 10,000 games. The heartbeat prints the state of training and representation of the board of the last game:

```bash
Episode 2,310,000/100,000,000

Performance stats:
Time taken for 10000 episodes: 3.884371042251587
keys in Quickly Learns, quickly forgets model: 2432283
keys in Wild Card model: 0

Motivational Stats:
Quickly Learns, quickly forgets Wins: 1,131,027, Wild Card Wins: 1,130,594, Draws: 48,380
We just lost!
Scores: Quickly Learns, quickly forgets: 28,  Wild Card: 36

Learning Stats:
learning rate: 0.2
exploration rate: 0.4318743772542243


| 1 | 1 | 1 |
| 4 | 2 | 3 |
| 5 | 6 | 5 |
|---|---|---|
| 2 | 2 |   |
| 4 | 4 | 3 |
| 4 | 5 | 4 |
```

## Available configurations

utils.play_game.GameRules exposes all available rule configurations. i.e. turning off the rule to remove opponent dice makes the game significantly simpler and in turn makes it quicker to train a model. Or using lower max value for dice can make for a smaller state, but it also makes the game more random, leaving choices less meaningful

The agent_trainer.train_agents accepts write_result_history: bool argument for exporting training progress to a a simple list that can be imported to jupyter notebook for visual representation


## V1 and V2

V1 was the initial implementation with code that's more human readable and represented the game state literally.

V2 is optimized game code that applies rules in the same way, but represents state differently(columns and rows are switched around), but because we are dealing with really really really really really tiny state size, there are some rudamentary optimizations like going using if statements to loop through an array instead of array loop. 

V1 will be deleted at some point


