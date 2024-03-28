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

To train models go to training/trainer_runner.py and run one of the models

## Available configurations

utils.play_game.GameRules exposes all available rule configurations. i.e. turning off the rule to remove opponent dice makes the game significantly simpler and in turn makes it quicker to train a model. Or using lower max value for dice can make for a smaller state, but it also makes the game more random, leaving choices less meaningful


## V1 and V2

V1 was the initial implementation with code that's more human readable and represented the game state literally.

V2 is optimized game code that applies rules in the same way, but represents state differently(columns and rows are switched around), but because we are dealing with really really really really really tiny state size, there are some rudamentary optimizations like going using if statements to loop through an array instead of array loop. 