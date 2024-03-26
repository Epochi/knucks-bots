# Overview

In this project different agents play board game knucklebones.

The agents tested here are:
1. Random
2. Q-Learning
3. Deep Q-Learning (Not implemented)
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
* /agents - have RL agents that are able to play the game
* /training - have scripts that play the game using the agents to train model. The training scripts can use different modifiers that the agents accept
* /models - models of the trained agents
* /notebooks - jupyter notebooks comparing different model win rates
* /tests - unit tests
* /play - files for manual play against one of the models

