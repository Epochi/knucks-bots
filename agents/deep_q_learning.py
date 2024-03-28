# Deep Q-Learning Agent Module(Compatible with the Game Engine V2)"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from agents.base_agent_v2 import AbstractAgent
import game.player_actions_v2 as pa


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Define the architecture of the network
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class DeepQLearningAgent(AbstractAgent):
    def __init__(
        self,
        state_size,
        action_size=3,
        nickname="The Brain",
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        memory_size=10000,
        batch_size=64,
        target_update=10,
    ):
        super().__init__(nickname)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.discount_factor = discount_factor
        self.target_update = target_update
        self.update_count = 0

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_move(self, game_engine):
        available_moves = pa.get_available_moves(game_engine)
        state = pa.get_board_state(game_engine)
        if random.uniform(0, 1) < self.exploration_rate:
            # Exploration: Randomly select from available moves
            action = random.choice(available_moves)
        else:
            # Exploitation: Select the action with the highest predicted Q-value from the available moves
            state = torch.FloatTensor(state).unsqueeze(
                0
            )  # Convert state to tensor and add batch dimension
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                action_values = self.model(state).squeeze(
                    0
                )  # Predict Q-values for all actions and remove batch dimension
            self.model.train()  # Set the model back to train mode

            # Filter the Q-values for only those actions that are available
            q_values_of_available_moves = action_values[available_moves]
            # Select the action corresponding to the highest Q-value from the available moves
            max_q_value, max_q_index = torch.max(q_values_of_available_moves, dim=0)
            action = available_moves[
                max_q_index.item()
            ]  # Convert the index to the corresponding action

        return action

    def learn(self, prev_state: str, action: tuple, reward: int, new_states: list):
        if len(self.memory) < self.batch_size:
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Get the current Q-values
        curr_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the expected Q-values
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (
            self.discount_factor * next_q_values * (1 - dones)
        )

        # Compute the loss between the current Q-values and the expected Q-values
        loss = self.criterion(curr_q_values, expected_q_values)

        # Zero the parameter gradients
        self.optimizer.zero_grad()
        # Backpropagate the loss
        loss.backward()
        # Update the model weights
        self.optimizer.step()

        # Update the exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )

        # Update the target network, if needed
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.update_target_model()

    def memorize(self, state, action, reward, next_state, done):
        """
        Stores a transition in the agent's memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        """
        Copies the weights from the model to the target model.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    # Additional code to interact with your specific environment might be needed here.

    def convert_state(self, board_state, dice_value):
        """
        Converts the current board state and dice value into a format suitable for the DQN.
        """
        state = np.array(board_state).flatten().tolist() + [dice_value]
        return state
