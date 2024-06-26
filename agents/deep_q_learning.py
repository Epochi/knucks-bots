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
        self.fc1 = nn.Linear(state_size, 128)  # First fully connected layer
        self.relu1 = nn.ReLU()  # ReLU activation for non-linearity
        self.dropout1 = nn.Dropout(p=0.2)  # Dropout for regularization

        self.fc2 = nn.Linear(128, 128)  # Second fully connected layer
        self.relu2 = nn.ReLU()  # ReLU activation for non-linearity
        self.dropout2 = nn.Dropout(p=0.2)  # Additional dropout layer

        self.fc3 = nn.Linear(128, action_size)  # Output layer

    def forward(self, x):
        x = self.fc1(x)  # Pass input through the first layer
        x = self.relu1(x)  # Apply ReLU activation
        x = self.dropout1(x)  # Apply dropout

        x = self.fc2(x)  # Pass through the second layer
        x = self.relu2(x)  # Apply ReLU activation
        x = self.dropout2(x)  # Apply dropout

        x = self.fc3(x)  # Output layer

        return x


class DeepQLearningAgent(AbstractAgent):
    def __init__(
        self,
        state_size=19,
        action_size=3,
        nickname="The Brain",
        should_save_model=True,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        memory_size=10000,
        batch_size=64,
        target_update=10,
        device="cuda",
    ):
        super().__init__(nickname, should_save_model, device)
        self.modelType = "DQ"
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

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_move(self, game_engine):
        available_moves = pa.get_available_moves(game_engine)
        state = self.convert_state(
            pa.get_board_state(game_engine), pa.get_dice_value(game_engine)
        )
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(available_moves)
        else:
            # Exploitation: Select the action with the highest predicted Q-value from the available moves
            state = (
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
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

    def learn(
        self,
        prev_state: str,
        action: tuple,
        reward: int,
        next_state: str,
        game_over: bool,
        winner=None,
    ):
        if len(self.memory) < self.batch_size:
            self.memorize_all_possible_transitions(
                prev_state,
                action,
                reward,
                next_state,
                game_over,
            )
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_state, dones = zip(*mini_batch)

        states = torch.FloatTensor(states).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        normalized_rewards = self.z_score_normalize_rewards(rewards)
        dones = torch.FloatTensor(dones).to(self.device)

        curr_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_model(next_state).max(1)[0].detach()
        next_q_values = next_q_values.mean()
        expected_q_values = normalized_rewards + (
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

    def memorize_all_possible_transitions(
        self, state, action, reward, next_state, done
    ):

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

    def z_score_normalize_rewards(self, rewards):
        """
        Normalizes rewards using Z-score normalization using PyTorch.

        Parameters:
        - rewards: A PyTorch tensor of rewards.

        Returns:
        - normalized_rewards: A tensor of Z-score normalized rewards.
        """
        mean_reward = torch.mean(rewards)
        std_deviation = torch.std(rewards)

        # Avoid division by zero in case all rewards are the same
        # std_deviation might be 0 if all rewards are the same, add a small epsilon to avoid division by zero
        normalized_rewards = (rewards - mean_reward) / (std_deviation + 1e-8)

        return normalized_rewards

    def min_max_normalize_rewards(self, rewards):
        """
        Normalizes rewards using Min-Max normalization using PyTorch.

        Parameters:
        - rewards: A PyTorch tensor of rewards.

        Returns:
        - normalized_rewards: A tensor of Min-Max normalized rewards.
        """
        min_reward = torch.min(rewards)
        max_reward = torch.max(rewards)

        # The range might be 0 if all rewards are the same, add a small epsilon to avoid division by zero
        normalized_rewards = (rewards - min_reward) / (max_reward - min_reward + 1e-8)

        return normalized_rewards
