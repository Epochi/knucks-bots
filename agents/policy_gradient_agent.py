import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from agents.base_agent_v2 import AbstractAgent
import game.player_actions_v2 as pa


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size=3):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


class PolicyGradientAgent(AbstractAgent):
    def __init__(
        self,
        state_size=19,
        action_size=3,
        entropy=0.01,
        nickname="Policy Master",
        should_save_model=True,
        learning_rate=0.001,
        device="cuda",
    ):
        super().__init__(nickname, should_save_model, device)
        self.modelType = "PG"
        self.state_size = state_size
        self.action_size = action_size
        self.entropy = entropy
        self.model = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.counter = 0

    def select_move(self, game_engine):
        available_moves = pa.get_available_moves(game_engine)
        state = self.convert_state(
            pa.get_board_state(game_engine), pa.get_dice_value(game_engine)
        )
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = self.model(state)

        probabilities = probabilities.cpu().numpy().squeeze()

        available_probabilities = probabilities[available_moves]
        available_probabilities /= available_probabilities.sum()
        if np.isnan(available_probabilities).any():
            print(f"State: {state}")
            print(f"board state: {pa.get_board_state(game_engine)}")
            print(f"Available moves: {available_moves}")
            print(f"Probabilities: {probabilities}")
            print(
                "🐍 File: agents/policy_gradient_agent.py | Line: 55 | select_move ~ self.model(state)",
                self.model(state),
            )
            print(f"couter: {self.counter}")
        self.counter += 1
        action = np.random.choice(available_moves, p=available_probabilities)
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
        self.memory.append(
            (
                prev_state,
                action,
                reward,
                game_over,
            )
        )
        if game_over:
            states, actions, rewards, dones = zip(*self.memory)
            self.optimize_model(states, actions, rewards)
            self.memory.clear()

    def optimize_model(self, states, actions, rewards):
        # torch.autograd.set_detect_anomaly(True)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        # Calculate the loss
        log_probs = torch.log(self.model(states))
        entropy = (
            -(log_probs * log_probs.exp()).sum(dim=1).mean()
        )  # Entropy of the policy
        normalized_rewards = self.min_max_normalize_rewards(rewards)
        selected_log_probs = log_probs.gather(1, actions).squeeze(1)
        loss = (
            -(selected_log_probs * normalized_rewards).mean() - self.entropy * entropy
        )

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def convert_state(self, board_state, dice_value):
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
