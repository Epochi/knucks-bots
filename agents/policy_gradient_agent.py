import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from agents.base_agent_v2 import AbstractAgent
import game.player_actions_v2 as pa

# Define the device: Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        state_size,
        action_size=3,
        nickname="Policy Master",
        should_save_model=True,
        learning_rate=0.001,
    ):
        super().__init__(nickname, should_save_model)
        self.modelType = "PG"
        self.state_size = state_size
        self.action_size = action_size

        self.model = PolicyNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)

    def select_move(self, game_engine):
        available_moves = pa.get_available_moves(game_engine)
        state = self.convert_state(
            pa.get_board_state(game_engine), pa.get_dice_value(game_engine)
        )
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = self.model(state)

        probabilities = probabilities.cpu().numpy().squeeze()

        available_probabilities = probabilities[available_moves]
        available_probabilities /= available_probabilities.sum()

        action = np.random.choice(available_moves, p=available_probabilities)
        return action

    def learn(self, prev_state, action, reward, new_states, game_over, winner=None):
        self.memory.append(
            (
                prev_state,
                action,
                reward,
                True if winner is not None else False,
            )
        )
        if True if winner is not None else False:
            states, actions, rewards, dones = zip(*self.memory)
            self.optimize_model(states, actions, rewards)
            self.memory.clear()

    def optimize_model(self, states, actions, rewards):
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).view(-1, 1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)

        # Calculate the loss
        log_probs = torch.log(self.model(states))
        selected_log_probs = rewards * log_probs.gather(1, actions).squeeze()
        loss = -selected_log_probs.mean()

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def convert_state(self, board_state, dice_value):
        state = np.array(board_state).flatten().tolist() + [dice_value]
        return state
