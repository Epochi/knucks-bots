import agents.simple_q_learning_v2 as sq
import game.player_actions_v2 as pa


class SimpleQWinReinforcementAgent(sq.QLearningAgent):
    def __init__(
        self,
        nickname="Cell, the Brain Cell",
        should_save_model=True,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
    ):
        super().__init__(
            nickname,
            should_save_model,
            learning_rate,
            discount_factor,
            exploration_rate,
            exploration_decay,
            min_exploration_rate,
        )
        self.memory = {}

    def memorize(self, prev_state, values):
        """store prev states in memory"""
        if prev_state not in self.memory:
            self.memory[prev_state] = values

    def undo_memories(self):
        """undo all state changes"""
        for state, values in self.memory.items():
            self.model[state] = values
        self.memory = {}

    def learn(
        self,
        prev_state: str,
        action: tuple,
        reward: int,
        new_states: list,
        game_over=False,
        winner=None,
    ):
        super().learn(prev_state, action, reward, new_states, game_over, winner)

        self.memorize(prev_state, self.model[prev_state])

        if game_over and not winner:
            self.undo_memories()
