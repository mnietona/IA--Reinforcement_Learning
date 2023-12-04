from rlenv import RLEnv, Observation
import numpy as np
import random

class QLearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=5):
        self.q_table = {}  # Dictionnaire pour les valeurs Q
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  # Taux d'exploration
        self.n_actions = n_actions

    def choose_action(self, observation: Observation):
        state = observation.state
        state_key = self._get_state_key(state)

        # On suppose que `available_actions` est un tableau binaire de la forme (n_actions,)
        # où 1 indique que l'action est disponible et 0 sinon.
        available_actions = observation.available_actions[0]
        available_actions_indices = np.where(available_actions > 0)[0]

        if state_key not in self.q_table:
            # Initialiser avec des valeurs de 1 pour encourager l'exploration
            self.q_table[state_key] = np.ones(self.n_actions)

        if random.uniform(0, 1) < self.epsilon:  # Exploration
            action = np.random.choice(available_actions_indices)
        else:  # Exploitation
            q_values = self.q_table[state_key][available_actions_indices]
            best_action_index = np.argmax(q_values)
            action = available_actions_indices[best_action_index]

        return action

    def update(self, observation, action, reward, next_observation):
        state = observation.state
        next_state = next_observation.state

        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.ones(self.n_actions)

        # Mise à jour de Q
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

        self.q_table[state_key][action] = new_value

    def _get_state_key(self, state):
        """ Génère une clé unique pour l'état donné pour être utilisée dans la table Q. """
        return hash(state.tobytes())

    def print_q_table(self):
        for state_key, values in self.q_table.items():
            print(f"State {state_key}: {values}")
        print(f"Q-table size: {len(self.q_table)}")
