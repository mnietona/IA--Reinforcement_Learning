from rlenv import RLEnv, Observation
import numpy as np
import numpy.typing as npt
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
        state_key = hash(state.tobytes())

        available_actions = observation.available_actions[0]
        available_actions_indices = np.where(available_actions > 0)[0]

        if state_key not in self.q_table:
            self.q_table[state_key] = np.ones(self.n_actions)

        if random.uniform(0, 1) < self.epsilon:  # Exploration
            action = np.random.choice(available_actions_indices)
            return action
        else:  # Exploitation
            q_values = self.q_table[state_key]
            q_values_filtered = q_values[available_actions_indices]
            best_action_index = np.argmax(q_values_filtered)
            return available_actions_indices[best_action_index]

    def update(self, observation, action, reward, next_observation):
        state = observation.state
        next_state = next_observation.state

        state_key = hash(state.tobytes())
        next_state_key = hash(next_state.tobytes())

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.ones(self.n_actions)

        # Update Q 
        current_q = self.q_table[state_key]
        next_q = self.q_table[next_state_key]
        old_value = current_q[action]
        next_max = np.max(next_q)
        print(f"old_value: {old_value}, next_max: {next_max}, reward: {reward}")
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

        current_q[action] = new_value
        self.q_table[state_key] = current_q


    def print_q_table(self):
        for state_key in self.q_table:
            print(f"State {state_key}: {self.q_table[state_key]}")
        
        print(f"Q-table size: {len(self.q_table)}")
