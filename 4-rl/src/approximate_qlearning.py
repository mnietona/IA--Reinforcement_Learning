from rlenv import RLEnv, Observation
from qlearning import QLearning
from features import feature_extraction
import numpy as np


class ApproximateQLearning(QLearning):
    def __init__(self, id, alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=5, n_features=4):
        super().__init__(id, alpha, gamma, epsilon, n_actions)
        self.weights = np.random.uniform(low=-0.01, high=0.01, size=(n_actions, n_features))
        
    def choose_action(self, observation: Observation) -> int:
        """ Choix d'une action """
        if self._should_explore():
            return self._explore(observation)
        else:
            return self._exploit(observation)
        
    def update(self, observation, action, reward, next_observation,done):
        """ Mise à jour des poids """
        state_features = feature_extraction(observation, self.id)
        next_state_features = feature_extraction(next_observation, self.id)

        q_values_next = self._get_q_values(next_state_features)
        q_value = self._get_q_values(state_features)[action]

        td_error = (reward + self.gamma * np.max(q_values_next)*(1-done)) - q_value
        td_error = np.clip(td_error, -1, 1)

        for i in range(len(state_features)):
            self.weights[action, i] += self.alpha * td_error * state_features[i]

        # Clipping des poids pour éviter les valeurs extrêmes
        self.weights = np.clip(self.weights, -10, 10)
        

    def _exploit(self, observation) -> int:
        """ Exploite en choisissant la meilleure action """
         # Etapes 1 : Extraction des caractéristiques de l'état du monde
        state_features = feature_extraction(observation, self.id)
        # Etape 2 : Calcul des valeurs Q pour chaque action
        q_values = self._get_q_values(state_features)
        
        available_actions_indices = self._get_available_actions_indices(observation)
        q_values_available = q_values[available_actions_indices]
        best_action_index = np.argmax(q_values_available)
        
        return available_actions_indices[best_action_index]
    
    def _get_q_values(self, features):
        """ Calcule les valeurs Q pour chaque action """
        return np.dot(features, self.weights.T) 
