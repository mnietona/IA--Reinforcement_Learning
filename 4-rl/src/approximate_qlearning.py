from rlenv import Observation
from qlearning import QLearning
from features import FeatureExtractor
import numpy as np

class ApproximateQLearning(QLearning):
    def __init__(self, id, n_features, alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=5):
        super().__init__(id, alpha, gamma, epsilon, n_actions)
        # Initialisation des poids aléatoirement entre -0.001 et 0.001
        self.weights = np.random.uniform(low=-0.001, high=0.001, size=(self.n_actions, n_features))

    def choose_action(self, observation: Observation) -> int:
        """ Choix d'une action """
        if self._should_explore():
            return self._explore(observation)
        else:
            return self._exploit(observation)

    def update(self, observation, action, reward, next_observation, done):
        """ Mise à jour des poids """
        features = self._extract_features(observation, reward)
        next_features = self._extract_features(next_observation, reward)
        q_value_current = self._calculate_q_values(np.squeeze(self.weights[action]), np.squeeze(features))
        q_value_next = np.max(self._calculate_q_values(self.weights, next_features)) if not done else 0
        td_error = self._calculate_td_error(reward, self.gamma, q_value_current, q_value_next, done)
        self.weights[action] += self.alpha * td_error * features
    
    def _exploit(self, observation) -> int:
        """ Exploite en choisissant la meilleure action """
        available_actions_indices = self._get_available_actions_indices(observation)
        features = self._extract_features(observation, reward=0)
        q_values_valid = self._calculate_q_values(self.weights[available_actions_indices], features)
        return self._select_best_action(q_values_valid, available_actions_indices)

    def _extract_features(self, observation, reward):
        """ Extraction des features """
        feature_extractor = FeatureExtractor(observation, self.id, reward)
        return feature_extractor.feature_extraction()

    def _calculate_q_values(self, weights, features):
        """ Calcul des Q-valeurs """
        return np.dot(weights, features)

    def _select_best_action(self, q_values_valid, available_actions_indices):
        """ Sélection de la meilleure action """
        best_action_index = np.argmax(q_values_valid)
        return available_actions_indices[best_action_index]

    def _calculate_td_error(self, reward, gamma, q_value_current, q_value_next, done):
        """ Calcul de l'erreur TD """
        q_value_next = q_value_next if not done else 0
        return (reward + gamma * q_value_next) - q_value_current