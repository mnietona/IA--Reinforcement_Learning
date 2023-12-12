from rlenv import RLEnv, Observation
from qlearning import QLearning
from features import FeatureExtractor
import numpy as np


class ApproximateQLearning(QLearning):
    def __init__(self, id, n_features, alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=5):
        super().__init__(id, alpha, gamma, epsilon, n_actions)
        self.weights = np.random.uniform(low=-0.001, high=0.001, size=(self.n_actions, n_features))

    def choose_action(self, observation: Observation) -> int:
        """ Choix d'une action """
        feature_extractor = FeatureExtractor(observation, self.id, reward = 0)  
        
        if self._should_explore():
            return self._explore(observation)
        else:
            return self._exploit(observation, feature_extractor)

    def _exploit(self, observation, feature_extractor) -> int:
        """ Exploite en choisissant la meilleure action parmi les actions valides """
        available_actions_indices = self._get_available_actions_indices(observation)
        features = feature_extractor.feature_extraction() 
        
        q_values_valid = np.dot(self.weights[available_actions_indices], features)
        best_action_index = np.argmax(q_values_valid)
        return available_actions_indices[best_action_index]

    def update(self, observation, action, reward, next_observation, done):
        """ Mise à jour des poids """
        feature_extractor = FeatureExtractor(observation, self.id, reward)  # Pour l'état actuel
        next_feature_extractor = FeatureExtractor(next_observation, self.id, reward)  # Pour le prochain état

        features = feature_extractor.feature_extraction()
        next_features = next_feature_extractor.feature_extraction()
        
        q_value_current = np.dot(np.squeeze(self.weights[action]), np.squeeze(features))
        
        q_value_next = np.max(np.dot(self.weights, next_features)) if not done else 0

        td_error = (reward + self.gamma * q_value_next) - q_value_current
        self.weights[action] += self.alpha * td_error * features


    


