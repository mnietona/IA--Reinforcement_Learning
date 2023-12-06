from rlenv import RLEnv, Observation
from lle import World
import numpy as np
import random
from typing import List, Tuple

class ApproximateQLearning:
    """ Approximate Q-Learning """
    
    def __init__(self, id, alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=5, n_features=4):
        self.weights = np.zeros((n_actions, n_features)) # Poids pour chaque action et caractéristique
        self.alpha = alpha        # Taux d'apprentissage
        self.gamma = gamma        # Facteur de remise
        self.epsilon = epsilon    # Facteur d'exploration
        self.n_actions = n_actions
        self.id = id

    def choose_action(self, observation: Observation) -> int:
        """ Choix d'une action """
        if self._should_explore():
            return self._explore(observation)
        else:
            return self._exploit(observation)

    def update(self, observation, action, reward, next_observation):
        """ Mise à jour des poids en fonction de l'erreur TD """
        features = self.feature_extraction(observation.state)
        next_features = self.feature_extraction(next_observation.state)
        
        q_value = self._get_q_value(features, action)
        next_max_q = max([self._get_q_value(next_features, a) for a in range(self.n_actions)])
        td_error = (reward + self.gamma * next_max_q) - q_value

        # Mise à jour des poids pour chaque caractéristique
        self.weights[action] += self.alpha * td_error * features

    def _should_explore(self): 
        """ Vérifie si l'agent doit explorer """
        return random.uniform(0, 1) < self.epsilon

    def _explore(self, observation) -> int:
        """ Explore en utilisant greedy-epsilon """
        available_actions_indices = self._get_available_actions_indices(observation)
        return np.random.choice(available_actions_indices)

    def _exploit(self, observation) -> int:
        """ Exploite en choisissant la meilleure action """
        features = self.feature_extraction(observation.state)
        available_actions_indices = self._get_available_actions_indices(observation)
        q_values = [self._get_q_value(features, a) for a in available_actions_indices]
        best_action_index = np.argmax(q_values)
        return available_actions_indices[best_action_index]

    def _get_q_value(self, features, action):
        """ Calcule la valeur Q pour un ensemble de caractéristiques et une action """
        return np.dot(self.weights[action], features)

    def _get_available_actions_indices(self, observation) -> np.array:
        """ Retourne les indices des actions disponibles """
        available_actions = observation.available_actions[self.id]
        return np.where(available_actions > 0)[0]

    def feature_extraction(self, world: World) -> np.array:
        """ Extrait des caractéristiques de l'état du monde pour un agent """
        # Caractéristiques à extraire
        
        gems_collected = world.gems_collected
        print(gems_collected)
        n_gems_not_collected = world.n_gems - gems_collected
        closest_gem_distance = self._find_closest_gem_distance(world)
        laser_north = self._is_laser_north(world)

        return np.array([n_gems_not_collected, *closest_gem_distance, laser_north])

    def _find_closest_gem_distance(self, world: World) -> Tuple[int, int]:
        """ Trouve la distance à la gemme la plus proche """
        min_row_dist = float('inf')
        min_col_dist = float('inf')

        agent_pos = world.agents_positions[self.id]
        for gem_pos, gem in world.gems:
            if not gem.is_collected:  # Vérifie si la gemme n'est pas collectée
                row_dist = abs(agent_pos[0] - gem_pos[0])
                col_dist = abs(agent_pos[1] - gem_pos[1])
                min_row_dist = min(min_row_dist, row_dist)
                min_col_dist = min(min_col_dist, col_dist)

        return min_row_dist, min_col_dist

    def _is_laser_north(self, world: World) -> int:
        """ Vérifie si un laser est au nord de l'agent """
        agent_pos = world.agents_positions[self.id]
        north_pos = (agent_pos[0] - 1, agent_pos[1])

        for laser_pos, laser in world.lasers:
            if laser_pos == north_pos:
                return 1
        return 0

    
