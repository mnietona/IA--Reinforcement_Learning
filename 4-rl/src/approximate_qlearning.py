from rlenv import RLEnv, Observation
from lle import World, WorldState
from qlearning import QLearning
import numpy as np
from typing import List, Tuple




def feature_extraction(observation: Observation, agent_id) -> np.array:
    """ Extrait des caractéristiques de l'état du monde pour un agent spécifique. """
  
    world = observation
    # Extraire les caractéristiques de l'état du monde
    agent_pos = world.agents_positions[agent_id]
    
    # Calculer le nombre de gemmes non ramassées f(1)
    n_gems_not_collected = world.n_gems - world.gems_collected

    # Calculer la distance en lignes et en colonnes à la gemme la plus proche f(2) et f(3)
    min_row_dist , min_col_dist = float('inf'), float('inf')
    for gem_pos, gem in world.gems:
        if not gem.is_collected:
            row_dist = abs(agent_pos[0] - gem_pos[0])
            col_dist = abs(agent_pos[1] - gem_pos[1])
            min_row_dist = min(min_row_dist, row_dist)
            min_col_dist = min(min_col_dist, col_dist)
            
    # Exemple de détection de laser au nord de l'agent f(4)
    laser_north = any(laser.position == (agent_pos[0] - 1, agent_pos[1]) for laser in world.lasers)
    
    print(f"n_gems_not_collected: {n_gems_not_collected}")
    print(f"min_row_dist: {min_row_dist}")
    print(f"min_col_dist: {min_col_dist}")
    print(f"laser_north: {laser_north}")

    return np.array([n_gems_not_collected, min_row_dist, min_col_dist, int(laser_north)])



class ApproximateQLearning(QLearning):
    def __init__(self, id, alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=5, n_features=4):
        super().__init__(id, alpha, gamma, epsilon, n_actions)
        self.weights = np.random.randn(n_actions, n_features) # Matrice des poids initialisée aléatoirement

    def update(self, observation, action, reward, next_observation):
        """ Mise à jour des poids """
        
        state_features = feature_extraction(observation, self.id)
        next_state_features = feature_extraction(next_observation, self.id)

        q_values_next = self._get_q_values(next_state_features)
        q_value = self._get_q_values(state_features)[action]

        td_error = (reward + self.gamma * np.max(q_values_next)) - q_value

        # Mise à jour des poids pour l'action spécifique
        for i in range(len(state_features)):
            self.weights[action, i] += self.alpha * td_error * state_features[i]

    def _exploit(self, observation) -> int:
        """ Exploite en choisissant la meilleure action """
        # Etapes 1 : Extraction des caractéristiques de l'état du monde
        state_features = feature_extraction(observation, self.id)
        
        # Etape 2 : Calcul des valeurs Q pour chaque action
        q_values = self._get_q_values(state_features)
        
        # Etape 3 : Sélection de la meilleure action
        available_actions_indices = self._get_available_actions_indices(observation)
        q_values_available = q_values[available_actions_indices]
        best_action_index = np.argmax(q_values_available)
        
        return available_actions_indices[best_action_index]
    
    def _get_q_values(self, features):
        """ Calcule les valeurs Q pour chaque action """
        return np.dot(features, self.weights.T)

