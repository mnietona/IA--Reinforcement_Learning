from rlenv import RLEnv, Observation
from lle import World, WorldState
from qlearning import QLearning
import numpy as np
from typing import List, Tuple


def feature_extraction(observation: Observation, agent_id: int) -> np.array:
    """ Extrait des caractéristiques de l'état du monde pour un agent spécifique en utilisant l'observation. """
  
    # Extraire la couche d'observation spécifique à cet agent
    agent_layer = observation.data[agent_id]
    
    # Extraire la position de l'agent sous forme de tuple
    agent_pos = tuple(np.argwhere(agent_layer[0] == 1)[0])

    # Calculer le nombre de gemmes sur la carte
    gem_layer = agent_layer[3] 
    n_gems = np.sum(gem_layer == 1)
    
    #Trouver toutes les positions des gemmes et les convertir en liste de tuples
    gem_positions = [tuple(pos) for pos in np.argwhere(gem_layer == 1)]
    
    # Calculer le nombre de gemmes non collectées
    if agent_pos in gem_positions:
        n_gems -= 1
    n_gems_not_collected = n_gems
 
    # Calculer la distance minimale en lignes et en colonnes à la gemme la plus proche
    if len(gem_positions) > 0:
        # Calculer les différences absolues pour chaque gemme et trouver la valeur minimale
        row_distances = [np.abs(pos[0] - agent_pos[0]) for pos in gem_positions]
        col_distances = [np.abs(pos[1] - agent_pos[1]) for pos in gem_positions]
        min_row_dist = min(row_distances)
        min_col_dist = min(col_distances)
    else:
        # S'il n'y a pas de gemmes, fixer les distances à 0
        min_row_dist, min_col_dist = 0, 0
    
    return np.array([n_gems_not_collected, min_row_dist, min_col_dist])



class ApproximateQLearning(QLearning):
    def __init__(self, id, alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=5, n_features=4):
        super().__init__(id, alpha, gamma, epsilon, n_actions)
        self.weights = np.random.rand(n_actions, n_features) # Matrice des poids initialisée aléatoirement

    def update(self, observation, action, reward, next_observation):
        """ Mise à jour des poids """

        state_features = feature_extraction(observation, self.id)
        next_state_features = feature_extraction(next_observation, self.id)

        q_values_next = self._get_q_values(next_state_features)
        q_value = self._get_q_values(state_features)[action]

        td_error = (reward + self.gamma * np.max(q_values_next)) - q_value

        for i in range(len(state_features)):
            self.weights[action, i] += self.alpha * td_error * state_features[i]

        # Clipping des poids pour éviter les valeurs extrêmes
        self.weights = np.clip(self.weights, -1e0, 1e0)
        

    def _exploit(self, observation) -> int:
        """ Exploite en choisissant la meilleure action """
        # Etapes 1 : Extraction des caractéristiques de l'état du monde
        state_features = feature_extraction(observation, self.id)
        
        # Etape 2 : Calcul des valeurs Q pour chaque action
        q_values = self._get_q_values(state_features)
        #print(f"q_values: {q_values}")
        # Etape 3 : Sélection de la meilleure action
        available_actions_indices = self._get_available_actions_indices(observation)
        q_values_available = q_values[available_actions_indices]
        best_action_index = np.argmax(q_values_available)
        
        return available_actions_indices[best_action_index]
    
    def _get_q_values(self, features):
        """ Calcule les valeurs Q pour chaque action """
        return np.dot(features, self.weights.T)

