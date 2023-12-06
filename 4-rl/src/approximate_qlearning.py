from rlenv import RLEnv, Observation
from lle import World
from qlearning import QLearning
import numpy as np
from typing import List, Tuple


# def feature_extraction(world):
#     features = {}
#     for agent_id, agent in world.agents.items():
#         # Exemple de features
#         features[agent_id] = [
#             # f1(s): Nombre de gemmes non collectées
#             world.count_uncollected_gems(),
#             # f2(s): Distance à la gemme la plus proche en lignes
#             world.distance_to_closest_gem(agent, axis='row'),
#             # f3(s): Distance à la gemme la plus proche en colonnes
#             world.distance_to_closest_gem(agent, axis='col'),
#             # f4(s): Présence ou absence d'un laser nord
#             int(world.is_laser_north(agent))
#             # Ajouter plus de features si nécessaire
#         ]
#     return features

def _find_closest_gem_distance(world: World) -> Tuple[int, int]:
    """ Trouve la distance à la gemme la plus proche """
    min_row_dist = float('inf')
    min_col_dist = float('inf')
    # id = agents_id
    agent_pos = world.agents_positions[id]
    for gem_pos, gem in world.gems:
        if not gem.is_collected:  # Vérifie si la gemme n'est pas collectée
            row_dist = abs(agent_pos[0] - gem_pos[0])
            col_dist = abs(agent_pos[1] - gem_pos[1])
            min_row_dist = min(min_row_dist, row_dist)
            min_col_dist = min(min_col_dist, col_dist)

    return min_row_dist, min_col_dist

def _is_laser_north( world: World) -> int:
    """ Vérifie si un laser est au nord de l'agent """
    # id = agents_id
    agent_pos = world.agents_positions[id]
    north_pos = (agent_pos[0] - 1, agent_pos[1])

    for laser_pos, laser in world.lasers:
        if laser_pos == north_pos:
            return 1
    return 0

def Obsevation_to_world(observation: Observation) -> World:
    """ Convertit une observation en un monde """
    

def feature_extraction(world: World, agent_id) -> np.array:
    """ Extrait des caractéristiques de l'état du monde pour un agent """
    # Caractéristiques à extraire
    #gems_collected = world.gems_collected
    #n_gems_not_collected = world.n_gems - gems_collected
    
    #closest_gem_distance = _find_closest_gem_distance(world)
    #laser_north = _is_laser_north(world)
    n_gems_not_collected = 0.0
    closest_gem_distance = [0.0, 0.0]
    laser_north = 0.0

    return np.array([n_gems_not_collected, *closest_gem_distance, laser_north])


class ApproximateQLearning(QLearning):
    def __init__(self, id, alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=5, n_features=4):
        super().__init__(id, alpha, gamma, epsilon, n_actions)
        # Initialisation aléatoire des poids pour chaque action
        self.weights = np.random.randn(n_actions, n_features)  

    def _get_q_values(self, features):
        # Calcul des valeurs Q pour chaque action
        return np.dot(features, self.weights.T) # Produit scalaire pour chaque action


    def choose_action(self, observation):
        state_features = feature_extraction(observation, self.id)
        print(state_features)
        q_values = self._get_q_values(state_features)
        print(q_values)
        if self._should_explore():
            return self._explore(observation)
        else:
            return np.argmax(q_values)

    def update(self, observation, action, reward, next_observation):
        state_features = feature_extraction(observation, self.id)
        next_state_features = feature_extraction(next_observation, self.id)

        q_values_next = self._get_q_values(next_state_features)
        q_value = self._get_q_values(state_features)[action]

        td_error = (reward + self.gamma * np.max(q_values_next)) - q_value

        # Mise à jour des poids pour l'action spécifique
        for i in range(len(state_features)):
            self.weights[action, i] += self.alpha * td_error * state_features[i]
