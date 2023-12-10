from rlenv import Observation
import numpy as np


def feature_extraction(observation: Observation, agent_id: int) -> np.array:
    """ Extrait des caractéristiques de l'état du monde pour un agent spécifique. """
    # Calculer les indices des couches
    wall_layer_index, laser_layer_start, laser_layer_end, _, gem_layer_index, exit_layer_index = calculate_layer_indices(observation, agent_id)
    
    # Extraire les caractéristiques
    all_layers = observation.data[agent_id]
    
    # Extraire les positions
    agent_pos, wall_positon, gem_positions, exit_pos, laser_positions = extract_all_position(all_layers, 
                                                                        agent_id, wall_layer_index, laser_layer_start,
                                                                        laser_layer_end, gem_layer_index, exit_layer_index)
    
    # Calculer les caractéristiques
    n_gems_not_collected = calculate_gems_not_collected(agent_pos, gem_positions)
    min_row_dist, min_col_dist = calculate_min_distance(agent_pos, gem_positions)
    row_dist_to_exit, col_dist_to_exit = calculate_min_distance(agent_pos, exit_pos)
    laser_north, laser_south, laser_east, laser_west = check_lasers_direction(agent_pos, laser_positions)
    
    return np.array([n_gems_not_collected, min_row_dist, min_col_dist, row_dist_to_exit, col_dist_to_exit, laser_north, laser_south, laser_east, laser_west])


def extract_all_position(all_layers, agent_id, wall_layer_index, laser_layer_start,
                         laser_layer_end, gem_layer_index, exit_layer_index):
    """ Extrait les positions de tous objet. """
    agent_pos = extract_agent_position(all_layers[agent_id])
    wall_positon = extract_positions(all_layers[wall_layer_index], value=1)
    gem_positions = extract_positions(all_layers[gem_layer_index], value=1)
    exit_positions = extract_positions(all_layers[exit_layer_index], value=1)
    laser_positions = extract_laser_positions(all_layers, laser_layer_start, laser_layer_end)
    return agent_pos, wall_positon, gem_positions, exit_positions, laser_positions
     
def calculate_layer_indices(observation: Observation, agent_id: int):
    """ Calcule les indices des différentes couches. """
    n_agents = observation.n_agents
    wall_layer_index = n_agents
    exit_layer_index = len(observation.data[agent_id]) - 1
    gem_layer_index = exit_layer_index - 1
    void_layer_index = gem_layer_index - 1
    laser_layer_start = wall_layer_index + 1
    laser_layer_end = void_layer_index - 1
    return wall_layer_index, laser_layer_start, laser_layer_end, void_layer_index, gem_layer_index, exit_layer_index

def extract_agent_position(agent_layer):
    """ Extrait la position de l'agent. """
    return tuple(np.argwhere(agent_layer == 1)[0])

def extract_positions(layer, value):
    """ Extrait les positions pour une valeur donnée dans une couche. """
    return [tuple(pos) for pos in np.argwhere(layer == value)]

def extract_laser_positions(all_layer, start, end):
    """ Extrait les positions des lasers. """
    positions = []
    for i in range(start, end + 1):
        positions.extend(extract_positions(all_layer[i], value=1))
        positions.extend(extract_positions(all_layer[i], value=-1))
    return positions

def calculate_gems_not_collected(agent_pos, gem_positions):
    """ Calcule le nombre de gemmes non collectées. """
    return sum(1 for pos in gem_positions if pos != agent_pos)

def calculate_min_distance(agent_pos, positions):
    """ Calcule la distance minimale en lignes et en colonnes à un ensemble de positions. """
    if positions:
        # Calculer la distance de Manhattan pour chaque position cible
        manhattan_distances = [np.abs(pos[0] - agent_pos[0]) + np.abs(pos[1] - agent_pos[1]) for pos in positions]

        # Trouver la distance minimale
        min_distance = min(manhattan_distances)

        # Trouver la position correspondant à la distance minimale
        min_distance_position = positions[manhattan_distances.index(min_distance)]

        # Calculer les distances en lignes et en colonnes séparément pour cette position
        min_row_dist = np.abs(min_distance_position[0] - agent_pos[0])
        min_col_dist = np.abs(min_distance_position[1] - agent_pos[1])

        return min_row_dist, min_col_dist

    return 0, 0

def check_lasers_direction(agent_pos, laser_positions):
    """ Vérifie la présence de lasers dans chaque direction par rapport à la position de l'agent. """
    laser_north = laser_south = laser_east = laser_west = 0
    for laser_pos in laser_positions:
        if laser_pos[0] < agent_pos[0] and laser_pos[1] == agent_pos[1]:
            laser_north = 1
        elif laser_pos[0] > agent_pos[0] and laser_pos[1] == agent_pos[1]:
            laser_south = 1
        elif laser_pos[1] > agent_pos[1] and laser_pos[0] == agent_pos[0]:
            laser_east = 1
        elif laser_pos[1] < agent_pos[1] and laser_pos[0] == agent_pos[0]:
            laser_west = 1

    return laser_north, laser_south, laser_east, laser_west
