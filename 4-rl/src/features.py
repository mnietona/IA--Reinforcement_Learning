from rlenv import Observation
import numpy as np

class FeatureExtractor:
    def __init__(self, observation: Observation, agent_id: int, reward):
        self.observation = observation
        self.agent_id = agent_id
        self.reward = reward
        
        self.wall_layer_index, self.laser_layer_start, self.laser_layer_end, \
        self.void_layer_index, self.gem_layer_index, self.exit_layer_index = self._calculate_layer_indices()
        
        self.agent_pos, self.wall_positions, self.gem_positions, \
        self.exit_pos, self.laser_positions = self._extract_all_positions(observation.data[agent_id])

    def feature_extraction(self) -> np.array:
        features = self._min_max_normalize([
            self._calculate_total_distance_to_gems_and_exit(),
            self._calculate_gems_not_collected(self.agent_pos, self.gem_positions), 
            *self._calculate_min_distance_to_targets(self.gem_positions),
            *self._check_lasers_direction(self.agent_pos, self.laser_positions), 
            self._check_direct_visibility(), 
            self._proximity_to_gems(),
            self._calculate_obstacle_density(),
            self._potential_path_to_nearest_gem(),
            self._count_possible_paths(), 
            self.reward < 0 # Agent est mort 
        ])

        return np.array(features)
    
    def _calculate_layer_indices(self):
        """ Calcule les indices des couches de la matrice d'observation. """
        n_agents = self.observation.n_agents
        wall_layer_index = n_agents
        exit_layer_index = len(self.observation.data[self.agent_id]) - 1
        gem_layer_index = exit_layer_index - 1
        void_layer_index = gem_layer_index - 1
        laser_layer_start = wall_layer_index + 1
        laser_layer_end = void_layer_index - 1
        return wall_layer_index, laser_layer_start, laser_layer_end, void_layer_index, gem_layer_index, exit_layer_index

    def _extract_all_positions(self,all_layers):
        """ Extrait les positions de tous objet. """
        agent_pos = self._extract_agent_position(all_layers[self.agent_id])
        wall_position = self._extract_positions(all_layers[self.wall_layer_index], value=1)
        gem_positions = self._extract_positions(all_layers[self.gem_layer_index], value=1)
        exit_positions = self._extract_positions(all_layers[self.exit_layer_index], value=1)
        laser_positions = self._extract_laser_positions(all_layers,self.laser_layer_start, self.laser_layer_end)
        
        return agent_pos, wall_position, gem_positions, exit_positions, laser_positions
        
    def _extract_agent_position(self, agent_layer):
        """ Extrait la position de l'agent. """
        return tuple(np.argwhere(agent_layer == 1)[0])

    def _extract_positions(self, layer, value):
        """ Extrait les positions pour une valeur donnée dans une couche. """
        return [tuple(pos) for pos in np.argwhere(layer == value)]

    def _extract_laser_positions(self, all_layer, start, end):
        """ Extrait les positions des lasers. """
        positions = []
        for i in range(start, end + 1):
            positions.extend(self._extract_positions(all_layer[i], value=1))
            positions.extend(self._extract_positions(all_layer[i], value=-1))
        return positions

    def _min_max_normalize(self, data):
        """ Normalise les données entre 0 et 1. """""
        min_val = np.min(data)
        max_val = np.max(data)
        return data if max_val - min_val == 0 else (data - min_val) / (max_val - min_val)
    
    def _calculate_total_distance_to_gems_and_exit(self):
        """ Calcule la distance totale de l'agent aux gemmes non collectées et à la sortie. """
        total_distance = 0
        if self.gem_positions:
            # Distance à la gemme la plus proche
            gem_distances = [np.abs(self.agent_pos[0] - gem[0]) + np.abs(self.agent_pos[1] - gem[1]) for gem in self.gem_positions]
            total_distance += min(gem_distances)
        else:
            # Distance à la sortie la plus proche si toutes les gemmes sont collectées
            exit_distances = [np.abs(self.agent_pos[0] - exit_pos[0]) + np.abs(self.agent_pos[1] - exit_pos[1]) for exit_pos in self.exit_pos]
            total_distance += min(exit_distances) if exit_distances else 0

        return total_distance

    def _calculate_gems_not_collected(self, agent_pos, gem_positions):
        """ Calcule le nombre de gemmes non collectées. """
        return sum(1 for pos in gem_positions if pos != agent_pos)

    def _calculate_min_distance_to_targets(self, targets):
        if not targets:
            return 0, 0

        positions_array = np.array(targets)
        agent_pos_array = np.array(self.agent_pos)
        manhattan_distances = np.abs(positions_array - agent_pos_array).sum(axis=1)
        min_index = np.argmin(manhattan_distances)
        min_distance_position = positions_array[min_index]

        return np.abs(min_distance_position[0] - self.agent_pos[0]), np.abs(min_distance_position[1] - self.agent_pos[1])

    def _check_lasers_direction(self, agent_pos, laser_positions):
        """ Vérifie la présence de lasers dans chaque direction par rapport à la position de l'agent. """
        laser_north = laser_south = laser_east = laser_west = 0
        for laser_pos in laser_positions:
            if laser_pos[0] < agent_pos[0] and laser_pos[1] == agent_pos[1]:
                laser_north += 1
            elif laser_pos[0] > agent_pos[0] and laser_pos[1] == agent_pos[1]:
                laser_south += 1
            elif laser_pos[1] > agent_pos[1] and laser_pos[0] == agent_pos[0]:
                laser_east += 1
            elif laser_pos[1] < agent_pos[1] and laser_pos[0] == agent_pos[0]:
                laser_west += 1

        return laser_north, laser_south, laser_east, laser_west

    def _is_aligned(self, pos1, pos2):
        """ Vérifie si deux positions sont alignées (horizontalement ou verticalement). """
        return pos1[0] == pos2[0] or pos1[1] == pos2[1]

    def _check_direct_visibility(self):
        """ Vérifie si l'agent a une visibilité directe vers la gemme la plus proche. """
        if not self.gem_positions:  # Si aucune gemme n'est restante va à la sortie
            # Convertir self.exit_pos en tuple si c'est un tableau numpy
            exit_positions = [tuple(pos) for pos in self.exit_pos] if isinstance(self.exit_pos, np.ndarray) else self.exit_pos
            return 1 if not any(self._is_aligned(exit_pos, wall_pos) for exit_pos in exit_positions for wall_pos in self.wall_positions) else 0

        nearest_gem_pos = min(self.gem_positions, key=lambda x: np.linalg.norm(np.array(x) - np.array(self.agent_pos)))
        return 1 if not any(self._is_aligned(nearest_gem_pos, wall_pos) for wall_pos in self.wall_positions) else 0

    def _proximity_to_gems(self, proximity_threshold=2):
        """ Calcule si des gemmes sont à proximité directe de l'agent. """
        close_gems_count = 0
        for gem_pos in self.gem_positions:
            distance = np.linalg.norm(np.array(gem_pos) - np.array(self.agent_pos), ord=1)
            if distance <= proximity_threshold:
                close_gems_count += 1
        return close_gems_count
    
    # Les suivantes Features ont été proposé par ChatGpt
    def _is_laser_position(self, row, col):
        """ Vérifie si la position donnée est une position de laser. """
        return any(laser_pos == (row, col) for laser_pos in self.laser_positions)

    def _calculate_obstacle_density(self, radius=2):
        """ Calcule la densité des obstacles autour de l'agent dans un rayon donné. """
        obstacle_count = 0

        for row in range(self.agent_pos[0] - radius, self.agent_pos[0] + radius + 1):
            for col in range(self.agent_pos[1] - radius, self.agent_pos[1] + radius + 1):
                if (row, col) in self.wall_positions or self._is_laser_position(row, col):
                    obstacle_count += 1

        return obstacle_count

    def _calculate_path_length(self, start_pos, end_pos):
        """ Calcule un chemin simplifié en ignorant les obstacles. """
        return np.abs(start_pos[0] - end_pos[0]) + np.abs(start_pos[1] - end_pos[1])

    def _potential_path_to_nearest_gem(self):
        """ Estime le chemin potentiel vers la gemme la plus proche. """
        if not self.gem_positions:
            return 0  # Si aucune gemme n'est restante, retourner 0

        nearest_gem_pos = min(self.gem_positions, key=lambda gem: np.sum(np.abs(np.array(gem) - np.array(self.agent_pos))))
        path_length = self._calculate_path_length(self.agent_pos, nearest_gem_pos)
        return path_length
    
    def _count_possible_paths(self):
        """ Compte le nombre de chemins possibles à partir de la position actuelle de l'agent. """
        possible_moves = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
            if new_pos not in self.wall_positions:
                possible_moves += 1
        return possible_moves