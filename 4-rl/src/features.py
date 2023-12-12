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
        self.exit_pos, self.laser_positions = self._extract_all_positions(observation.data[self.agent_id])

    def feature_extraction(self) -> np.array:
        """ Extrait des caractéristiques de l'état du monde pour un agent spécifique. """
        
        total_distance_to_gems_and_exit = self._calculate_total_distance_to_gems_and_exit()
        n_gems_not_collected = self._calculate_gems_not_collected(self.agent_pos, self.gem_positions)
        min_row_dist, min_col_dist = self._calculate_min_distance(self.agent_pos, self.gem_positions)
        laser_north, laser_south, laser_east, laser_west = self._check_lasers_direction(self.agent_pos, self.laser_positions)
        dist_to_exit_if_no_gems = self._distance_to_exit_if_no_gems()
        nearest_target_indicator = self._nearest_target_indicator()
        direct_visibility_to_nearest_gem = self._check_direct_visibility_to_nearest_gem()
        obstacle_density = self._calculate_obstacle_density()
        average_distance_to_remaining_gems = self._calculate_average_distance_to_remaining_gems()
        potential_path_to_nearest_gem = self._potential_path_to_nearest_gem()
        is_agent_alive = self._is_agent_alive()
        proximity_to_gems = self._proximity_to_gems()
        count_possible_paths = self._count_possible_paths()
        action_success = self._action_success()
        
        features = self._min_max_normalize([
            total_distance_to_gems_and_exit, is_agent_alive,
            n_gems_not_collected, min_row_dist, min_col_dist,
            laser_north, laser_south, laser_east, laser_west,
            dist_to_exit_if_no_gems, nearest_target_indicator,
            direct_visibility_to_nearest_gem, obstacle_density,
            average_distance_to_remaining_gems, potential_path_to_nearest_gem,
            proximity_to_gems, count_possible_paths, action_success
        ])

        return np.array(features)
    
    def _action_success(self):
        """ Vérifie si l'action a réussi. """
        return self.reward > 0
    
    def _count_possible_paths(self):
        """ Compte le nombre de chemins possibles à partir de la position actuelle de l'agent. """
        possible_moves = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
            if new_pos not in self.wall_positions:
                possible_moves += 1
        return possible_moves

    
    def _proximity_to_gems(self, proximity_threshold=2):
        """ Calcule si des gemmes sont à proximité directe de l'agent. """
        close_gems_count = sum(1 for gem_pos in self.gem_positions if np.linalg.norm(np.array(gem_pos) - np.array(self.agent_pos), ord=1) <= proximity_threshold)
        return close_gems_count

    
    def _is_agent_alive(self):
        """ Détermine si l'agent est vivant. """
        # L'état "mort" peut être déterminé par la superposition de la position de l'agent avec des lasers ou des chutes
        # Cette implémentation suppose que l'agent est mort s'il se trouve sur la même position qu'un laser
        for laser_pos in self.laser_positions:
            if self.agent_pos == laser_pos:
                return 0 if self.reward < 0 else 1 # L'agent est considéré comme mort
        # Ajoutez ici toute autre logique pertinente pour votre jeu
        return 1  # L'agent est vivant

    
    def _potential_path_to_nearest_gem(self):
        """ Estime le chemin potentiel vers la gemme la plus proche. """
        if not self.gem_positions:
            return 0  # Si aucune gemme n'est restante, retourner 0

        nearest_gem_pos = min(self.gem_positions, key=lambda gem: np.sum(np.abs(np.array(gem) - np.array(self.agent_pos))))
        path_length = self._calculate_path_length(self.agent_pos, nearest_gem_pos)
        return path_length

    def _calculate_path_length(self, start_pos, end_pos):
        """ Calcule un chemin simplifié en ignorant les obstacles. """
        return np.abs(start_pos[0] - end_pos[0]) + np.abs(start_pos[1] - end_pos[1])

    
    def _calculate_average_distance_to_remaining_gems(self):
        """ Calcule la distance moyenne de l'agent à toutes les gemmes non collectées. """
        if not self.gem_positions:  # Si toutes les gemmes sont déjà collectées
            return 0

        total_distance = sum([np.abs(self.agent_pos[0] - gem[0]) + np.abs(self.agent_pos[1] - gem[1]) for gem in self.gem_positions])
        average_distance = total_distance / len(self.gem_positions) if self.gem_positions else 0

        return average_distance

    
    def _calculate_obstacle_density(self, radius=2):
        """ Calcule la densité des obstacles autour de l'agent dans un rayon donné. """
        obstacle_count = 0

        for row in range(self.agent_pos[0] - radius, self.agent_pos[0] + radius + 1):
            for col in range(self.agent_pos[1] - radius, self.agent_pos[1] + radius + 1):
                if (row, col) in self.wall_positions or self._is_laser_position(row, col):
                    obstacle_count += 1

        return obstacle_count

    def _is_laser_position(self, row, col):
        """ Vérifie si la position donnée est une position de laser. """
        return any(laser_pos == (row, col) for laser_pos in self.laser_positions)

    
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
    
    def _check_direct_visibility_to_nearest_gem(self):
        """ Vérifie si l'agent a une visibilité directe vers la gemme la plus proche. """
        # Implémentation simplifiée - À ajuster en fonction de votre environnement
        if not self.gem_positions:
            return 0
        # Supposons une ligne de vue directe si aucune position de mur n'est alignée avec la gemme
        nearest_gem_pos = min(self.gem_positions, key=lambda x: np.linalg.norm(np.array(x) - np.array(self.agent_pos)))
        return 1 if not any(self._is_aligned(nearest_gem_pos, wall_pos) for wall_pos in self.wall_positions) else 0

    def _is_aligned(self, pos1, pos2):
        """ Vérifie si deux positions sont alignées (horizontalement ou verticalement). """
        return pos1[0] == pos2[0] or pos1[1] == pos2[1]

    def _distance_to_exit_if_no_gems(self):
        """ Distance à la sortie si toutes les gemmes sont collectées. """
        if not self.gem_positions:  # Si aucune gemme n'est restante
            min_row_dist_to_exit, _ = self._calculate_min_distance(self.agent_pos, self.exit_pos)
            return min_row_dist_to_exit  # Utilisez seulement la distance en ligne
        return 0  # Ou une autre valeur par défaut

    def _nearest_target_indicator(self):
        """ Indicateur si la gemme la plus proche est plus proche que la sortie. """
        if not self.gem_positions:  # Si aucune gemme n'est restante
            return 0  # La sortie est le seul objectif restant
        min_row_dist_to_gem, min_col_dist_to_gem = self._calculate_min_distance(self.agent_pos, self.gem_positions)
        min_row_dist_to_exit, min_col_dist_to_exit = self._calculate_min_distance(self.agent_pos, self.exit_pos)

        total_dist_to_gem = min_row_dist_to_gem + min_col_dist_to_gem
        total_dist_to_exit = min_row_dist_to_exit + min_col_dist_to_exit

        return int(total_dist_to_gem < total_dist_to_exit)  # 1 si la gemme est plus proche, sinon 0

    def _calculate_gems_not_collected(self, agent_pos, gem_positions):
        """ Calcule le nombre de gemmes non collectées. """
        return sum(1 for pos in gem_positions if pos != agent_pos)

    def _calculate_min_distance(self, agent_pos, positions):
        """ Calcule la distance minimale en lignes et en colonnes à un ensemble de positions. """
        if not positions:
            return 0, 0

        # Conversion des positions en tableau NumPy pour une manipulation efficace
        positions_array = np.array(positions)
        agent_pos_array = np.array(agent_pos)

        # Calcul des distances de Manhattan de manière vectorisée
        manhattan_distances = np.abs(positions_array - agent_pos_array).sum(axis=1)

        # Obtention de l'indice de la distance minimale
        min_index = np.argmin(manhattan_distances)

        # Extraction de la position avec la distance minimale
        min_distance_position = positions_array[min_index]

        # Calcul des distances en lignes et en colonnes
        min_row_dist = np.abs(min_distance_position[0] - agent_pos[0])
        min_col_dist = np.abs(min_distance_position[1] - agent_pos[1])

        return min_row_dist, min_col_dist

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

    def _calculate_layer_indices(self):
        """ Calcule les indices des différentes couches. """
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
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Vérifie si la différence max-min est zéro pour éviter la division par zéro
        if max_val - min_val == 0:
            return data  # ou retournez un traitement alternatif, comme np.zeros(data.shape)
        else:
            return (data - min_val) / (max_val - min_val)