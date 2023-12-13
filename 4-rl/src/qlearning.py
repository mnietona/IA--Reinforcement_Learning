from rlenv import Observation
import numpy as np
import random

class QLearning:
    """ Q-Learning Tabular """
    
    def __init__(self,id, alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=5):
        self.q_table = {}          # Dictionnaire des Q-valeurs
        self.alpha = alpha         # Taux d'apprentissage
        self.gamma = gamma         # Facteur de remise
        self.epsilon = epsilon     # Facteur d'exploration
        self.n_actions = n_actions # Nombre d'actions possibles
        self.id = id
    
    def choose_action(self, observation: Observation) -> int:
        """ Choix d'une action """
        state_key = self._get_state_key(observation.state)
        self._initialize_state(state_key)
        
        if self._should_explore():
            return self._explore(observation)
        else:
            return self._exploit(state_key, observation)

    def update(self, observation, action, reward, next_observation,done):
        """ Mise à jour de la table Q"""
        state_key = self._get_state_key(observation.state)       
        next_state_key = self._get_state_key(next_observation.state)
        self._initialize_state(next_state_key)
        
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] = self._calculate_new_value(old_value, reward, next_max, done)

    def epsilon_decay(self, episode, n_episodes):
        """ Décroissance de l'epsilon. """
        
        self.epsilon = max(0.0 , 1 / (1 + np.exp(0.001 * (episode - n_episodes /2 )))) # Formule proposée par ChatGpt pour une décroissance de l'epsilon
        
    def _get_state_key(self, state) -> int:
        """ Retourne la clé de l'état """
        return hash(state.tobytes())

    def _initialize_state(self, state_key):
        """ Initialise l'état dans la table Q """
        if state_key not in self.q_table:
            self.q_table[state_key] = np.ones(self.n_actions)

    def _get_available_actions_indices(self, observation) -> np.array:
        """ Retourne les indices des actions disponibles """
        available_actions = observation.available_actions[self.id]
        return np.where(available_actions > 0)[0]

    def _should_explore(self) -> bool: 
        """ Vérifie si l'agent doit explorer """
        return random.uniform(0, 1) < self.epsilon

    def _explore(self, observation) -> int:
        """ Explore en utilisant greedy-epsilon """
        available_actions_indices = self._get_available_actions_indices(observation)
        return np.random.choice(available_actions_indices)

    def _exploit(self,state_key, observation) -> int :
        """ Exploite en choisissant la meilleure action """
        available_actions_indices = self._get_available_actions_indices(observation)
        q_values = self.q_table[state_key][available_actions_indices]
        best_action_index = np.argmax(q_values)
        return available_actions_indices[best_action_index]

    def _calculate_new_value(self, old_value, reward, next_max, done) -> float:
        """ Calcule la nouvelle valeur Q avec la formule de Bellman """
        return (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max * (1-done)) 
    
    