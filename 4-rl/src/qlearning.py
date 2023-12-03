from rlenv import RLEnv, Observation
import numpy as np
import numpy.typing as npt
import pandas as pd

class QLearning:
    """Tabular QLearning avec un dictionnaire pour les valeurs Q."""
    def __init__(self, env: RLEnv, alpha: float, gamma: float, epsilon: float):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Utilisation d'un dictionnaire pour stocker les valeurs Q.
        self.Q = {}

    def get_q_value(self, state, action):
        """Retourne la valeur Q pour un état et une action donnés."""
        # Hashage de l'état si c'est un tableau numpy.
        state_hash = hash(state.tobytes()) if isinstance(state, np.ndarray) else state
        return self.Q.get((state_hash, action), 1)  # Initialisation à 1 pour encourager l'exploration.

    def set_q_value(self, state, action, value):
        """Définit la valeur Q pour un état et une action donnés."""
        state_hash = hash(state.tobytes()) if isinstance(state, np.ndarray) else state
        self.Q[(state_hash, action)] = value

    
    def choose_action(self, state, available_actions) -> int:
        """Choisit une action basée sur la stratégie ε-greedy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            q_values = {a: self.get_q_value(state, a) for a in available_actions}
            return max(q_values, key=q_values.get)

    
    def update(self, state, action, reward, next_state) -> None:
        """Met à jour la valeur Q pour un état et une action donnés."""
        # Obtenir la meilleure valeur Q pour le prochain état.
        max_next_q = max([self.get_q_value(next_state, a) for a in range(self.env.n_actions)])
        # Calculer la nouvelle valeur Q.
        new_q = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * (reward + self.gamma * max_next_q)
        # Mettre à jour la valeur Q.
        self.set_q_value(state, action, new_q)
    
    # Methode qui affcihe les valeur de Q de chaque action a chaque state 


    def display_q_values(self):
        """Organise et affiche les valeurs Q dans un DataFrame pandas."""
        # Préparer les données pour le DataFrame
        data = {}
        for (state_hash, action), value in self.Q.items():
            if state_hash not in data:
                data[state_hash] = {}
            data[state_hash][action] = value
        
        # Créer le DataFrame
        df = pd.DataFrame(data).T  # .T pour transposer le DataFrame

        # Trier les colonnes (actions) dans l'ordre souhaité
        sorted_columns = sorted(df.columns)
        df = df[sorted_columns]

        df.fillna(0, inplace=True)  # Remplacer les NaN par 0

        # Afficher le DataFrame
        print(df)

        


