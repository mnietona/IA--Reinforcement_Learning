from mdp import MDP, S, A
from typing import Generic

class ValueIteration(Generic[S, A]):
    def __init__(self, mdp: MDP[S, A], gamma: float):
        self.gamma = gamma  
        self.mdp = mdp  
        self.values = {state: 0.0 for state in self.mdp.states()}  

    def value(self, state: S) -> float:
        """ Retourne la valeur de l'état donné."""
        return self.values.get(state, 0.0)

    def policy(self, state: S) -> A:
        """ Retourne la meilleure action pour l'état donné."""
        best_action = max(self.mdp.available_actions(state),
                          key=lambda action: self.qvalue(state, action))
        return best_action

    def qvalue(self, state: S, action: A) -> float:
        """ Retourne la Q-valeur de l'état et de l'action donnés."""
        if self.mdp.is_final(state):
            return 0.0
        return sum(probability * (self.mdp.reward(state, action, new_state) + self.gamma * self.value(new_state))
                   for new_state, probability in self.mdp.transitions(state, action))

    def _compute_value_from_qvalues(self, state: S) -> float:
        """ Retourne la valeur de l'état donné en utilisant les Q-valeurs."""
        if self.mdp.is_final(state):
            return 0.0
        return max(self.qvalue(state, action) for action in self.mdp.available_actions(state))

    def value_iteration(self, n: int):
        """ Effectue n itérations de l'algorithme de Value Iteration."""""
        for _ in range(n):
            for state in self.mdp.states():
                self.values[state] = self._compute_value_from_qvalues(state)