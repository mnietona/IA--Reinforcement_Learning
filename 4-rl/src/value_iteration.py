from mdp import MDP, S, A
from typing import Generic, Dict

class ValueIteration(Generic[S, A]):
    def __init__(self, mdp: MDP[S, A], gamma: float):
        self.gamma = gamma  # Discount factor
        self.mdp = mdp  # The Markov Decision Process
        self.values = {state: 0.0 for state in self.mdp.states()}  # Initialize state values

    def value(self, state: S) -> float:
        """Returns the value of the given state."""
        return self.values.get(state, 0.0)

    def policy(self, state: S) -> A:
        """Returns the action that maximizes the Q-value of the given state."""
        best_action = max(self.mdp.available_actions(state),
                          key=lambda action: self.qvalue(state, action))
        return best_action

    def qvalue(self, state: S, action: A) -> float:
        """Returns the Q-value of the given state-action pair based on the state values."""
        return sum(probability * (self.mdp.reward(state, action, new_state) + self.gamma * self.value(new_state))
                   for new_state, probability in self.mdp.transitions(state, action))

    def _compute_value_from_qvalues(self, state: S) -> float:
        """Computes the value for a state based on its Q-values."""
        if self.mdp.is_final(state):
            return 0.0
        return max(self.qvalue(state, action) for action in self.mdp.available_actions(state))

    def value_iteration(self, n: int):
        for _ in range(n):
            for state in self.mdp.states():
                self.values[state] = self._compute_value_from_qvalues(state)

