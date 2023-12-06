from mdp import MDP, S, A
from typing import Generic



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
        """ Returns the Q-value of the given state-action pair based on the state values.""" 
        if self.mdp.is_final(state):
            return 0.0
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
    
    def optimal_policy(self) -> dict[S, A]:
        """ Returns the optimal policy."""
        return {state: self.policy(state) for state in self.mdp. states() if not self.mdp.is_final(state)}
    

if __name__ == "__main__":
    from lle import World
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from tests.world_mdp import WorldMDP


    def policy_iteration(value_iteration, n: int):
        stable = False
        iteration = 0
        previous_policy = {state: None for state in value_iteration.mdp.states()}

        while not stable and iteration < n:
            stable = True  
            new_values = {}

            for state in value_iteration.mdp.states():
                new_value = value_iteration._compute_value_from_qvalues(state)
                new_values[state] = new_value

                new_action = value_iteration.policy(state)
                if previous_policy[state] != new_action:
                    stable = False
                    previous_policy[state] = new_action

            value_iteration.values = new_values
            iteration += 1

        print(f"Politique stabilisée après {iteration} itérations.")

    def get_policy_grid(value_iteration, world):
        policy_grid_with_gems = np.zeros((world.height, world.width), dtype=object)
        policy_grid_without_gems = np.zeros((world.height, world.width), dtype=object)

        for state in value_iteration.mdp.states():
            policy_action = value_iteration.policy(state)
            agent_position = state.agents_positions[0]
            x, y = agent_position
            gems_collected = state.gems_collected[0]

            if gems_collected:
                policy_grid_with_gems[x][y] = str(policy_action[0])
            else:
                policy_grid_without_gems[x][y] = str(policy_action[0])

        return policy_grid_with_gems, policy_grid_without_gems

    def plot_arrow( x, y, action, color, offset=(0, 0)):
            dx, dy = 0, 0  # Delta x et y pour la direction de la flèche
            head_w, head_l = 0.1, 0.1  # Largeur et longueur de la tête de la flèche

            if action == 'North':
                dy = -0.4
            elif action == 'South':
                dy = 0.4
            elif action == 'West':
                dx = -0.4
            elif action == 'East':
                dx = 0.4

            # Appliquer le décalage
            x += offset[0]
            y += offset[1]

            plt.arrow(x, y, dx, dy, head_width=head_w, head_length=head_l, fc=color, ec=color, lw=2)

    def plot_policy_action(w, policy_grid_with_gems, policy_grid_without_gems):
        plt.figure(figsize=(5, 5))  # Ajustez la taille de la figure

        # Définit le décalage pour les flèches rouges
        offset_red = (0.1, 0)
        # Définit le décalage pour les flèches bleues
        offset_blue = (-0.1, 0)

        for y in range(w.height):
            for x in range(w.width):
                action_with_gems = policy_grid_with_gems[y, x]
                action_without_gems = policy_grid_without_gems[y, x]

                # Tracer les actions avec les gemmes collectées (bleu) avec décalage
                if action_with_gems:
                    plot_arrow(x, y, action_with_gems, 'blue', offset_blue)

                # Tracer les actions sans les gemmes collectées (rouge) avec décalage
                if action_without_gems:
                    plot_arrow(x, y, action_without_gems, 'red', offset_red)

        plt.xlim(-0.5, w.width - 0.5)
        plt.ylim(-0.5, w.height - 0.5)
        plt.gca().invert_yaxis()  # Inverser l'axe y pour l'affichage
        plt.xticks(range(w.width))
        plt.yticks(range(w.height))
        plt.show()

    def execute_policy(mdp, value_iteration):
        mdp.world.reset()
        step = 0
        while not mdp.is_final(mdp.world.get_state()):
            current_state = mdp.world.get_state()
            action = value_iteration.policy(current_state)
            mdp.transitions(current_state, action)
            step += 1
        if mdp.world.get_state().gems_collected[0]:
            print(f" Done in {step} steps with {(len(mdp.world.get_state().gems_collected))} gems")


    mdp = WorldMDP(World.from_file("level1"))
    w = World.from_file("level1")
    
    value_iteration = ValueIteration(mdp, gamma=0.9)

    policy_iteration(value_iteration, 1000) 
    policy_grid_with_gems, policy_grid_without_gems = get_policy_grid(value_iteration, w)
    execute_policy(mdp, value_iteration)
    plot_policy_action(w, policy_grid_with_gems, policy_grid_without_gems)

    
    
    
    

