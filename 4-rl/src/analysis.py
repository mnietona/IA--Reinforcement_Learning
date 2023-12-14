from dataclasses import dataclass

@dataclass
class Parameters:
    reward_live: float
    """Reward for living at each time step"""
    gamma: float
    """Discount factor"""
    noise: float
    """Probability of taking a random action instead of the chosen one"""

def prefer_close_exit_following_the_cliff() -> Parameters:
    # Préférer la sortie proche (+1) en longeant la falaise.
    return Parameters(reward_live = -0.9, gamma = 0.5, noise = 0.1)

def prefer_close_exit_avoiding_the_cliff() -> Parameters:
    # Préférer la sortie proche (+1) en évitant la falaise.
    return Parameters(reward_live = -0.9, gamma = 0.1, noise = 0.9)

def prefer_far_exit_following_the_cliff() -> Parameters:
    # Préférer la sortie distante (+10) en longeant la falaise.
    return Parameters(reward_live = 0.3, gamma = 0.9, noise = 0.1)

def prefer_far_exit_avoiding_the_cliff() -> Parameters:
    # Préférer la sortie distante (+10) en évitant la falaise.
    return Parameters(reward_live = 0.3, gamma = 0.5, noise = 0.9)

def never_end_the_game() -> Parameters:
    #  Eviter de terminer le jeu (l’épisode ne se termine jamais).
    return Parameters(reward_live = 1, gamma = 1, noise = 1)
