from dataclasses import dataclass

@dataclass
class Parameters:
    reward_live: float
    gamma: float
    noise: float

def prefer_close_exit_following_the_cliff() -> Parameters:
    # Préférer la sortie proche (+1) en longeant la falaise.
    reward_live, gamma, noise = -0.9, 0.9, 0.5
    return Parameters(reward_live, gamma, noise)

def prefer_close_exit_avoiding_the_cliff() -> Parameters:
    # Préférer la sortie proche (+1) en évitant la falaise.
    reward_live, gamma, noise = -0.9, 0.5, 0.9
    return Parameters(reward_live, gamma, noise)

def prefer_far_exit_following_the_cliff() -> Parameters:
    # Préférer la sortie distante (+10) en longeant la falaise.
    reward_live, gamma, noise = -0.5, 0.5, 0.1
    return Parameters(reward_live, gamma, noise)

def prefer_far_exit_avoiding_the_cliff() -> Parameters:
    # Préférer la sortie distante (+10) en évitant la falaise.
    reward_live, gamma, noise = 0.5, 0.5, 0.5 
    return Parameters(reward_live, gamma, noise)

def never_end_the_game() -> Parameters:
    #  Eviter de terminer le jeu (l’épisode ne se termine jamais).
    reward_live, gamma, noise = 1, 1, 1
    return Parameters(reward_live, gamma, noise)
