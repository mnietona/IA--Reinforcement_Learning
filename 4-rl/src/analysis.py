from dataclasses import dataclass

@dataclass
class Parameters:
    reward_live: float  # Can be any value.
    gamma: float        # Should be between 0 and 1.
    noise: float        # Should be between 0 and 1.

def prefer_close_exit_following_the_cliff() -> Parameters:
    # High negative reward to drive the agent quickly to the exit, high gamma for future reward, low noise
    return Parameters(reward_live=-0.9, gamma=0.95, noise=0.05)

def prefer_close_exit_avoiding_the_cliff() -> Parameters:
    # Less negative reward for living to ensure the agent avoids the cliff, high gamma, low noise
    return Parameters(reward_live=-0.2, gamma=0.95, noise=0.05)

def prefer_far_exit_following_the_cliff() -> Parameters:
    # Positive reward for living to motivate reaching the far exit, very high gamma, moderate noise
    return Parameters(reward_live=0.01, gamma=0.99, noise=0.15)

def prefer_far_exit_avoiding_the_cliff() -> Parameters:
    # Positive living reward, very high gamma to prioritize distant rewards, low noise to avoid cliff
    return Parameters(reward_live=0.01, gamma=0.99, noise=0.05)

def never_end_the_game() -> Parameters:
    # High positive living reward to keep the game going, gamma of 1 to not discount future rewards, moderate noise
    return Parameters(reward_live=0.1, gamma=1.0, noise=0.2)
