from lle import LLE
from rlenv.wrappers import TimeLimit
from qlearning import QLearning 

alpha = 0.1    
gamma = 0.9    
epsilon = 0.3
episodes = 10

env = TimeLimit(LLE.level(1), 5) 

Action = ["NORTH (UP)", "SOUTH (DOWN)", "EAST (RIGHT)","WEST (LEFT)", "STAY"]

agents = [QLearning(alpha, gamma, epsilon) for _ in range(env.n_agents)]
# Entrainment
print("Entrainement")
observation = env.reset()
done = truncated = False
score = 0

for episode in range(episodes):
    while not done :
        actions = [a.choose_action(observation) for a in agents]
        print(f"Actions : {Action[actions[0]]}")
        next_observation, reward, done, truncated, info = env.step(actions)
        print(f"Reward: {reward}, info: {info}, dones: {done}, truncated: {truncated}")

        for a in agents:
            a.update(observation, actions, reward, next_observation)
        
        observation = next_observation
        score += reward
    
    print(f"Episode {episode} - Score: {score}")

agents[0].print_q_table()