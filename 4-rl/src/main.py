from lle import LLE
from rlenv.wrappers import TimeLimit
from qlearning import QLearning 

alpha = 0.1    # Taux d'apprentissage
gamma = 0.9    # Facteur de remise
epsilon = 1.0  # Commence avec une exploration complète
epsilon_min = 0.1  # Valeur minimale pour epsilon
epsilon_decay = 0.995  # Facteur de décroissance d'epsilon
episodes = 5000

env = TimeLimit(LLE.level(1), 100)

Action = ["NORTH (UP)", "SOUTH (DOWN)", "EAST (RIGHT)", "WEST (LEFT)", "STAY"]

agents = [QLearning(alpha, gamma, epsilon) for _ in range(env.n_agents)]
# Entraînement
print("Entraînement")
scores = []  # Pour suivre l'évolution du score

for episode in range(episodes):
    observation = env.reset()
    done = truncated = False
    score = 0

    while not (done or truncated):
        actions = [agent.choose_action(observation) for agent in agents]

        next_observation, reward, done, truncated, info = env.step(actions)

        for agent in agents:
            agent.update(observation, actions, reward, next_observation)
        
        observation = next_observation
        score += reward

    scores.append(score)
    epsilon = max(epsilon_min, epsilon_decay * epsilon)  # Réduit epsilon
    #print(f"Episode {episode} - Score: {score}, Epsilon: {epsilon}")

    # Mise à jour des agents avec le nouvel epsilon
    for agent in agents:
        agent.epsilon = epsilon

# Affichage de la table Q de l'un des agents
#agents[0].print_q_table()

#Analyse du score
average_score = sum(scores) / len(scores)
print(f"Score moyen sur {episodes} épisodes: {average_score}")

# Joue le monde 

observation = env.reset()
done = truncated = False
score = 0
step = 0
gems = 0
agents[0].epsilon = 0.0

while not done:
    
    actions = [agent.choose_action(observation) for agent in agents]
    next_observation, reward, done, truncated, info = env.step(actions)

    observation = next_observation
    score += reward
    step += 1

    

print(f"Step {step} - Score: {score}, Gems: {info.get('gems_collected')}")

