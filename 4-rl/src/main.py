from lle import LLE , Action, World# Assurez-vous que cela pointe vers votre implémentation de l'environnement LLE
from rlenv.wrappers import TimeLimit
from qlearning import QLearning  # Assurez-vous que cela pointe vers votre implémentation de QLearning


# Paramètres pour l'agent Q-learning
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 0.4 # Exploration rate

# Créer une instance temporaire de LLE pour obtenir les dimensions
temp_env = LLE.level(1)
max_time_steps = temp_env.width * temp_env.height
print(f"Nombre maximal d'étapes: {max_time_steps}")

env = TimeLimit(LLE.level(1), max_time_steps)  # Remplacer LLE.level(1) par votre environnement spécifique

# Création des agents
agents = [QLearning(env, alpha, gamma, epsilon) for _ in range(env.n_agents)]
# Boucle d'entraînement
observation = env.reset()
done = truncated = False
score = 0
while not (done or truncated):
    available_actions = env.available_actions()  # Tableau binaire des actions disponibles
    action_values = []
    for idx, agent in enumerate(agents):
        # Filtrer les actions disponibles pour cet agent
        agent_available_actions = [i for i, available in enumerate(available_actions[idx]) if available]
        
        # Choisir une action parmi les actions disponibles
        action_index = agent.choose_action(observation, agent_available_actions)
        
        # Convertir l'indice d'action en valeur entière de l'objet Action correspondant
        action = Action(action_index).value
        action_values.append(action)
    
    # Appeler step avec les valeurs entières des actions
    next_observation, reward, done, truncated, info = env.step(action_values)
    
    # Mise à jour de chaque agent
    for agent in agents:
        agent.update(observation, action, reward, next_observation)

    # Mise à jour du score et de l'observation
    score += reward
    observation = next_observation
    




