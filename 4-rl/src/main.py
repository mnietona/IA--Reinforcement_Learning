from lle import LLE, World, Action, ObservationType
from rlenv.wrappers import TimeLimit
from qlearning import QLearning 
from approximate_qlearning import ApproximateQLearning
import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores,level, window_size=100):
    # Convertir la liste de listes en tableau numpy pour faciliter les calculs
    scores_array = np.array(scores)
    episodes = np.arange(scores_array.shape[1])
    
    # Calcul de la moyenne et de la déviation standard pour chaque épisode
    mean_scores = np.mean(scores_array, axis=0)
    std_scores = np.std(scores_array, axis=0)

    # Calcul de la moyenne et de la déviation standard sur la fenêtre glissante
    mean_scores_window = np.convolve(mean_scores, np.ones(window_size)/window_size, mode='valid')
    std_scores_window = np.convolve(std_scores, np.ones(window_size)/window_size, mode='valid')

    # Tracer le graphique
    plt.figure(figsize=(12, 6))
    plt.plot(episodes[:len(mean_scores_window)], mean_scores_window, label='Moyenne des scores')
    plt.fill_between(episodes[:len(mean_scores_window)], 
                     mean_scores_window - std_scores_window, 
                     mean_scores_window + std_scores_window, 
                     color='gray', alpha=0.2, label='Déviation standard')
    
    plt.title(f'Score Moyen par Épisode level {level}')
    plt.xlabel('Épisodes')
    plt.ylabel('Score Moyen')
    plt.legend()
    plt.grid()
    plt.savefig(f"plot_level_{level}.png")
    #plt.show()

def train_agents_on_level(env, agents, episodes=200):
    epsilon = 1.0  # Commence avec une exploration complète
    epsilon_min = 0.1  # Valeur minimale pour epsilon
    epsilon_decay = 0.995  # Facteur de décroissance d'epsilone
   
    scores = []
    for episode in range(episodes):
        observation = env.reset()
        done = truncated = False
        score = 0

        while not (done or truncated):

            actions = [agent.choose_action(observation) for agent in agents]
            
            next_observation, reward, done, truncated, info = env.step(actions)

            for agent in agents:
                agent.update(observation, actions, reward, next_observation,done)
                
            observation = next_observation
            score += reward

        scores.append(score)
        
        # #epsilon = max(epsilon_min, epsilon * epsilon_decay)
        # average_score = sum(scores) / len(scores)
        # for agent in agents:
        #     #agent.epsilon = epsilon
        #     agent.update_epsilon(episode, average_score)


        
        #print(f"Episode: {_} - Score: {score} - epsilon: {epsilon}")

    #print(f"epsilon: {epsilon}")
    average_score = sum(scores) / len(scores)
    print(f"Score moyen sur {episodes} épisodes: {average_score}")
    return average_score, scores

def execute_actions(env, agents, epsilon = 0):
    observation = env.reset()
    done  = False
    score = 0
    step = 0

    # Ajuster epsilon pour chaque agent
    for agent in agents:
        agent.epsilon = epsilon

    l_actions = []
    while not done:
        actions = [agent.choose_action(observation) for agent in agents]
        next_observation, reward, done, _, info = env.step(actions)
        l_actions.append(actions)
        observation = next_observation
        score += reward
        step += 1
        print(f"Step {step} - Score: {score}")
    
    gems = info['gems_collected']
    print(f"Step {step} - Score: {score}, Gems: {gems}")
    return l_actions, score, gems

def visualize_actions(level_file: str, l_actions: list[list[int]]):

    # Dictionnaire pour mapper les entiers aux objets Action correspondants.
    action_mapping = {
        0: Action.NORTH,
        1: Action.SOUTH,
        2: Action.EAST,
        3: Action.WEST,
        4: Action.STAY,
    }

    # Initialisation de l'état du monde à partir du fichier de niveau.
    w = World.from_file(level_file)
    w.reset()

    for actions_list in l_actions:
        for action_number in actions_list:
            # Convertir le numéro d'action en objet Action correspondant et le placer dans une liste.
            action_sequence = [action_mapping[action_number]]  # Création d'une liste d'actions
            w.step(action_sequence)
            img = w.get_image()
            cv2.imshow("Visualization", img)
            # Attendre que l'utilisateur appuie sur une touche pour passer à l'étape suivante.
            key = cv2.waitKey(500)

    # Attendre 1 seconde et fermer la fenêtre.
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

def scores_to_graph(level: int, scores: list[list[float]], args: tuple[float, float, float], min_val: int = 0, max_val: int = 1):
    """
    Plot the scores of the training episodes and save the graph as a png file
    """

    # Get the mean of the scores
    y = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)

    x = [*range(1, len(y) + 1)]

    # Plot the scores
    plt.scatter(x, y, s=0.5, color="blue", label="Mean Score")

    # Plot the standard deviation
    plt.fill_between(x, y - std_scores, y + std_scores, color="grey", alpha=0.2, label="Standard Deviation")

    # Plot the mean of the scores
    # y_window = np.convolve(y, np.ones(50) / 50, "valid")
    # plt.plot(x[: len(y_window)], y_window, color="red", label="Mean Score")

    # Plot the max and min values
    plt.axhline(y=max_val, color="green", linestyle="--", label=f"Optimal Score = {max_val}")
    plt.axhline(y=min_val, color="green", linestyle="--", label=f"Pessimistic Score = {min_val}")

    # Title
    title = f"Level {level} : Mean Score per Episode\n"
    # Add greec letters to the title
    #title += f"α={args[0]}, γ={args[1]}, ε₀={args[2]}"
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.show()
    plt.savefig("file.png")
    plt.clf()




if __name__ == "__main__":
    level = 1
    
    if level == 1:
        level_name = "level1"
    elif level == 3:
        level_name = "level3"
    elif level == 6:
        level_name = "level6"
    
    
    env = TimeLimit(LLE.level(level, ObservationType.LAYERED), 80)
    
    agents = [QLearning(id, alpha=0.1, gamma=0.9, epsilon=0.1) for id in range(env.n_agents)]
    
    #agents = [ApproximateQLearning(id, alpha=0.01, gamma=0.9 , epsilon=1.0, n_actions=5, n_features=9) for id in range(env.n_agents)]
    
    print(f'Entraînement sur le niveau {level_name} avec l\'algorithme {agents[0].__class__.__name__}')
    score = []
    for entrainement in range(100):
        average_score, scores = train_agents_on_level(env, agents)
        score.append(scores)
        print(f"Entraînement {entrainement} terminé!")
    print("Entraînement terminé!")
    
    scores_to_graph(level, score, (0.1, 0.7, 0.1), min_val=0, max_val=3)
    plot_scores(score,level)
    #l_actions, total_score, total_gems = execute_actions(env, agents, epsilon=0)
    #visualize_actions(level_name, l_actions)

