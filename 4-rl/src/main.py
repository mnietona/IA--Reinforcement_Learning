from lle import LLE, World, Action
from rlenv.wrappers import TimeLimit
from qlearning import QLearning 
from approximate_qlearning import ApproximateQLearning
import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores, window_size=100):
    # Convertir la liste de listes en tableau numpy pour faciliter les calculs
    scores_array = np.array(scores)
    episodes = np.arange(scores_array.shape[1])
    
    # Calcul de la moyenne et de la déviation standard pour chaque épisode
    mean_scores = np.mean(scores_array, axis=0)
    std_scores = np.std(scores_array, axis=0)
    # Calcul de la moyenne mobile sur les scores moyens
    moving_average = np.convolve(mean_scores, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.scatter(episodes, mean_scores, label='Score Moyen par Épisode', alpha=0.4)
    plt.plot(episodes[window_size - 1:], moving_average, label='Moyenne Mobile', color='orange', linewidth=2)
    
    # Ajouter la déviation standard
    plt.fill_between(episodes, mean_scores - std_scores, mean_scores + std_scores, color='blue' , alpha=0.2)

    plt.title("Évolution du Score au Cours des Épisodes")
    plt.xlabel("Épisodes")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()


def train_agents_on_level(env, agents, episodes=1000):
    """ Entraîne les agents sur un niveau """
    epsilon = 1.0  # Commence avec une exploration complète
    epsilon_min = 0.1  # Valeur minimale pour epsilon
    epsilon_decay = 0.995  # Facteur de décroissance d'epsilon

    scores = []  # Pour suivre l'évolution du score

    for _ in range(episodes):
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

        # Mise à jour des agents avec le nouvel epsilon
        for agent in agents:
            agent.epsilon = epsilon
        
        #print(f"Episode: {_} - Score: {score}")

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




if __name__ == "__main__":
    
    level = 1
    
    if level == 1:
        level_name = "level1"
    elif level == 3:
        level_name = "level3"
    elif level == 6:
        level_name = "level6"
    
    env = TimeLimit(LLE.level(level), 80)
    #agents = [QLearning(id, alpha=0.1, gamma=0.9, epsilon=1.0) for id in range(env.n_agents)]
    agents =  [ApproximateQLearning(id, alpha=0.1, gamma=0.9, epsilon=1.0, n_actions=5, n_features=4) for id in range(env.n_agents)]
    print("Entraînement des agents...")
    score = []
    for entrainement in range(10):
        average_score, scores = train_agents_on_level(env, agents)
        score.append(scores)
        print(f"Entraînement {entrainement} terminé!")
    print("Entraînement terminé!")
    plot_scores(score)
    l_actions, total_score, total_gems = execute_actions(env, agents, epsilon=0)
    #visualize_actions(level_name, l_actions)

