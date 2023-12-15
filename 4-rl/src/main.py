from lle import LLE, ObservationType
from rlenv.wrappers import TimeLimit
from qlearning import QLearning 
from approximate_qlearning import ApproximateQLearning
import numpy as np
import matplotlib.pyplot as plt

def train_agents(env, agents, n_episodes):
    scores = []
    for episode in range(n_episodes):
        observation = env.reset()
        total_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            actions = [agent.choose_action(observation) for agent in agents]
            next_observation, reward, done, truncated, info = env.step(actions)

            for agent in agents:
                agent.update(observation, actions[agent.id], reward, next_observation, done)

            total_reward += reward
            observation = next_observation

        scores.append(total_reward)
        for agent in agents:
            agent.epsilon_decay(episode, n_episodes)

    return scores

def execute_actions(env, agents, epsilon=0):
    observation = env.reset()
    done, score, step = False, 0, 0
    for agent in agents:
        agent.epsilon = epsilon

    l_actions = []

    while not done and step < 25:  # < 25 car level 1 et 3 pour level 6 retirer
        actions = [agent.choose_action(observation) for agent in agents]
        next_observation, reward, done, _, info = env.step(actions)
        l_actions.append(actions)
        observation = next_observation
        score += reward
        step += 1

    gems = info.get('gems_collected', 0)
    print(f"Actions: {l_actions}")
    print(f"Step {step} - Score: {score}, Gems: {gems}")

def initialize_agents(env, AgentClass, n_agents, **kwargs):
    return [AgentClass(id, **kwargs) for id in range(n_agents)]

def plot_scores(scores1, scores2,level, window_size=100):
    # Convertir les listes de listes en tableaux numpy
    scores_array1 = np.array(scores1)
    scores_array2 = np.array(scores2)
    episodes = np.arange(1, scores_array1.shape[1] + 1)

    # Calcul de la moyenne et de la déviation standard pour chaque ensemble de scores
    mean_scores1 = np.mean(scores_array1, axis=0)
    std_scores1 = np.std(scores_array1, axis=0)
    mean_scores_window1 = np.convolve(mean_scores1, np.ones(window_size)/window_size, mode='valid')
    std_scores_window1 = np.convolve(std_scores1, np.ones(window_size)/window_size, mode='valid')

    mean_scores2 = np.mean(scores_array2, axis=0)
    std_scores2 = np.std(scores_array2, axis=0)
    mean_scores_window2 = np.convolve(mean_scores2, np.ones(window_size)/window_size, mode='valid')
    std_scores_window2 = np.convolve(std_scores2, np.ones(window_size)/window_size, mode='valid')

    # Tracer les graphiques pour les deux ensembles de scores
    plt.figure(figsize=(8, 5))
    plt.plot(episodes[:len(mean_scores_window1)], mean_scores_window1, label='Q-Learning Tabulaire - Moyenne des scores')
    plt.fill_between(episodes[:len(mean_scores_window1)], 
                     mean_scores_window1 - std_scores_window1, 
                     mean_scores_window1 + std_scores_window1, 
                     color='C0', alpha=0.2)

    plt.plot(episodes[:len(mean_scores_window2)], mean_scores_window2, label='Q-Learning Approximatif - Moyenne des scores', color='C1')
    plt.fill_between(episodes[:len(mean_scores_window2)], 
                     mean_scores_window2 - std_scores_window2, 
                     mean_scores_window2 + std_scores_window2, 
                     color='C1', alpha=0.2)

    # Ajout du titre et des légendes
    plt.title(f'Score Moyen par Épisode - Niveau {level}')
    plt.xlabel('Épisodes')
    plt.ylabel('Score Moyen')
    # en bas a droite
    plt.legend(fontsize='medium', loc='lower right', framealpha=0.8)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    level = 1
    n_episodes = 3000
    n_trainings = 10
    lscores_qlearning, lscores_aql = [], []
    alpha, gamma, epsilon = 0.1, 0.9, 1.0

    env = TimeLimit(LLE.level(level, ObservationType.LAYERED), 80)
    
    print(f'Entraînement sur le niveau {level}')

    for training in range(n_trainings):
        agents_qlearning = initialize_agents(env, QLearning, env.n_agents, alpha=alpha, gamma=gamma, epsilon=epsilon)
        agents_aql = initialize_agents(env, ApproximateQLearning, env.n_agents, n_features=14, alpha=alpha, gamma=gamma, epsilon=epsilon, n_actions=5)

        scores_qlearning = train_agents(env, agents_qlearning, n_episodes)
        scores_aql = train_agents(env, agents_aql, n_episodes)

        print(f"Entraînement {training}: Score moyen Q-Learning {np.mean(scores_qlearning)}, Approximate Q-Learning {np.mean(scores_aql)}")
        lscores_qlearning.append(scores_qlearning)
        lscores_aql.append(scores_aql)

    print("Entraînement terminé!")
    plot_scores(lscores_qlearning, lscores_aql, level)
    #execute_actions(env, agents_aql, epsilon=0)