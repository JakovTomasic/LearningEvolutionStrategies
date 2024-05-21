import numpy as np
import gymnasium as gym
from neural_network import NeuralNetwork
from hyperparameters import Hyperparameters
from es import train
np.random.seed(0)


env_name = 'CartPole-v1'
IL = 4 #input layer nodes
HL = 50 #hidden layer nodes
OL = 2 #output layer nodes
neural_network = NeuralNetwork([IL, HL, OL])
hyperparameters = Hyperparameters(
    npop = 50,
    sigma = 0.1,
    alpha = 0.1,
    n_iter = 200,
    simulation_num_episodes = 50,
    good_enough_fitness = 475,
)




def fitness_function(w, test_env, hyperparams):

    w_list = neural_network.reshape_parameters(w)
    total_reward = 0

    for episode in range(hyperparams.simulation_num_episodes):
        observation = test_env.reset()[0]
        #observe initial state
        while True:
            action = neural_network.predict(observation, w_list)
            action = np.argmax(action)
            #execute action
            observation_new, reward, terminated, truncated, _ = test_env.step(action)
            #collect reward
            total_reward += reward
            #update state
            observation = observation_new
            #end episode
            if terminated or truncated:
                break

    return total_reward / hyperparams.simulation_num_episodes


def show_to_humans(env_name, w):
    print("Running showcase...")
    showcase_env = gym.make(env_name, render_mode="human")
    observation, _ = showcase_env.reset()
    w_list = neural_network.reshape_parameters(w)

    showcase_reward = 0
    while True:
        action = neural_network.predict(observation, w_list)
        action = np.argmax(action)
        observation, reward, terminated, truncated, _ = showcase_env.step(action)
        showcase_reward += reward

        if terminated or truncated:
            if terminated:
                print(f'Dead! Reward = {showcase_reward}')
            else:
                print("Win!")
            observation, _ = showcase_env.reset()
            showcase_reward = 0

    showcase_env.close()



def main():
    test_env = gym.make(env_name)
    # reset() should (in the typical use case) be called with a seed right after initialization and then never again.
    test_env.reset(seed=0)
    best_w, fitness = train(fitness_function, neural_network.weights_count, test_env, hyperparameters)
    test_env.close()

    print("done!")
    print("reward =", fitness)
    print("w =", best_w)

    show_to_humans(env_name, best_w)

main()


