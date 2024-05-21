import numpy as np
import gymnasium as gym
from neural_network import NeuralNetwork, sigmoid
from hyperparameters import Hyperparameters
from es import train

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)


env_name = 'CartPole-v1'
IL = 4 #input layer nodes
HL = 50 #hidden layer nodes
OL = 2 #output layer nodes
neural_network = NeuralNetwork([IL, HL, OL], process_out_action=lambda x: np.argmax(sigmoid(x)))
hyperparameters = Hyperparameters(
    npop = 50,
    sigma = 0.1,
    alpha = 0.1,
    n_iter = 200,
    simulation_num_episodes = 50,
    good_enough_fitness = 475,
)
SHOW_ALL_FITNESS_REWARDS = False

# env_name = 'Acrobot-v1'
# neural_network = NeuralNetwork([6, 25, 3], process_out_action=lambda x: np.argmax(sigmoid(x)))
# hyperparameters = Hyperparameters(
#     npop = 10,
#     sigma = 0.2,
#     alpha = 0.1,
#     n_iter = 200,
#     simulation_num_episodes = 10,
#     good_enough_fitness = -150,
# )
# SHOW_ALL_FITNESS_REWARDS = True

# env_name = 'MountainCarContinuous-v0'
# neural_network = NeuralNetwork([2, 25, 1], process_out_action=lambda x: x)
# hyperparameters = Hyperparameters(
#     npop = 10,
#     sigma = 0.2,
#     alpha = 0.1,
#     n_iter = 200,
#     simulation_num_episodes = 10,
#     good_enough_fitness = 90,
# )
# SHOW_ALL_FITNESS_REWARDS = False





def fitness_function(w, test_env, hyperparams):

    w_list = neural_network.reshape_parameters(w)
    total_reward = 0

    for episode in range(hyperparams.simulation_num_episodes):
        observation = test_env.reset()[0]
        #observe initial state
        while True:
            action = neural_network.predict(observation, w_list)
            #execute action
            observation_new, reward, terminated, truncated, _ = test_env.step(action)
            #collect reward
            total_reward += reward
            #update state
            observation = observation_new
            #end episode
            if terminated or truncated:
                break

    if SHOW_ALL_FITNESS_REWARDS:
        print(f'fitness_function reward = {total_reward}')

    return total_reward / hyperparams.simulation_num_episodes


def show_to_humans(env_name, w):
    print("Running showcase...")
    showcase_env = gym.make(env_name, render_mode="human")
    observation, _ = showcase_env.reset()
    w_list = neural_network.reshape_parameters(w)

    showcase_reward = 0
    while True:
        action = neural_network.predict(observation, w_list)
        observation, reward, terminated, truncated, _ = showcase_env.step(action)
        showcase_reward += reward

        if terminated or truncated:
            if terminated:
                print(f'Terminated! Reward = {showcase_reward}')
            else:
                print(f'Truncated! Reward = {showcase_reward}')
            observation, _ = showcase_env.reset()
            showcase_reward = 0

    showcase_env.close()



def main():
    test_env = gym.make(env_name)
    # reset() should (in the typical use case) be called with a seed right after initialization and then never again.
    test_env.reset(seed=RANDOM_SEED)
    best_w, fitness = train(fitness_function, neural_network.weights_count, test_env, hyperparameters)
    test_env.close()

    print("done!")
    print("reward =", fitness)
    print("w =", best_w)

    show_to_humans(env_name, best_w)

main()


