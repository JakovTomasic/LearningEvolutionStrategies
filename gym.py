import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from neural_network import NeuralNetwork, sigmoid
from hyperparameters import Hyperparameters
from es import train

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
additional_args = dict()

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
title_label = "CartPole Learning Curve - %d Episodes" % hyperparameters.simulation_num_episodes

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
# SHOW_ALL_FITNESS_REWARDS = False
# title_label = "Acrobot Learning Curve - %d Episodes" % hyperparameters.simulation_num_episodes


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
# title_label = "MountainCarContinuous Learning Curve - %d Episodes" % hyperparameters.simulation_num_episodes


# env_name = 'LunarLander-v2'
# additional_args = dict(
#     continuous = False,
#     gravity = -10.0,
#     enable_wind = False,
#     wind_power = 15.0,
#     turbulence_power = 1.5
# )
# neural_network = NeuralNetwork([8, 25, 4], process_out_action=lambda x: np.argmax(sigmoid(x)))
# hyperparameters = Hyperparameters(
#     npop = 10,
#     sigma = 0.2,
#     alpha = 0.1,
#     n_iter = 200,
#     simulation_num_episodes = 50,
#     good_enough_fitness = 225,
# )
# SHOW_ALL_FITNESS_REWARDS = False
# title_label = "LunarLander Learning Curve - %d Episodes" % hyperparameters.simulation_num_episodes





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


def plot_learning_curve(max_iteration, iteration_rewards):
    #label="Variance: %.2f" % np.var(iteration_rewards)
    plt.plot(np.arange(max_iteration+1), iteration_rewards, label="Variance: %.2f" % np.var(iteration_rewards))
    plt.xticks(np.arange(max_iteration+1, step=round(max_iteration/10)))
    plt.xlabel("Iterations")
    plt.ylabel("Total Reward / Num Episodes")
    plt.title(title_label)
    plt.legend()
    plt.show()
    

def main():
    test_env = gym.make(env_name, **additional_args)
    # reset() should (in the typical use case) be called with a seed right after initialization and then never again.
    test_env.reset(seed=RANDOM_SEED)
    best_w, fitness, iteration_rewards, max_iteration = train(fitness_function, neural_network.weights_count, test_env, hyperparameters)
    test_env.close()

    print("done!")
    print("reward =", fitness)
    print("w =", best_w)

    #plot_learning_curve(max_iteration, iteration_rewards)

    show_to_humans(env_name, best_w)

main()


