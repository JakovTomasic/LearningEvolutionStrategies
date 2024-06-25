import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from neural_network import NeuralNetwork
from hyperparameters import Hyperparameters
from es import train
from fcmaes.optimizer import crfmnes, cmaes
from process_out import ProcessOut
import signal
import sys

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
additional_args = dict()
po = ProcessOut()

env_name = 'CartPole-v1'
IL = 4 #input layer nodes
HL = 50 #hidden layer nodes
OL = 2 #output layer nodes
neural_network = NeuralNetwork([IL, HL, OL], process_out_action=po.argmax_selection)
hyperparameters = Hyperparameters(
    npop = 50,
    sigma = 0.1,
    alpha = 0.1,
    n_iter = 200,
    simulation_num_episodes = 50,
    good_enough_fitness = 500,
)
SHOW_ALL_FITNESS_REWARDS = False
title_label = "CartPole Learning Curve - %d Episodes" % hyperparameters.simulation_num_episodes
title_label_exploration = "CartPole - Action space exploration vs. No action space exploration"


# env_name = 'Acrobot-v1'
# neural_network = NeuralNetwork([6, 25, 3], process_out_action=po.argmax_selection)
# hyperparameters = Hyperparameters(
#     npop = 30,
#     sigma = 0.2,
#     alpha = 0.1,
#     n_iter = 200,
#     simulation_num_episodes = 10,
#     good_enough_fitness = -150,
# )
# SHOW_ALL_FITNESS_REWARDS = False
# title_label = "Acrobot Learning Curve - %d Episodes" % hyperparameters.simulation_num_episodes
# title_label_exploration = "Acrobot - Action space exploration vs. No action space exploration"


# env_name = 'MountainCarContinuous-v0'
# neural_network = NeuralNetwork([2, 25, 1], process_out_action=po.direct_value)
# hyperparameters = Hyperparameters(
#     npop = 20,
#     sigma = 0.2,
#     alpha = 0.1,
#     n_iter = 200,
#     simulation_num_episodes = 10,
#     good_enough_fitness = 90,
# )
# SHOW_ALL_FITNESS_REWARDS = False
# title_label = "MountainCarContinuous Learning Curve - %d Episodes" % hyperparameters.simulation_num_episodes
# title_label_exploration = "MountainCarContinuous - \nAction space exploration vs. No action space exploration"


# env_name = 'LunarLander-v2'
# additional_args = dict(
#     continuous = False,
#     gravity = -10.0,
#     enable_wind = False,
#     wind_power = 15.0,
#     turbulence_power = 1.5
# )
# neural_network = NeuralNetwork([8, 25, 4], process_out_action=po.argmax_selection)
# hyperparameters = Hyperparameters(
#     npop = 30,
#     sigma = 0.2,
#     alpha = 0.1,
#     n_iter = 200,
#     simulation_num_episodes = 10,
#     good_enough_fitness = 225,
# )
# SHOW_ALL_FITNESS_REWARDS = False
# title_label = "LunarLander Learning Curve - %d Episodes" % hyperparameters.simulation_num_episodes
# title_label_exploration = "LunarLander - Action space exploration vs. No action space exploration"


# env_name = 'FrozenLake-v1'
# neural_network = NeuralNetwork([16, 25, 4], process_out_action=po.argmax_selection)
# additional_args = dict(
#     desc = None,
#     map_name = "4x4", 
#     is_slippery = False
# )
# hyperparameters = Hyperparameters(
#     npop = 50,
#     sigma = 0.2,  # probaj još 0.6 i 0.9
#     alpha = 0.1,
#     n_iter = 200,
#     simulation_num_episodes = 1,
#     good_enough_fitness = 1_000_000,
# )
# SHOW_ALL_FITNESS_REWARDS = False
# title_label = "FrozenLake Learning Curve - %d Episodes" % hyperparameters.simulation_num_episodes
# title_label_exploration = "FrozenLake - Action space exploration vs. No action space exploration"




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

def get_reward(observation):
    num_rows = 4
    goal = 15
    holes = [5, 7, 11, 12]

    if observation in holes:
        return 0
    
    if observation == goal:
        return 1_000_000
    
    current_row = int(observation/num_rows)
    current_column = observation - current_row*num_rows
    
    return (current_row + current_column)*5


def frozen_lake_fitness_function(w, test_env, hyperparams):
    w_list = neural_network.reshape_parameters(w)
    total_reward = 0

    for episode in range(hyperparams.simulation_num_episodes):
        observation = test_env.reset()[0]
        #observe initial state

        i = 0
        while i <= 10:
            observation_array = np.zeros(16)
            observation_array[observation] = 1.0
            action = neural_network.predict(observation_array, w_list)
            #execute action
            observation_new, _, terminated, _, _ = test_env.step(action)
            #collect reward
            total_reward += get_reward(observation_new)
            #update state
            observation = observation_new
            i += 1
            #end episode
            if terminated:
                break
            
    if SHOW_ALL_FITNESS_REWARDS:
        print(f'fitness_function reward = {total_reward}')

    return total_reward / hyperparams.simulation_num_episodes

open = True
is_preview_active = False

def signal_handler(sig, frame):
    global open
    global is_preview_active
    if is_preview_active:
        print('Closing the environment...\n\n')
        open = False
    else:
        sys.exit(0)

def show_to_humans(env_name, w):
    global open
    global is_preview_active
    print("Running showcase...")
    showcase_env = gym.make(env_name, **additional_args, render_mode="human")
    observation, _ = showcase_env.reset()
    w_list = neural_network.reshape_parameters(w)

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    showcase_reward = 0
    while open:
        is_preview_active = True
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
    open = True
    is_preview_active = False


def frozen_lake_show_to_humans(env_name, w):
    global open
    global is_preview_active
    print("Running showcase...")
    showcase_env = gym.make(env_name, **additional_args, render_mode="human")
    observation, _ = showcase_env.reset()
    w_list = neural_network.reshape_parameters(w)

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    showcase_reward = 0
    i = 0
    while open:
        is_preview_active = True
        observation_array = np.zeros(16)
        observation_array[observation] = 1.0

        action = neural_network.predict(observation_array, w_list)
        observation, _, terminated, _, _ = showcase_env.step(action)
        showcase_reward += get_reward(observation)
        i += 1

        if terminated or i > 10:
            if terminated:
                print(f'Terminated! Reward = {showcase_reward}')
            else:
                print(f'Truncated! Reward = {showcase_reward}')
            observation, _ = showcase_env.reset()
            showcase_reward = 0
            i = 0

    showcase_env.close()
    open = True
    is_preview_active = False


def plot_learning_curve(max_iteration, iteration_rewards):
    #label="Variance: %.2f" % np.var(iteration_rewards)
    plt.plot(np.arange(max_iteration), iteration_rewards, label="Variance: %.2f" % np.var(iteration_rewards))
    plt.xticks(np.arange(max_iteration+1, step=round(max_iteration/10)))
    plt.xlabel("Iterations")
    plt.ylabel("Total Reward / Num Episodes")
    plt.title(title_label)
    plt.legend()
    plt.show()

def plot_learning_curve_comparison(rewards_1, rewards_2):
    max_iteration = max(len(rewards_1), len(rewards_2))
    plt.plot(np.arange(len(rewards_1)), rewards_1, label="Without exploration (variance: %.2f)" % np.var(rewards_1))
    plt.plot(np.arange(len(rewards_2)), rewards_2, label="With exploration (variance: %.2f)" % np.var(rewards_2))
    plt.xticks(np.arange(0, max_iteration+1, step=round(max_iteration/10)))
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title(title_label_exploration)
    plt.legend()
    plt.show()
    

def main():
    test_env = gym.make(env_name, **additional_args)
    # Random initialization
    start_w = np.random.randn(neural_network.weights_count)
    
    ff = frozen_lake_fitness_function if env_name == 'FrozenLake-v1' else fitness_function

    # reset() should (in the typical use case) be called with a seed right after initialization and then never again.
    test_env.reset(seed=RANDOM_SEED)
    best_w, fitness, iteration_rewards, max_iteration = train(
        ff,          
        neural_network.weights_count, 
        test_env,
        hyperparameters, 
        start_w
    )
    test_env.close()

    print("done!")
    print("reward =", fitness)
    print("w =", best_w)
    print("\n")

    # plot_learning_curve(max_iteration, iteration_rewards)
    if env_name == 'FrozenLake-v1':
        frozen_lake_show_to_humans(env_name, best_w)
    else:
        show_to_humans(env_name, best_w)           

    ####################################################
    ###### Training with action space exploration #####

    if env_name == 'MountainCarContinuous-v0':
        neural_network.process_out = po.value_with_noise  
    else:
        neural_network.process_out = po.roullete_wheel_selection   

    if env_name == 'FrozenLake-v1':
        hyperparameters.simulation_num_episodes = 10  

    test_env.reset(seed=RANDOM_SEED)
    best_w, fitness, iteration_rewards_exploration, max_iteration_exploration = train(
        ff,
        neural_network.weights_count,
        test_env,
        hyperparameters,
        start_w
    )
    test_env.close()

    print("done!")
    print("reward =", fitness)
    print("w =", best_w)
    print("\n")

    if env_name != 'MountainCarContinuous-v0':
        print("Suboptimal actions chosen: ", po.suboptimal_action_chosen)
        print("Total number of actions: ", po.number_of_actions)
        print("Exploration ratio (%): ", po.suboptimal_action_chosen/po.number_of_actions)
        print("\n")

    # vrati na originalni selection radi prikazivanja:
    if env_name == 'MountainCarContinuous-v0':
        neural_network.process_out = po.direct_value   
    else:
        neural_network.process_out = po.argmax_selection  

    if env_name == 'FrozenLake-v1':
        frozen_lake_show_to_humans(env_name, best_w)
    else:
        show_to_humans(env_name, best_w)
    
    plot_learning_curve_comparison(iteration_rewards, iteration_rewards_exploration)
    ####################################################

main()

################## FCMA Lib ##########################

test_env = gym.make(env_name, **additional_args)
test_env.reset(seed=RANDOM_SEED)

evaluations = 0
max_iter = 0
iteration_rewards = []

ff = frozen_lake_fitness_function if env_name == 'FrozenLake-v1' else fitness_function

def update_fitness(w):
    global evaluations
    global max_iter

    ret_val = ff(w, test_env, hyperparameters)

    if(evaluations == hyperparameters.npop):
        max_iter += 1
        evaluations = 0
        iteration_rewards.append(ret_val)
        print("iteration", max_iter, ":", ret_val)

    evaluations += 1
    # print(evaluations, ret_val)

    return -ret_val

def main_cmaes():

    w = np.random.randn(neural_network.weights_count)

    res = cmaes.minimize(update_fitness, x0=w, input_sigma=hyperparameters.sigma, popsize=hyperparameters.npop, 
                            max_evaluations=hyperparameters.npop*hyperparameters.n_iter
                        )
    
    # res = crfmnes.minimize(update_fitness, x0=w, input_sigma=hyperparameters.sigma, popsize=hyperparameters.npop, 
    #                     max_evaluations=hyperparameters.npop*hyperparameters.n_iter
    #                 )

    test_env.close()

    print("done!")
    print(res)
    print("reward =", res.fun)
    print("w =", res.x)

    plot_learning_curve(max_iter, iteration_rewards)

    if env_name == 'FrozenLake-v1':
        frozen_lake_show_to_humans(env_name, res.x)
    else:
        show_to_humans(env_name, res.x)

# main_cmaes()

