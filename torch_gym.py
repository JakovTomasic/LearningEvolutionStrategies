import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork, sigmoid, CNNActionValue
from hyperparameters import Hyperparameters
from es import train
import torch
import torch.nn as nn
import torch.nn.functional as F

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)





# env_name = 'CartPole-v1'
# IL = 4 #input layer nodes
# HL = 50 #hidden layer nodes
# OL = 2 #output layer nodes
# neural_network = NeuralNetwork([IL, HL, OL], process_out_action=lambda x: np.argmax(sigmoid(x)))
# hyperparameters = Hyperparameters(
#     npop = 50,
#     sigma = 0.1,
#     alpha = 0.1,
#     n_iter = 200,
#     simulation_num_episodes = 50,
#     good_enough_fitness = 475,
# )
# SHOW_ALL_FITNESS_REWARDS = False

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

env_name = 'CarRacing-v2'
IMG_DIM = 84
neural_network = NeuralNetwork([IMG_DIM*IMG_DIM*4, 1024, 512, 256, 5], process_out_action=lambda x: np.argmax(sigmoid(x)))
hyperparameters = Hyperparameters(
    npop = 25,
    sigma = 0.1,
    alpha = 0.1,
    n_iter = 200,
    simulation_num_episodes = 1,
    good_enough_fitness = 90,
)
SHOW_ALL_FITNESS_REWARDS = True

FRAMES_TO_SKIP = 50
MAX_FRAMES_TO_TRAIN = 200
PREVIOUS_IMAGES_COUNT = 4

network = CNNActionValue(PREVIOUS_IMAGES_COUNT, 5)


def preprocess_observation(img):
    img = img[:84, 6:90] # CarRacing-v2-specific cropping
    # img = cv2.resize(img, dsize=(IMG_DIM, IMG_DIM))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

# def plot_image(image):
#     img = image[:84, 6:90] # CarRacing-v2-specific cropping
#     # img = cv2.resize(img, dsize=(IMG_DIM, IMG_DIM))
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
#     fig, axes = plt.subplots(1, 1, figsize=(20, 5))
#     axes.imshow(img, cmap='gray')
#     axes.axis('off')
#     plt.show()

def fitness_function(w, test_env, hyperparams):

    # w_list = neural_network.reshape_parameters(w)
    network.apply_weights_1d_to_model(w)
    total_reward = 0

    for episode in range(hyperparams.simulation_num_episodes):

        observation, _ = test_env.reset()
        for _ in range(FRAMES_TO_SKIP):
            observation, reward, terminated, truncated, _ = test_env.step(0)
            total_reward += reward
            # previous_frames.append(preprocess_observation(observation))

        s = preprocess_observation(observation)
        stacked_state = np.tile(s, (PREVIOUS_IMAGES_COUNT, 1, 1))  # [4, 84, 84]

        # plot_image(observation)

        #observe initial state
        index = 0
        while index < MAX_FRAMES_TO_TRAIN:
            # previous_frames = previous_frames[:PREVIOUS_IMAGES_COUNT]
            # predict_input = np.concatenate(previous_frames, axis=0)
            # action = neural_network.predict(predict_input, w_list)

            x = stacked_state
            x = torch.from_numpy(x).float().unsqueeze(0).to('cpu')
            q = network(x)
            action = torch.argmax(q).item()

            #execute action
            observation_new, reward, terminated, truncated, _ = test_env.step(action)
            #collect reward
            total_reward += reward

            #update state
            s = preprocess_observation(observation_new)
            stacked_state = np.concatenate((stacked_state[1:], s[np.newaxis]), axis=0)
            # previous_frames.append(preprocess_observation(observation_new))

            index += 1
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
    previous_frames = []

    showcase_reward = 0
    while True:
        if len(previous_frames) >= PREVIOUS_IMAGES_COUNT:
            previous_frames = previous_frames[:PREVIOUS_IMAGES_COUNT]
            predict_input = np.concatenate(previous_frames)
            action = neural_network.predict(predict_input, w_list)
        else:
            action = 0
        observation, reward, terminated, truncated, _ = showcase_env.step(action)
        previous_frames.append(preprocess_observation(observation))
        showcase_reward += reward

        if terminated or truncated:
            if terminated:
                print(f'Terminated! Reward = {showcase_reward}')
            else:
                print(f'Truncated! Reward = {showcase_reward}')
            observation, _ = showcase_env.reset()
            previous_frames = []
            showcase_reward = 0

    showcase_env.close()



def main():
    # test_env = gym.make(env_name, continuous=False, render_mode="human")
    test_env = gym.make(env_name, continuous=False)
    # reset() should (in the typical use case) be called with a seed right after initialization and then never again.
    test_env.reset(seed=RANDOM_SEED)
    best_w, fitness = train(fitness_function, neural_network.weights_count, test_env, hyperparameters)
    test_env.close()

    print("done!")
    print("reward =", fitness)
    print("w =", best_w)

    show_to_humans(env_name, best_w)

main()


