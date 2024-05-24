import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork, sigmoid
from hyperparameters import Hyperparameters
from es import train

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
IMG_SIZE = 8
neural_network = NeuralNetwork([IMG_SIZE*IMG_SIZE*4, 256, 64, 5], process_out_action=lambda x: np.argmax(sigmoid(x)))
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

def is_green(pixel) -> bool:
    return pixel[1] > 150 #and pixel[1] > sum(pixel) / 2

def preprocess_internal(img):
    # img = img[:84, 6:90] # CarRacing-v2-specific cropping
    # img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE)) # or you can simply use rescaling
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    # return img
    
    img = img[:84, 6:90] # CarRacing-v2-specific cropping
    img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE)) # or you can simply use rescaling
    img2 = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # img2[i, j] = img[i, j][1]
            if is_green(img[i, j]):
                img2[i, j] = 10
            else:
                img2[i, j] = -10
    # img2 = cv2.resize(img2, dsize=(IMG_SIZE, IMG_SIZE)) # or you can simply use rescaling
    return img2

def preprocess_observation(img):
    return preprocess_internal(img).flatten()

def plot_image(image):
    img = preprocess_internal(image)
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].axis('off')
    axes[1].imshow(img, cmap='gray')
    axes[1].axis('off')
    plt.show()

def fitness_function(w, test_env, hyperparams):

    w_list = neural_network.reshape_parameters(w)
    total_reward = 0

    for episode in range(hyperparams.simulation_num_episodes):

        previous_frames = []

        observation, _ = test_env.reset()
        for _ in range(FRAMES_TO_SKIP):
            observation, reward, terminated, truncated, _ = test_env.step(0)
            total_reward += reward
            previous_frames.append(preprocess_observation(observation))

        # plot_image(observation)

        #observe initial state
        index = 0
        while index < MAX_FRAMES_TO_TRAIN:
            previous_frames = previous_frames[:4]
            predict_input = np.concatenate(previous_frames)
            action = neural_network.predict(predict_input, w_list)
            #execute action

            # action = 3
            observation_new, reward, terminated, truncated, _ = test_env.step(action)
            # if index % 50 == 0:
            #     plot_image(observation_new)

            #collect reward
            total_reward += reward
            #update state
            previous_frames.append(preprocess_observation(observation_new))
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
        if len(previous_frames) >= 4:
            previous_frames = previous_frames[:4]
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
    test_env = gym.make(env_name, continuous=False, render_mode="human")
    # test_env = gym.make(env_name, continuous=False)
    # reset() should (in the typical use case) be called with a seed right after initialization and then never again.
    test_env.reset(seed=RANDOM_SEED)
    best_w, fitness = train(fitness_function, neural_network.weights_count, test_env, hyperparameters)
    test_env.close()

    print("done!")
    print("reward =", fitness)
    print("w =", best_w)

    show_to_humans(env_name, best_w)

main()


