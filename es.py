import numpy as np
import gymnasium as gym
from neural_network import NeuralNetwork
np.random.seed(0)

class Hyperparameters:
    def __init__(self, npop: int, sigma: float, alpha: float, n_iter: int, simulation_num_episodes: int, good_enough_fitness: float):
        self.npop: int = npop # population size
        self.sigma: float = sigma # noise standard deviation
        self.alpha: float = alpha # learning rate
        self.n_iter: int = n_iter # number of iterations
        self.simulation_num_episodes: int = simulation_num_episodes
        self.good_enough_fitness: float = good_enough_fitness # early stop


#load environment
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


def train(fitness, n_params, test_env, hyperparams):

    # Random initialization
    w = np.random.randn(n_params)

    best_w = w
    best_fitness = 0

    for i in range(hyperparams.n_iter):

        # if i % (n_iter // 20) == 0:
            # print('iter %d. w: %s, reward: %f' % (i, str(w), fitness(w)))
        print(f'iter {i}. reward: {fitness(w, test_env, hyperparams)}')

        # initialize memory for a population of w's, and their rewards
        N = np.random.randn(hyperparams.npop, n_params) # samples from a normal distribution N(0,1)
        R = np.zeros(hyperparams.npop)
        for j in range(hyperparams.npop):
            w_try = w + hyperparams.sigma*N[j] # jitter w using gaussian of sigma
            R[j] = fitness(w_try, test_env, hyperparams) # evaluate the jittered version

        # standardize the rewards to have a gaussian distribution
        A = (R - np.mean(R)) / np.std(R)
        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        w = w + hyperparams.alpha/(hyperparams.npop*hyperparams.sigma) * np.dot(N.T, A)

        # TODO: pick best one from the population! No need to run fitness again + just pick the best one ever found?
        f = fitness(w, test_env, hyperparams)
        if f > best_fitness:
            best_w, best_fitness = w, f

        if best_fitness >= hyperparams.good_enough_fitness:
            break

    return best_w, best_fitness


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


