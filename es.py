import numpy as np
import gymnasium as gym
np.random.seed(0)

#load environment
env_name = 'CartPole-v1'


class NeuralNetwork:

    def __init__(self):
        self.IL = 4 #input layer nodes
        self.HL = 50 #hidden layer nodes
        self.OL = 2 #output layer nodes

        self.tmp_w1 = np.random.randn(self.HL, self.IL) / np.sqrt(self.IL)
        self.tmp_w2 = np.random.randn(self.OL, self.HL) / np.sqrt(self.HL)
        self.num_weights1 = len(self.tmp_w1.flatten())
        self.num_weights2 = len(self.tmp_w2.flatten())

    def reshape_parameters(self, w):
        w1_list = w[:self.num_weights1]
        w2_list = w[self.num_weights1:]
        w1 = w1_list.reshape(self.tmp_w1.shape)
        w2 = w2_list.reshape(self.tmp_w2.shape)
        return [w1, w2]

    #forward propagation
    def predict(self, s, w_list):
        h = np.dot(w_list[0], s) #input to hidden layer
        h[h<0]=0 #relu
        out = np.dot(w_list[1], h) #hidden layer to output
        out = 1.0 / (1.0 + np.exp(-out)) #sigmoid 
        return out


# hyperparameters
npop: int = 50 # population size
sigma: float = 0.1 # noise standard deviation
alpha: float = 0.1 # learning rate
n_iter: int = 200 # number of iterations
simulation_num_episodes: int = 50
good_enough_fitness: float = 475 # early stop
neural_network = NeuralNetwork()


# the function we want to optimize
def f(w, test_env):

    w_list = neural_network.reshape_parameters(w)
    total_reward = 0

    for episode in range(simulation_num_episodes):
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

    return total_reward / simulation_num_episodes


def train(fitness, n_params, test_env):

    # Random initialization
    w = np.random.randn(n_params)

    best_w = w
    best_fitness = 0

    for i in range(n_iter):

        # if i % (n_iter // 20) == 0:
            # print('iter %d. w: %s, reward: %f' % (i, str(w), fitness(w)))
        print(f'iter {i}. reward: {fitness(w, test_env)}')

        # initialize memory for a population of w's, and their rewards
        N = np.random.randn(npop, n_params) # samples from a normal distribution N(0,1)
        R = np.zeros(npop)
        for j in range(npop):
            w_try = w + sigma*N[j] # jitter w using gaussian of sigma
            R[j] = fitness(w_try, test_env) # evaluate the jittered version

        # standardize the rewards to have a gaussian distribution
        A = (R - np.mean(R)) / np.std(R)
        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        w = w + alpha/(npop*sigma) * np.dot(N.T, A)

        # TODO: pick best one from the population! No need to run fitness again + just pick the best one ever found?
        f = fitness(w, test_env)
        if f > best_fitness:
            best_w, best_fitness = w, f

        if best_fitness >= good_enough_fitness:
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
    best_w, fitness = train(f, neural_network.num_weights1 + neural_network.num_weights2, test_env)
    test_env.close()

    print("done!")
    print("reward =", fitness)
    print("w =", best_w)

    show_to_humans(env_name, best_w)

main()


