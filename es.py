import numpy as np
import gymnasium as gym
np.random.seed(0)

#load environment
env_name = 'CartPole-v1'
env = gym.make(env_name)

# the function we want to optimize
def f(w):
    # solution = np.array([0.5, 0.1, -0.3])
    # # here we would normally:
    # # ... 1) create a neural network with weights w
    # # ... 2) run the neural network on the environment for some time
    # # ... 3) sum up and return the total reward
    # reward = -np.sum(np.square(solution - w))
    # return reward

    #parameters
    max_iterations = 1000

    w1_try, w2_try = reshape_parameters(w)
    observation = env.reset()[0]
    #observe initial state
    total_reward = 0
    iter_index = 0
    while iter_index < max_iterations:
        iter_index += 1
        Action = predict(observation, w1_try, w2_try)
        Action = np.argmax(Action)
        #execute action
        observation_new, reward, done, _, _ = env.step(Action)
        #collect reward
        total_reward += reward
        #update state
        observation = observation_new
        #end episode
        if done:
            break

    return total_reward


def train(fitness, n_params):

    # hyperparameters
    npop: int = 50 # population size
    sigma: float = 0.01 # noise standard deviation
    alpha: float = 0.01 # learning rate
    n_iter: int = 300 # number of iterations

    # Random initialization
    w = np.random.randn(n_params)

    for i in range(n_iter):

        if i % (n_iter // 20) == 0:
            # print('iter %d. w: %s, reward: %f' % (i, str(w), fitness(w)))
            print(f'iter {i}. reward: {fitness(w)}')

        # initialize memory for a population of w's, and their rewards
        N = np.random.randn(npop, n_params) # samples from a normal distribution N(0,1)
        R = np.zeros(npop)
        for j in range(npop):
            w_try = w + sigma*N[j] # jitter w using gaussian of sigma
            R[j] = fitness(w_try) # evaluate the jittered version

        # standardize the rewards to have a gaussian distribution
        A = (R - np.mean(R)) / np.std(R)
        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        w = w + alpha/(npop*sigma) * np.dot(N.T, A)
        # print(f'wR = {fitness(w)}, max_R = {np.max(R)}, mean_R = {np.mean(R)}')

    return w, fitness(w)


#forward propagation
def predict(s,w1,w2):
    h = np.dot(w1,s) #input to hidden layer
    h[h<0]=0 #relu
    out = np.dot(w2,h) #hidden layer to output
    out = 1.0 / (1.0 + np.exp(-out)) #sigmoid 
    return out


# add hidden layers or nodes according to needs
IL = 4 #input layer nodes
HL = 50 #hidden layer nodes
OL = 2 #output layer nodes

tmp_w1 = np.random.randn(HL,IL) / np.sqrt(IL)
tmp_w2 = np.random.randn(OL,HL) / np.sqrt(HL)
NumWeights1 = len(tmp_w1.flatten())
NumWeights2 = len(tmp_w2.flatten())


def reshape_parameters(w):
    w1_list = w[:NumWeights1]
    w2_list = w[NumWeights1:]
    w1 = w1_list.reshape(tmp_w1.shape)
    w2 = w2_list.reshape(tmp_w2.shape)
    return w1, w2







best_w, fitness = train(f, NumWeights1 + NumWeights2)

env.close()

print("done!")
print("reward =", fitness)
print("w =", best_w)
print("Running showcase...")




showcase_env = gym.make(env_name, render_mode="human")
observation, info = showcase_env.reset()

while True:
    action = showcase_env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = showcase_env.step(action)

    if terminated or truncated:
        observation, info = showcase_env.reset()

showcase_env.close()













