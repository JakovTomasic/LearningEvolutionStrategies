import numpy as np
import gymnasium as gym
np.random.seed(0)

#load environment
env_name = 'CartPole-v1'
env = gym.make(env_name)


#forward propagation
def predict(s,w1,w2):
    h = np.dot(w1,s) #input to hidden layer
    h[h<0]=0 #relu
    out = np.dot(w2,h) #hidden layer to output
    out = 1.0 / (1.0 + np.exp(-out)) #sigmoid 
    return out


# the function we want to optimize
def f(w):
    # solution = np.array([0.5, 0.1, -0.3])
    # # here we would normally:
    # # ... 1) create a neural network with weights w
    # # ... 2) run the neural network on the environment for some time
    # # ... 3) sum up and return the total reward
    # reward = -np.sum(np.square(solution - w))
    # return reward

    # hyperparameters
    num_episodes = 50

    w1_try, w2_try = reshape_parameters(w)
    total_reward = 0

    for episode in range(num_episodes):
        observation = env.reset()[0]
        #observe initial state
        while True:
            action = predict(observation, w1_try, w2_try)
            action = np.argmax(action)
            #execute action
            observation_new, reward, terminated, truncated, _ = env.step(action)
            #collect reward
            total_reward += reward
            #update state
            observation = observation_new
            #end episode
            if terminated or truncated:
                break

    return total_reward / num_episodes


def train(fitness, n_params):

    # hyperparameters
    npop: int = 50 # population size
    sigma: float = 0.1 # noise standard deviation
    alpha: float = 0.1 # learning rate
    n_iter: int = 200 # number of iterations

    # Random initialization
    w = np.random.randn(n_params)

    best_w = w
    best_fitness = 0

    for i in range(n_iter):

        # if i % (n_iter // 20) == 0:
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

        # TODO: pick best one from the population! No need to run fitness again + just pick the best one ever found?
        f = fitness(w)
        if f > best_fitness:
            best_w, best_fitness = w, f

        # TODO: just for testing. Remove!
        if best_fitness >= 300:
            break

    return best_w, best_fitness




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
observation, _ = showcase_env.reset()
w1_try, w2_try = reshape_parameters(best_w)

showcase_reward = 0
while True:
    action = predict(observation, w1_try, w2_try)
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













