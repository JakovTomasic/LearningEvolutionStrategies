import numpy as np
import math

def train(fitness, n_params, test_env, hyperparams):

    # Random initialization
    w = np.random.randn(n_params)
    # with open('filename.npy', 'rb') as file:
    #     w = np.load(file)


    best_w = w
    best_fitness = -math.inf

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
        A = (R - np.mean(R)) / np.std(R) if np.std(R) != 0 else np.zeros(hyperparams.npop)
        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        w = w + hyperparams.alpha/(hyperparams.npop*hyperparams.sigma) * np.dot(N.T, A)

        # TODO: pick best one from the population! No need to run fitness again + just pick the best one ever found?
        f = fitness(w, test_env, hyperparams)
        if f > best_fitness:
            best_w, best_fitness = w, f
        
        with open(f'result_{i}_f_{f}.npy', 'wb') as file:
            np.save(file, w)


        if best_fitness >= hyperparams.good_enough_fitness:
            break

    return best_w, best_fitness
