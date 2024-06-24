import numpy as np
from neural_network import sigmoid

class ProcessOut:
    def __init__(self):
        self.number_of_actions = 0 
        self.suboptimal_action_chosen = 0 

    def argmax_selection(self, x):
        return np.argmax(sigmoid(x))

    def roullete_wheel_selection(self, x):
        argmax_action = np.argmax(sigmoid(x))
        roulette_action = self.__spin_roulette_wheel(sigmoid(x))

        self.number_of_actions += 1
        if(argmax_action != roulette_action):
            self.suboptimal_action_chosen += 1

        return roulette_action

    def direct_value(self, x):
        return x
    
    def value_with_noise(self, x):
        return self.__apply_gaussian_noise(x)


    def __spin_roulette_wheel(self, candidates):
        normalized_candidates = candidates/np.sum(candidates)
        roulette_wheel = np.cumsum(normalized_candidates, dtype=float)
        spin = np.random.rand()

        for action, candidate in enumerate(roulette_wheel):
            if(spin < candidate):
                return action
            
        return -1

    def __apply_gaussian_noise(self, x):
        noise_probability = 0.5
        sigma = 0.1

        p = np.random.rand()

        if(p < noise_probability):
            x = x + sigma*np.random.randn()

        return x