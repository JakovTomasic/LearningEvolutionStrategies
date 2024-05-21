
class Hyperparameters:
    def __init__(self, npop: int, sigma: float, alpha: float, n_iter: int, simulation_num_episodes: int, good_enough_fitness: float):
        self.npop: int = npop # population size
        self.sigma: float = sigma # noise standard deviation
        self.alpha: float = alpha # learning rate
        self.n_iter: int = n_iter # number of iterations
        self.simulation_num_episodes: int = simulation_num_episodes
        self.good_enough_fitness: float = good_enough_fitness # early stop

