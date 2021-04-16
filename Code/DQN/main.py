import tensorflow as tf
import numpy as np
import random 
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from q_network import DQNModel
from doubleQ_network import DoubleQModel

class Bandit:
    
    def __init__(self, k):
        """
        k: number of bandits 
        """
        self.k = k
        self.mean_sd_list = [] # Storing mean and sd of each bandit
        
        self.max_mean = 0
        self.max_i = 0
        
        for i in range(k):
            mean = random.uniform(-1, 1)
            sigma = random.uniform(0, 2)
            self.mean_sd_list.append((mean, sigma))
            
            if mean > self.max_mean:
                self.max_mean = mean
                self.max_i = i
        
    def generate_reward(self, i):
        mu, sigma = self.mean_sd_list[i]
        return np.random.normal(mu, sigma)
    
    def generate_optimum_reward(self):
        return self.generate_reward(self.max_i)




def main(num_bandits, time_steps, episodes, trials, epsilon, beta):
    
    bandit = Bandit(num_bandits)   
    solvers = [
    # DQNModel(bandit, episodes, time_steps, trials, epsilon, beta), 
    DoubleQModel(bandit, episodes, time_steps, trials, epsilon, beta)
    ]

    for s in solvers:
        average_regret = s.estimate_average_regret_on_permutations(bandit)
        plt.plot( average_regret, label = "before" )

        for i in range(trials):
            for j in range(time_steps):
                bandit = Bandit(num_bandits) 
                s.train_on_one_pass(bandit)
            
            average_regret = s.estimate_average_regret_on_permutations(bandit)

            print("Round", str(i), "done.")
            plt.plot( average_regret, label = str(i) )

        plt.legend()
        plt.show()

if __name__ == "__main__":
    num_bandits = 2
    # time_steps = 50
    # trials = 10
    time_steps = 3
    trials = 2
    episodes = 1
    epsilon = 0.2
    beta = 0.9
    main(num_bandits, time_steps, episodes, trials, epsilon, beta)