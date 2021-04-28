import tensorflow as tf
import numpy as np
import random 
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from q_network import DQNModel
from double_q_network import DoubleQModel
from actor_critic import ActorCritic

class Bandit:
    
    def __init__(self, k):
        """
        k: number of bandits 
        """
        self.k = k
        self.mean_sd_list = [] # Storing mean and sd of each bandit
        
        self.max_mean = 0
        self.max_i = 0
        
        # mu = [0.364, -0.48]
        # sd = [1.315, 0.072]
        for i in range(k):
            mean = random.uniform(-1, 1)
            sigma = random.uniform(0, 2)
            # mean = mu[i]
            # sigma = sd[i]
            self.mean_sd_list.append((mean, sigma))
            
            if mean > self.max_mean:
                self.max_mean = mean
                self.max_i = i
        
    def generate_reward(self, i):
        mu, sigma = self.mean_sd_list[i]
        return np.random.normal(mu, sigma)
    
    def generate_optimum_reward(self):
        return self.generate_reward(self.max_i)


def compute_min_regret(total_time, bandit):

    regret = []
    mu_max = bandit.max_mean

    for t in range(1, total_time+1):
        r = 0
        for i, (mu, sd) in enumerate(bandit.mean_sd_list):
            if mu == mu_max:
                continue
            r += 2 * sd**2/(mu_max - mu)
        r = r * np.log(t)
        regret.append(r)
    # print(regret)
    return regret

# def compare_main(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta):
    
#     test_bandit = Bandit(num_bandits)   
#     s = [
#     DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
#     ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
#     ActorCritic(bandit, episodes, time_steps, trials, epsilon, beta)
#     ]

#     color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#     # for s in solvers:
#     print("Test bandits : ")
#     for k,v in test_bandit.mean_sd_list:
#         print("{:.3f}\t{:.3f}".format(k, v))

#     average_regret = s[0].estimate_average_regret(test_bandit)
#     plt.plot(average_regret, color = color[0], label = "before")

#     average_regret = s[1].estimate_average_regret(test_bandit)
#     plt.plot(average_regret, '--', color = color[0], label = "before")

#     for i in range(rounds):
#         for j in range(episodes):
#             bandit = Bandit(num_bandits) 
#             s[0].train_on_one_pass(bandit)
#             s[1].train_on_one_pass(bandit)
#             if (j+1)%10 == 0:
#                 s[2].train_on_one_pass(bandit, update = True)
#             else:
#                 s[2].train_on_one_pass(bandit, update = False)
        
#         # print(bandit.mean_sd_list)
#         # for k,v in bandit.mean_sd_list:
#             # print("{:.3f}\t{:.3f}".format(k, v))
#         # average_regret = s.estimate_average_regret_on_permutations(bandit)
#         average_regret = s[0].estimate_average_regret(test_bandit)
#         plt.plot(average_regret, color = color[i+1], label = str((i+1)*episodes))
        
#         average_regret = s[1].estimate_average_regret(test_bandit)
#         plt.plot(average_regret, '--', color = color[i+1], label = str((i+1)*episodes))

#         print("Round", str(i), "done.")

#     plt.title('Average regret v/s time steps for {} bandits. (epsilon = {}, beta = {})'.format(num_bandits, epsilon, beta))
#     plt.xlabel('Time steps')
#     plt.ylabel('Average regret over {} trials'.format(trials))
#     plt.legend(title = 'Episode')
#     plt.savefig('results/dqn_doubleq_test_bandits_{}'.format(num_bandits, time_steps))
#     plt.show()
        


def main(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta):
    
    test_bandit = Bandit(num_bandits)  
    
    solvers = [
    DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]

    min_regret = compute_min_regret(time_steps, test_bandit)
    for s in solvers:
        # average_regret = s.estimate_average_regret_on_permutations(test_bandit)
        print("Test bandit:")
        for k,v in test_bandit.mean_sd_list:
            print("{:.3f}\t{:.3f}".format(k, v))

        average_regret = s.estimate_average_regret(test_bandit)
        plt.plot(average_regret, label = "before")

        for i in range(rounds):
            for j in range(episodes):
                bandit = Bandit(num_bandits) 
                s.train_on_one_pass(bandit)
            
            # print(bandit.mean_sd_list)
            # for k,v in bandit.mean_sd_list:
            #     print("{:.3f}\t{:.3f}".format(k, v))
            # average_regret = s.estimate_average_regret_on_permutations(bandit)
            average_regret = s.estimate_average_regret(test_bandit)

            print("Round", str(i), "done.")
            plt.plot(average_regret, label = str((i+1)*episodes))

        plt.plot(min_regret, '--', label = "Min regret")
        plt.title('Average regret v/s time steps for {} bandits. (epsilon = {}, beta = {})'.format(num_bandits, epsilon, beta))
        plt.xlabel('Time steps')
        plt.ylabel('Average regret over {} trials'.format(trials))
        plt.legend(title = 'Episode')
        plt.savefig('results/min_r_b{}_mu1_trials_{}_q'.format(num_bandits, trials))
        plt.show()




if __name__ == "__main__":
    num_bandits = 8
    rounds = 5
    time_steps = 100
    trials = 10
    episodes = 5
    epsilon = 0.2
    beta = 0.9
    main(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta)