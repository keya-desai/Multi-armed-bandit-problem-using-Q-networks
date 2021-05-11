import tensorflow as tf
import numpy as np
import random 
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from q_network import DQNModel
from double_q_network import DoubleQModel
from actor_critic import ActorCritic
from collections import defaultdict
import sys

class NormalBandit:
    
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

    def get_max_mean(self):
        return self.max_mean

    def get_mean(self, i):
        return self.mean_sd_list[i][0]

class NormalBanditWithSumZero:
    
    def __init__(self, k):
        """
        k: number of bandits 
        """
        self.k = k
        self.mean_sd_list = [] # Storing mean and sd of each bandit
        
        self.max_mean = 0
        self.max_i = 0
        self.mean_sum = 0
        
        for i in range(k):
            if i == k - 1:
                mean = -self.mean_sum
            else:
                mean = random.uniform(-1, 1)
            self.mean_sum += mean
            sigma = random.uniform(0, 2)
            self.mean_sd_list.append((mean, sigma))
            
            if mean > self.max_mean:
                self.max_mean = mean
                self.max_i = i
        # print("Mean sum check: ", self.mean_sum)
        
    def generate_reward(self, i):
        mu, sigma = self.mean_sd_list[i]
        return np.random.normal(mu, sigma)
    
    def generate_optimum_reward(self):
        return self.generate_reward(self.max_i)

    def get_max_mean(self):
        return self.max_mean

    def get_mean(self, i):
        return self.mean_sd_list[i][0]

class ExponentialBandit:

    def __init__(self, k):
        """
        k: number of bandits 
        """
        self.k = k
        # self.lambda_list = [] # Storing lambda of each bandit
        self.mean_list = []        
        self.max_mean = 0  # 1/lambda for each bandit
        self.max_i = 0

        # for i in range(k):
        #     _lambda = random.uniform(0, 1)  # Can change to -2 to 2
        #     self.lambda_list.append(_lambda)
            
        #     if _lambda != 0:
        #         mean = 1.0/_lambda 

        #     if mean > self.max_mean:
        #         self.max_mean = mean
        #         self.max_i = i

        for i in range(k):
            mean = random.uniform(0.01, 1)
            self.mean_list.append(mean)
            
            if mean > self.max_mean:
                self.max_mean = mean
                self.max_i = i
     
    def generate_reward(self, i):
        # _lambda = self.lambda_list[i]
        return np.random.exponential(self.mean_list[i])
            # Add Mixed gaussain and other distributions
    
    def generate_optimum_reward(self):
        return self.generate_reward(self.max_i)

    def get_max_mean(self):
        return self.max_mean

    def get_mean(self, i):
        return self.mean_list[i]

class GaussianMixtureBandit:
    
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
            p_1 = random.uniform(0, 1)
            p_2 = 1 - p_1
            sigma_1 = random.uniform(0, 2)
            sigma_2 = random.uniform(0, 2)
            p_1, p_2 = 0.5, 0.5
            self.mean_sd_list.append((mean, p_1, p_2, sigma_1, sigma_2))
            
            if mean > self.max_mean:
                self.max_mean = mean
                self.max_i = i
        
    def generate_reward(self, i):
        mu,  p_1, p_2, sigma_1, sigma_2 = self.mean_sd_list[i]
        return p_1*np.random.normal(mu, sigma_1) + p_2*np.random.normal(mu, sigma_2)
    
    def generate_optimum_reward(self):
        return self.generate_reward(self.max_i)

    def get_max_mean(self):
        return self.max_mean

    def get_mean(self, i):
        return self.mean_sd_list[i][0]


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

def compare_permutations(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta):
    
    test_bandit = NormalBandit(num_bandits)   
    s = [
    DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ,ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]

    print("Test bandits : ")
    for k,v in test_bandit.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))

    color = ['#4b6584', 
            '#8e44ad', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#f39c12', '#2c2c54']

    # ideal_regret = compute_min_regret(time_steps, test_bandit)
    # plt.plot(ideal_regret, '--' , color = color[0], label = "min")


    average_regret = s[0].estimate_average_regret(test_bandit)
    plt.plot(average_regret, color = color[1], label = "before")

    average_regret = s[0].estimate_average_regret_on_permutations(test_bandit)
    plt.plot(average_regret, '-.', color = color[1], label = "before (permutations)")


    for i in range(rounds):
        for j in range(episodes):
            bandit = NormalBandit(num_bandits) 
            s[0].train_on_one_pass(bandit)
        
        

        average_regret = s[0].estimate_average_regret(test_bandit)
        plt.plot(average_regret, color = color[i+2], label = str(i))
        
        average_regret = s[0].estimate_average_regret_on_permutations(bandit)
        plt.plot(average_regret, '-.', color = color[i+2], label = str(i))

        print("Round", str(i), "done.")

    plt.title('Permutations : Average regret v/s time steps for {} bandits. (epsilon = {}, beta = {}, episodes per round = {})'.format(num_bandits, epsilon, beta, episodes))
    plt.xlabel('Time steps')
    plt.ylabel('Average regret over {} trials'.format(trials))
    plt.legend(title = 'Round')
    plt.savefig('results/permutations_b{}_episodes{}'.format(num_bandits, episodes))
    plt.show()

def compare_bandits(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta):
    test_bandit_normal = NormalBandit(num_bandits)   
    test_bandit_exp = ExponentialBandit(num_bandits)   

    solvers = [
    DQNModel(test_bandit_normal, episodes, time_steps, trials, epsilon, beta)
    # ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ,ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]

    print("Test bandits (normal): ")
    str_normal = ""
    for k,v in test_bandit_normal.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))
        str_normal += "({:.3f}, {:.3f})".format(k, v)

    print("Test bandits (exponential): ")
    str_exp = ""
    for m in test_bandit_exp.mean_list:
        print("{:.3f}".format(m))
        str_exp += "({:.3f})".format(m)

    color = ['#4b6584', 
            '#8e44ad', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#f39c12', '#2c2c54']

    plt.figure(figsize=(10, 8))
    # min_regret = compute_min_regret(time_steps, test_bandit)
    # plt.plot(min_regret, '--', label = "Min regret")
    for s in solvers:
        average_regret = s.estimate_average_regret(test_bandit_exp)
        plt.plot(average_regret, color = color[0], label = "before (exp)")        
        average_regret = s.estimate_average_regret(test_bandit_normal)
        plt.plot(average_regret, '--', color = color[0], label = "before (normal)")


        for i in range(rounds):
            for j in range(episodes):
                bandit = NormalBandit(num_bandits) 
                s.train_on_one_pass(bandit)
            

            average_regret = s.estimate_average_regret(test_bandit_exp)
            plt.plot(average_regret, color = color[i+1], label = "Exp Round : {}".format(i))

            average_regret = s.estimate_average_regret(test_bandit_normal)
            plt.plot(average_regret, '--', color = color[i+1], label = "Normal Round : {}".format(i))

            print("Round", str(i), "done.")
            
        
        plt.title('Exponential : {}   Normal : {} \n Average regret v/s time steps for {} bandits. \n (e = {}, b = {}, episodes per round = {})'.format(str_exp, str_normal, num_bandits, epsilon, beta, episodes))
        plt.xlabel('Time steps')
        plt.ylabel('Average regret over {} trials'.format(trials))
        plt.legend(title = 'Round')
        plt.savefig('results/exp_normal_b{}_mean_regret_2'.format(num_bandits))
        plt.show()

def compare_mixed_gaussian(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta):
    test_bandit_normal = NormalBandit(num_bandits)   
    test_bandit_mixed_gaussian = GaussianMixtureBandit(num_bandits)   

    solvers = [
    DQNModel(test_bandit_normal, episodes, time_steps, trials, epsilon, beta)
    # ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ,ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]

    print("Test bandits (normal): ")
    str_normal = ""
    for k,v in test_bandit_normal.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))
        str_normal += "({:.3f}, {:.3f})".format(k, v)

    print("Test bandits (mixed gaussain): ")
    str_mixed = "mu, sigma_1, sigma_2 : "
    
    for mean, p_1, p_2, sigma_1, sigma_2 in test_bandit_mixed_gaussian.mean_sd_list:
        print("{:.3f}\t{:.3f}\t{:.3f}".format(mean, sigma_1, sigma_2))
        str_mixed += "({:.3f}, {:.3f}, {:.3f})".format(mean, sigma_1, sigma_2)

    color = ['#4b6584', 
            '#8e44ad', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#f39c12', '#2c2c54']

    plt.figure(figsize=(10, 8))
    # min_regret = compute_min_regret(time_steps, test_bandit)
    # plt.plot(min_regret, '--', label = "Min regret")
    for s in solvers:
        average_regret = s.estimate_average_regret(test_bandit_mixed_gaussian)
        plt.plot(average_regret, color = color[0], label = "before (Mixed Guassian)")        
        average_regret = s.estimate_average_regret(test_bandit_normal)
        plt.plot(average_regret, '--', color = color[0], label = "before (normal)")


        for i in range(rounds):
            for j in range(episodes):
                bandit = NormalBandit(num_bandits) 
                s.train_on_one_pass(bandit)
            

            average_regret = s.estimate_average_regret(test_bandit_mixed_gaussian)
            plt.plot(average_regret, color = color[i+1], label = "Mixed Guassian Round : {}".format(i))

            average_regret = s.estimate_average_regret(test_bandit_normal)
            plt.plot(average_regret, '--', color = color[i+1], label = "Normal Round : {}".format(i))

            print("Round", str(i), "done.")
            
        
        plt.title('Mixed guassian : {}  \n Normal : {} \n Average regret v/s time steps for {} bandits. (e = {}, b = {}, episodes per round = {})'.format(str_mixed, str_normal, num_bandits, epsilon, beta, episodes))
        plt.xlabel('Time steps')
        plt.ylabel('Average regret over {} trials'.format(trials))
        plt.legend(title = 'Round')
        plt.savefig('results/mixed_normal_b{}_mean_regret'.format(num_bandits))
        plt.show()

def compare_all(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta, replay_samples = 100):
    
    test_bandit = NormalBandit(num_bandits)   
    s = [
    DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta, replay_buffer = False, decaying_epsilon = True)
    ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta, decaying_epsilon = True)
    ,ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta, decaying_epsilon = True)
    ,DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta, replay_buffer = True)
    ]
    methods = {0: 'Q', 1: 'Double Q', 2: 'Actor Critic', 3 : 'Q with replay buffer'}

    str_normal = ""
    for k,v in test_bandit.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))
        str_normal += "({:.3f}, {:.3f}) ".format(k, v)

    color = ['#4b6584', 
            '#fc5c65', '#4b7bec', '#26de81', "#9b59b6", 
            '#eb3b5a', '#3867d6', '#20bf6b', "#8e44ad"]

    
    plt.figure(figsize=(10, 8))
    # ideal_regret = compute_min_regret(time_steps, test_bandit)
    # plt.plot(ideal_regret, '--' , color = color[0], label = "min")

    for method in range(len(s)):
        average_regret = s[method].estimate_average_regret(test_bandit)
        plt.plot(average_regret, '-.', color = color[method + 1], label = "before ({})".format(methods[method]))

    Min_Regret_dict = {0 : {'list' : [], 'mean' : sys.maxsize, 'round' : 0}}
    for i in range(1, len(s)):
        Min_Regret_dict[i] = {'list' : [], 'mean' : sys.maxsize, 'round' : 0}

    for i in range(rounds):
        for j in range(episodes):
            bandit = NormalBandit(num_bandits) 
            s[0].train_on_one_pass(bandit)
            s[1].train_on_one_pass(bandit)
            s[3].train_on_one_pass(bandit)
            if (j+1)%10 == 0:
                s[2].train_on_one_pass(bandit, update = True)
            else:
                s[2].train_on_one_pass(bandit, update = False)

            if (j+1) % 5 == 0:
                s[3].replay_buffer_fit(replay_samples)
        
        # average_regret = s.estimate_average_regret_on_permutations(bandit)

        for method in range(len(s)):
            average_regret = s[method].estimate_average_regret(test_bandit)
            method_dict = Min_Regret_dict[method]
            if np.mean(average_regret) > method_dict['mean']:
                continue
            method_dict['list'] = average_regret
            method_dict['mean'] = np.mean(average_regret)
            method_dict['round'] = i


        # print(Min_Regret_dict)
        # average_regret = s[0].estimate_average_regret(test_bandit)
        # plt.plot(average_regret, color = color[i+1], label = str((i+1)*episodes))
        
        # average_regret = s[1].estimate_average_regret(test_bandit)
        # plt.plot(average_regret, '--', color = color[i+1], label = str((i+1)*episodes))

        print("Round", str(i), "done.")

    for method in range(len(s)):
        min_regret_list = Min_Regret_dict[method]['list']
        min_round = Min_Regret_dict[method]['round']
        plt.plot(min_regret_list, color = color[method + len(s) + 1], label = methods[method] + " Rounds : {}".format(min_round))

    
    plt.title('Test bandit : {} \n Average regret v/s time steps for {} bandits. \n (epsilon = {}/t, beta = {}, episodes per round = {}, replay buffer samples = {})'.format(str_normal ,num_bandits, epsilon, beta, episodes, replay_samples))
    plt.xlabel('Time steps')
    plt.ylabel('Average regret over {} trials'.format(trials))
    plt.legend(title = 'Episode')
    plt.savefig('results/compare_all_b{}_episodes{}'.format(num_bandits, episodes))
    # plt.show()
        
def main(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta, replay_buffer = False, replay_samples = 100):
    
    test_bandit = NormalBandit(num_bandits)  
    
    solvers = [
    DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]

    min_regret = compute_min_regret(time_steps, test_bandit)
    for s in solvers:

        print("Test bandit:")
        for k,v in test_bandit.mean_sd_list:
            print("{:.3f}\t{:.3f}".format(k, v))

        average_regret = s.estimate_average_regret(test_bandit)
        plt.plot(average_regret, label = "before")

        for i in range(rounds):
            for j in range(episodes):
                bandit = NormalBandit(num_bandits) 
                s.train_on_one_pass(bandit)
                # if replay_buffer and (j+1) % 5 == 0:
                    # s.replay_buffer(replay_samples)
            
            average_regret = s.estimate_average_regret(test_bandit)

            print("Round", str(i), "done.")
            plt.plot(average_regret, label = str((i+1)*episodes))

        plt.plot(min_regret, '--', label = "Min regret")
        plt.title('Average regret v/s time steps for {} bandits. (epsilon = {}, beta = {})'.format(num_bandits, epsilon, beta))
        plt.xlabel('Time steps')
        plt.ylabel('Average regret over {} trials'.format(trials))
        plt.legend(title = 'Episode')
        plt.savefig('results/single_k{}_e{}_new'.format(num_bandits, episodes))
        plt.show()

def multiple_test_bandits(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta):
    
    num_test_bandits = 3
    test_bandits = [NormalBandit(num_bandits) for _ in range(num_test_bandits)]
    
    s = DQNModel(test_bandits[0], episodes, time_steps, trials, epsilon, beta)

    bandit_mean_str = []

    for i in range(num_test_bandits):
        print("\nTest bandit :", i+1)
        bandit = test_bandits[i]

        string = ""
        for k,v in bandit.mean_sd_list:
            print("{:.3f}\t{:.3f}".format(k, v))
            string += "({:.3f}, {:.3f})".format(k, v)
        bandit_mean_str.append(string)
    
        min_regret = compute_min_regret(time_steps, bandit)
        average_regret = s.estimate_average_regret(bandit)    
        plt.figure(i, figsize = (10, 8))
        plt.plot(min_regret, '--', label = "min")
        plt.plot(average_regret, label = "before")
    
    for i in range(rounds):
        for j in range(episodes):
            bandit = NormalBandit(num_bandits) 
            s.train_on_one_pass(bandit)

        for test_bandit_idx in range(num_test_bandits):
            average_regret = s.estimate_average_regret(test_bandits[test_bandit_idx])
            plt.figure(test_bandit_idx)
            plt.plot(average_regret, label = i)
        
        print("Round", str(i), "done.")

    # plt.plot(min_regret, '--', label = "Min regret")
    for test_bandit_idx in range(num_test_bandits):
        plt.figure(test_bandit_idx)
        plt.title(' Test bandit : {} \n Average regret v/s time steps for {} bandits. (e = {}, b = {}, episodes per round = {})'.format(bandit_mean_str[test_bandit_idx], num_bandits, epsilon, beta, episodes))
        plt.xlabel('Time steps')
        plt.ylabel('Average regret over {} trials'.format(trials))
        plt.legend(title = 'Round')
        plt.savefig('results/multi_b{}_e{}_{}_with_min'.format(num_bandits, episodes, test_bandit_idx))
    plt.show()

def compare_q_replay(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta, replay_samples = 2):
    
    test_bandit = NormalBandit(num_bandits)   
    s = [
    DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta, replay_buffer = False)
    ,DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta, replay_buffer = True)
    # ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ,ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]
    methods = {0: 'Q', 1: 'Q with Replay buffer'}

    # color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # for s in solvers:
    print("Test bandits : ")
    str_test = ""
    for k,v in test_bandit.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))
        str_test += "({:.3f}, {:.3f})".format(k, v)

    color = ['#4b6584', 
            '#8e44ad', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#f39c12', '#2c2c54']

    # ideal_regret = compute_min_regret(time_steps, test_bandit)
    plt.figure(figsize=(10, 8))
    # plt.plot(ideal_regret, '--' , color = color[0], label = "min")

    average_regret = s[0].estimate_average_regret(test_bandit)
    plt.plot(average_regret, '-.', color = color[0], label = "before")

    for i in range(rounds):
        for j in range(episodes):
            bandit = NormalBandit(num_bandits) 
            s[0].train_on_one_pass(bandit)
            s[1].train_on_one_pass(bandit)
            if (j+1) % 5 == 0:
                s[1].replay_buffer_fit(replay_samples)
        
        average_regret = s[0].estimate_average_regret(test_bandit)
        plt.plot(average_regret, color = color[i+1], label = "Q : Round {}".format(i))
        
        average_regret = s[1].estimate_average_regret(test_bandit)
        # print(average_regret)
        plt.plot(average_regret, '--', color = color[i+1], label = "Q with replay buffer : Round {}".format(i))

        print("Round", str(i), "done.")

    plt.title('Test bandit : {} \n Average regret v/s time steps for {} bandits. \n (epsilon = {}, beta = {}, episodes per round = {})'.format(str_test ,num_bandits, epsilon, beta, episodes))
    plt.xlabel('Time steps')
    plt.ylabel('Average regret over {} trials'.format(trials))
    plt.legend()
    plt.savefig('results/compare_q_replay_b{}_episodes{}'.format(num_bandits, episodes))
    plt.show()
        
def compare_replay(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta, replay_samples = 2):
    
    test_bandit = NormalBandit(num_bandits)   
    s = [
    DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta, replay_buffer = True)
    ,DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta, replay_buffer = True)
    # ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ,ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]
    methods = {0: 'Q Replay Buffer (samples = 100)', 1: 'Q with Replay buffer (samples = 250)'}

    # color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # for s in solvers:
    print("Test bandits : ")
    test_str = ""
    for k,v in test_bandit.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))
        test_str += "({:.3f}, {:.3f})".format(k, v)
    print(test_str)
    color = ['#4b6584', 
            '#8e44ad', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#f39c12', '#2c2c54']

    # ideal_regret = compute_min_regret(time_steps, test_bandit)
    plt.figure(figsize=(10, 8))
    # plt.plot(ideal_regret, '--' , color = color[0], label = "min")

    average_regret = s[0].estimate_average_regret(test_bandit)
    plt.plot(average_regret, '-.', color = color[0], label = "before")


    for i in range(rounds):
        for j in range(episodes):
            bandit = NormalBandit(num_bandits) 
            s[0].train_on_one_pass(bandit)
            s[1].train_on_one_pass(bandit)
            if (j+1) % 5 == 0:
                s[0].replay_buffer_fit(k = 100)
                s[1].replay_buffer_fit(k = 250)
        
        average_regret = s[0].estimate_average_regret(test_bandit)
        plt.plot(average_regret, '--', color = color[i+1], label = "Q RB (samples = 100): Round {}".format(i))
        
        average_regret = s[1].estimate_average_regret(test_bandit)
        # print(average_regret)
        plt.plot(average_regret, color = color[i+1], label = "Q RB (samples = 250) : Round {}".format(i))

        print("Round", str(i), "done.")

    plt.title('Test bandit : {} \n Average regret v/s time steps for {} bandits. \n (epsilon = {}, beta = {}, episodes per round = {})'.format(test_str, num_bandits, epsilon, beta, episodes))
    plt.xlabel('Time steps')
    plt.ylabel('Average regret over {} trials'.format(trials))
    plt.legend()
    plt.savefig('results/compare_replay_100_250_b{}_episodes{}'.format(num_bandits, episodes))
    plt.show()

def compare_decaying_epsilon(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta):
    
    test_bandit = NormalBandit(num_bandits)   
    s = [
    DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta, replay_buffer = False, decaying_epsilon = False)
    ,DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta, replay_buffer = False, decaying_epsilon = True)
    # ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ,ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]
    # methods = {0: 'Q Replay Buffer (samples = 100)', 1: 'Q with Replay buffer (samples = 250)'}

    # color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # for s in solvers:
    str_test = ""
    print("Test bandits : ")
    for k,v in test_bandit.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))
        str_test += "({:.3f},{:.3f})".format(k, v)

    color = ['#4b6584', 
            '#8e44ad', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#f39c12', '#2c2c54']

    # ideal_regret = compute_min_regret(time_steps, test_bandit)
    plt.figure(figsize=(10, 8))
    # plt.plot(ideal_regret, '--' , color = color[0], label = "min")

    average_regret = s[0].estimate_average_regret(test_bandit)
    plt.plot(average_regret, '-.', color = color[0], label = "before")


    for i in range(rounds):
        for j in range(episodes):
            bandit = NormalBandit(num_bandits) 
            s[0].train_on_one_pass(bandit)
            s[1].train_on_one_pass(bandit)
        
        average_regret = s[0].estimate_average_regret(test_bandit)
        plt.plot(average_regret,  color = color[i+1], label = "Q (epsilon = 0.2): Round {}".format(i))
        
        average_regret = s[1].estimate_average_regret(test_bandit)
        # print(average_regret)
        plt.plot(average_regret, '--', color = color[i+1], label = "Q (decaying epsilon = 0.2/t) : Round {}".format(i))

        print("Round", str(i), "done.")

    plt.title('Test bandit : {} \n Average regret v/s time steps for {} bandits. (beta = {}, episodes per round = {})'.format(str_test, num_bandits, beta, episodes))
    plt.xlabel('Time steps')
    plt.ylabel('Average regret over {} trials'.format(trials))
    plt.legend()
    plt.savefig('results/decaying_epsilon_b{}_episodes{}'.format(num_bandits, episodes))
    plt.show()

def compare_big_reward(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta):
    
    test_bandit = NormalBandit(num_bandits)   
    s = [
    DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta, replay_buffer = False, decaying_epsilon = False, big_reward = False)
    ,DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta, replay_buffer = False, decaying_epsilon = False, big_reward = True)
    # ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ,ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]
    # methods = {0: 'Q Replay Buffer (samples = 100)', 1: 'Q with Replay buffer (samples = 250)'}

    # color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # for s in solvers:
    str_test = ""
    print("Test bandits : ")
    for k,v in test_bandit.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))
        str_test += "({:.3f},{:.3f})".format(k, v)

    color = ['#4b6584', 
            '#8e44ad', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#f39c12', '#2c2c54']

    ideal_regret = compute_min_regret(time_steps, test_bandit)
    plt.figure(figsize=(10, 8))
    plt.plot(ideal_regret, '--' , color = color[0], label = "min")

    average_regret = s[0].estimate_average_regret(test_bandit)
    plt.plot(average_regret, '-.', color = color[0], label = "before")


    for i in range(rounds):
        for j in range(episodes):
            bandit = NormalBandit(num_bandits) 
            s[0].train_on_one_pass(bandit)
            s[1].train_on_one_pass(bandit)
        
        average_regret = s[0].estimate_average_regret(test_bandit)
        plt.plot(average_regret,  color = color[i+1], label = "Q: Round {}".format(i))
        
        average_regret = s[1].estimate_average_regret(test_bandit)
        # print(average_regret)
        plt.plot(average_regret, '--', color = color[i+1], label = "Q (big reward) : Round {}".format(i))

        print("Round", str(i), "done.")

    plt.title('Test bandit : {} \n Average regret v/s time steps for {} bandits. (epsilon = {}, beta = {}, episodes per round = {})'.format(str_test, num_bandits, epsilon, beta, episodes))
    plt.xlabel('Time steps')
    plt.ylabel('Average regret over {} trials'.format(trials))
    plt.legend()
    plt.savefig('results/big_reward_b{}_episodes{}'.format(num_bandits, episodes))
    plt.show()

def compare_bandit_with_sum_zero(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta):

    test_bandit_normal = NormalBandit(num_bandits)   
    test_bandit_sum_zero = NormalBanditWithSumZero(num_bandits)   

    s = [
    DQNModel(test_bandit_normal, episodes, time_steps, trials, epsilon, beta)       
    ,DQNModel(test_bandit_sum_zero, episodes, time_steps, trials, epsilon, beta)
    # ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ,ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]

    print("Test bandits (normal): ")
    str_normal = ""
    for k,v in test_bandit_normal.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))
        str_normal += "({:.3f}, {:.3f})".format(k, v)

    print("Test bandits (normal with sum 0): ")
    str_sum0 = ""
    for k,v in test_bandit_sum_zero.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))
        str_sum0 += "({:.3f}, {:.3f})".format(k, v)

    color = ['#4b6584', 
            '#8e44ad', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#f39c12', '#2c2c54']

    plt.figure(figsize=(10, 8))
    # min_regret = compute_min_regret(time_steps, test_bandit)
    # plt.plot(min_regret, '--', label = "Min regret")

   
    average_regret = s[0].estimate_average_regret(test_bandit_normal)
    plt.plot(average_regret, color = color[0], label = "before (normal)")
    
    average_regret = s[1].estimate_average_regret(test_bandit_sum_zero)
    plt.plot(average_regret,'--', color = color[0], label = "before (normal with sum 0)")     


    for i in range(rounds):
        for j in range(episodes):
            bandit = NormalBandit(num_bandits) 
            s[0].train_on_one_pass(bandit)

            bandit = NormalBanditWithSumZero(num_bandits)
            s[1].train_on_one_pass(bandit)
            
        average_regret = s[0].estimate_average_regret(test_bandit_normal)
        plt.plot(average_regret, color = color[i+1], label = "Normal bandit Round : {}".format(i))

        average_regret = s[1].estimate_average_regret(test_bandit_sum_zero)
        plt.plot(average_regret, '--', color = color[i+1], label = "Normal bandit with sum 0 Round : {}".format(i))

        print("Round", str(i), "done.")
            
        
    plt.title('Normal : {}   Normal w sum 0: {} \n Average regret v/s time steps for {} bandits. (e = {}, b = {}, episodes per round = {})'.format(str_normal, str_sum0 , num_bandits, epsilon, beta, episodes))
    plt.xlabel('Time steps')
    plt.ylabel('Average regret over {} trials'.format(trials))
    plt.legend(title = 'Round')
    plt.savefig('results/sum0_normal_b{}'.format(num_bandits))
    plt.show()



if __name__ == "__main__":
    num_bandits = 3
    rounds = 5
    
    trials = 10
    episodes = 10
    # episodes = 5

    time_steps = 100
    epsilon = 0.2
    beta = 0.9

    multiple_test_bandits(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta)
