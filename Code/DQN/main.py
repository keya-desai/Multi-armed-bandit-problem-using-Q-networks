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
    test_bandit = NormalBandit(num_bandits)   
    test_bandit_exp = ExponentialBandit(num_bandits)   

    solvers = [
    DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ,ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]

    # print("Test bandits (normal): ")
    # for k,v in test_bandit_normal.mean_sd_list:
        # print("{:.3f}\t{:.3f}".format(k, v))

    print("Test bandits (exponential): ")
    for m in test_bandit_exp.mean_list:
        print("{:.3f}".format(m))

    color = ['#4b6584', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#eb3b5a', '#3867d6', '#20bf6b']

    # min_regret = compute_min_regret(time_steps, test_bandit)
    for s in solvers:
        # average_regret = s.estimate_average_regret_on_permutations(test_bandit)
        plt.figure(figsize=(10, 8))
        # plt.plot(min_regret, '--', label = "Min regret")
        average_regret = s.estimate_average_regret(test_bandit)
        plt.plot(average_regret, label = "before")


        for i in range(rounds):
            for j in range(episodes):
                bandit = NormalBandit(num_bandits) 
                s.train_on_one_pass(bandit)
            
            # print(bandit.mean_sd_list)
            # for k,v in bandit.mean_sd_list:
            #     print("{:.3f}\t{:.3f}".format(k, v))
            # average_regret = s.estimate_average_regret_on_permutations(bandit)
            average_regret = s.estimate_average_regret(test_bandit_exp)

            print("Round", str(i), "done.")
            plt.plot(average_regret, label = str(i))

        
        plt.title('Exponential Test bandits : Average regret v/s time steps for {} bandits. (e = {}, b = {}, episodes per round = {})'.format(num_bandits, epsilon, beta, episodes))
        plt.xlabel('Time steps')
        plt.ylabel('Average regret over {} trials'.format(trials))
        plt.legend(title = 'Round')
        plt.savefig('results/exp_b{}_mu1_q'.format(num_bandits))
        plt.show()


def compare_main(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta):
    
    test_bandit = NormalBandit(num_bandits)   
    s = [
    DQNModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ,DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ,ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]
    methods = {0: 'Q', 1: 'Double Q', 2: 'Actor Critic'}

    # color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # for s in solvers:
    print("Test bandits : ")
    for k,v in test_bandit.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))

    color = ['#4b6584', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#eb3b5a', '#3867d6', '#20bf6b']

    # ideal_regret = compute_min_regret(time_steps, test_bandit)
    plt.figure(figsize=(10, 8))
    # plt.plot(ideal_regret, '--' , color = color[0], label = "min")

    for method in range(len(s)):
        average_regret = s[method].estimate_average_regret_on_permutations(test_bandit)
        plt.plot(average_regret, '-.', color = color[method + 1], label = "before ({})".format(methods[method]))

    Min_Regret_dict = {0 : {'list' : [], 'mean' : sys.maxsize, 'round' : 0}}
    for i in range(1, len(s)):
        Min_Regret_dict[i] = {'list' : [], 'mean' : sys.maxsize, 'round' : 0}

    for i in range(rounds):
        for j in range(episodes):
            bandit = NormalBandit(num_bandits) 
            s[0].train_on_one_pass(bandit)
            s[1].train_on_one_pass(bandit)
            if (j+1)%10 == 0:
                s[2].train_on_one_pass(bandit, update = True)
            else:
                s[2].train_on_one_pass(bandit, update = False)
        
        # average_regret = s.estimate_average_regret_on_permutations(bandit)

        for method in range(len(s)):
            average_regret = s[method].estimate_average_regret_on_permutations(test_bandit)
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
        plt.plot(min_regret_list, color = color[method + 4], label = methods[method] + " Rounds : {}".format(min_round))

    

    plt.title('Average regret v/s time steps for {} bandits. (epsilon = {}, beta = {}, episodes per round = {})'.format(num_bandits, epsilon, beta, episodes))
    plt.xlabel('Time steps')
    plt.ylabel('Average regret over {} trials'.format(trials))
    plt.legend(title = 'Episode')
    plt.savefig('results/compare_b{}_episodes{}_perm'.format(num_bandits, episodes))
    plt.show()
        

def main(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta, replay_buffer = False, replay_samples = 100):
    
    test_bandit = NormalBandit(num_bandits)  
    
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
                bandit = NormalBandit(num_bandits) 
                s.train_on_one_pass(bandit)
                if replay_buffer and (j+1) % 5 == 0:
                    s.replay_buffer(replay_samples)
            
            # print(bandit.mean_sd_list)
            # for k,v in bandit.mean_sd_list:
            #     print("{:.3f}\t{:.3f}".format(k, v))
            # average_regret = s.estimate_average_regret_on_permutations(bandit)
            average_regret = s.estimate_average_regret(test_bandit)

            print("Round", str(i), "done.")
            plt.plot(average_regret, label = str((i+1)*episodes))

        # plt.plot(min_regret, '--', label = "Min regret")
        plt.title('Average regret v/s time steps for {} bandits. (epsilon = {}, beta = {})'.format(num_bandits, epsilon, beta))
        plt.xlabel('Time steps')
        plt.ylabel('Average regret over {} trials'.format(trials))
        plt.legend(title = 'Episode')
        plt.savefig('results/replay_q_k{}_e{}'.format(num_bandits, episodes))
        plt.show()


def main_multip_test_bandits(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta):
    
    test_bandit_1 = NormalBandit(num_bandits) 
    test_bandit_2 = NormalBandit(num_bandits) 
    
    solvers = [
    DQNModel(test_bandit_1, episodes, time_steps, trials, epsilon, beta)
    # DoubleQModel(test_bandit, episodes, time_steps, trials, epsilon, beta)
    # ActorCritic(test_bandit, episodes, time_steps, trials, epsilon, beta)
    ]

    min_regret_1 = compute_min_regret(time_steps, test_bandit_1)
    for s in solvers:
        # average_regret = s.estimate_average_regret_on_permutations(test_bandit)
        print("\nTest bandit 1:")
        for k,v in test_bandit_1.mean_sd_list:
            print("{:.3f}\t{:.3f}".format(k, v))
        
        print("\nTest bandit 2:")
        for k,v in test_bandit_2.mean_sd_list:
            print("{:.3f}\t{:.3f}".format(k, v))


        average_regret = s.estimate_average_regret(test_bandit)
        plt.plot(average_regret, label = "before")

        for i in range(rounds):
            for j in range(episodes):
                bandit = NormalBandit(num_bandits) 
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
        # plt.savefig('results/min_r_b{}_mu1_trials_{}_q'.format(num_ba=ndits, trials))
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
    for k,v in test_bandit.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))

    color = ['#4b6584', 
            '#8e44ad', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#f39c12', '#2c2c54']

    # ideal_regret = compute_min_regret(time_steps, test_bandit)
    plt.figure(figsize=(10, 8))
    # plt.plot(ideal_regret, '--' , color = color[0], label = "min")

    average_regret = s[0].estimate_average_regret_on_permutations(test_bandit)
    plt.plot(average_regret, '-.', color = color[0], label = "before")


    for i in range(rounds):
        for j in range(episodes):
            bandit = NormalBandit(num_bandits) 
            s[0].train_on_one_pass(bandit)
            s[1].train_on_one_pass(bandit)
            if (j+1) % 5 == 0:
                s[1].replay_buffer_fit(replay_samples)
        
        average_regret = s[0].estimate_average_regret_on_permutations(test_bandit)
        plt.plot(average_regret, color = color[i+1], label = "Q : Round {}".format(i))
        
        average_regret = s[1].estimate_average_regret_on_permutations(test_bandit)
        # print(average_regret)
        plt.plot(average_regret, '--', color = color[i+1], label = "Q with replay buffer : Round {}".format(i))

        print("Round", str(i), "done.")

    plt.title('Average regret v/s time steps for {} bandits. (epsilon = {}, beta = {}, episodes per round = {})'.format(num_bandits, epsilon, beta, episodes))
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
    for k,v in test_bandit.mean_sd_list:
        print("{:.3f}\t{:.3f}".format(k, v))

    color = ['#4b6584', 
            '#8e44ad', 
            '#fc5c65', '#4b7bec', '#26de81', 
            '#f39c12', '#2c2c54']

    # ideal_regret = compute_min_regret(time_steps, test_bandit)
    plt.figure(figsize=(10, 8))
    # plt.plot(ideal_regret, '--' , color = color[0], label = "min")

    average_regret = s[0].estimate_average_regret_on_permutations(test_bandit)
    plt.plot(average_regret, '-.', color = color[0], label = "before")


    for i in range(rounds):
        for j in range(episodes):
            bandit = NormalBandit(num_bandits) 
            s[0].train_on_one_pass(bandit)
            s[1].train_on_one_pass(bandit)
            if (j+1) % 5 == 0:
                s[0].replay_buffer_fit(replay_samples = 100)
                s[1].replay_buffer_fit(replay_samples = 250)
        
        average_regret = s[0].estimate_average_regret_on_permutations(test_bandit)
        plt.plot(average_regret, '--', color = color[i+1], label = "Q RB (samples = 100): Round {}".format(i))
        
        average_regret = s[1].estimate_average_regret_on_permutations(test_bandit)
        # print(average_regret)
        plt.plot(average_regret, color = color[i+1], label = "Q RB (samples = 250) : Round {}".format(i))

        print("Round", str(i), "done.")

    plt.title('Average regret v/s time steps for {} bandits. (epsilon = {}, beta = {}, episodes per round = {})'.format(num_bandits, epsilon, beta, episodes))
    plt.xlabel('Time steps')
    plt.ylabel('Average regret over {} trials'.format(trials))
    plt.legend()
    plt.savefig('results/compare_replay_100_250_b{}_episodes{}'.format(num_bandits, episodes))
    plt.show()

if __name__ == "__main__":
    num_bandits = 3
    rounds = 3
    
    trials = 2
    episodes = 3
    # episodes = 5

    time_steps = 10
    epsilon = 0.2
    beta = 0.9
    # main(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta, replay_buffer = True, replay_samples = 2)
    compare_replay(num_bandits, time_steps, rounds, episodes, trials, epsilon, beta, replay_samples = 100)
