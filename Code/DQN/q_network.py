# -*- coding: utf-8 -*-
"""DQN Tf2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EwNM6ludPuzpbN3IeQMahp4x536fmE3W

DQN with TensorFlow 2 and Bandit Class integration
Answering -  Given basic Q learning, how many episodes do you need to effectively 
learn how to do bandits over 100 time steps?
"""


import tensorflow as tf
import numpy as np
import random 
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
  

    def generate_sample_regret_trajectory(self, bandit):
        pass

    def estimate_average_regret(self, bandit):
        '''
        Passing bandit because it could be the permuted_bandit(temp_bandit)
        '''
        regret_history = np.zeros(self.time_steps)
        for t in range(self.trials):
            regret_trajectory = self.generate_sample_regret_trajectory(bandit)
            regret_history += np.cumsum(regret_trajectory)
        
        return regret_history / self.trials

    def estimate_average_regret_on_permutations(self, bandit):
        num_bandits = bandit.k
        mu_sigma_dicts = [ defaultdict(tuple) for i in range(num_bandits)]
        for i in range(bandit.k):
            for j in range(num_bandits):
                mu_sigma_dicts[i][j] = bandit.mean_sd_list[ ( i + j ) % bandit.k ]

        average_regret = np.zeros(self.time_steps)
        
        for i in range(num_bandits):
          temp_bandit = bandit
          temp_bandit.mean_sd_list = mu_sigma_dicts[i]
          regret = self.estimate_average_regret(temp_bandit)
          average_regret += regret
        average_regret /= num_bandits
        return average_regret

    def train_on_one_pass(self, bandit):
        pass


class DQNModel(Solver):
    def __init__(self, bandit, episodes, time_steps, trials, epsilon, beta):
        super().__init__(bandit)
        self.episodes = episodes
        self.time_steps = time_steps
        self.trials = trials
        self.epsilon = epsilon
        self.beta = beta
        #########################################################################################################################################
        
        layer_init = tf.keras.initializers.VarianceScaling()
        act_fn = tf.nn.relu

        #########################################################################################################################################
        bandit_inputs = layers.Input( shape = (bandit.k,2) )
        flatten = layers.Flatten()( bandit_inputs )
        dense1 = layers.Dense( units = 100, activation = act_fn , kernel_initializer = layer_init, bias_initializer = layer_init )( flatten )
        dense2 = layers.Dense( units = 50, activation = act_fn , kernel_initializer = layer_init, bias_initializer = layer_init )( dense1 )
        Q_vals = layers.Dense( units = bandit.k, activation = None , kernel_initializer = layer_init, bias_initializer = layer_init )( dense2 )
        self.Q_value_compute = tf.keras.Model( inputs = bandit_inputs, outputs = Q_vals )
        #########################################################################################################################################


        #########################################################################################################################################
        action_inputs = layers.Input( shape = (bandit.k,) )
        selected_Q_value = layers.Dot( axes = 1 )( [ Q_vals, action_inputs ] )
        self.Q_value_selected = tf.keras.Model( inputs = [ bandit_inputs, action_inputs ], outputs = selected_Q_value )
        self.Q_value_selected.compile( optimizer = 'adam', loss = 'mean_squared_error' )
        #########################################################################################################################################

    def train_on_one_pass(self, bandit):
        # Q_value_compute, Q_value_selected
        num_bandits = bandit.k
        current_state = np.zeros((num_bandits, 2))
        
        for t in range(self.time_steps):
            
            if random.random() < self.epsilon:
                action_selected = np.random.randint(num_bandits)
            else:
                q_values = self.Q_value_compute.predict( np.asarray([current_state]))[0]
                action_selected = np.argmax(q_values)
            
            action_encoded = np.asarray( [ tf.keras.utils.to_categorical( action_selected,num_bandits) ] )
            reward = bandit.generate_reward(action_selected)
            
            bandit_mean  = current_state[ action_selected ][0]
            bandit_count = current_state[ action_selected ][1]
            
            next_state = np.copy( current_state )
            next_state[ action_selected ][0] = (bandit_mean * bandit_count + reward) / (bandit_count + 1)
            next_state[ action_selected ][1] = bandit_count + 1
            
            next_q_value = np.max(self.Q_value_compute.predict(np.asarray([next_state]))[0])
            q_update_value = reward + self.beta * next_q_value
            # print(q_update_value)

            if t == self.time_steps - 1:
                if bandit.mean_sd_list[action_selected][0] == bandit.get_max_mean():
                    q_update_value = 100
                else:
                    q_update_value = 0 
            
            state_input = np.asarray([current_state])
            
            self.Q_value_selected.fit( [ state_input, action_encoded ],  np.asarray( [ q_update_value ] ) , epochs = 1, verbose = False ) 
                   
            current_state = np.copy(next_state)  

    def generate_sample_regret_trajectory(self, bandit):
        # mean_list = [value[0] for value in mu_sigma_dict.values()]
        # max_reward = np.max( mean_list )
        max_reward = bandit.get_max_mean()
        num_bandits = bandit.k
        current_state = np.zeros((num_bandits, 2))
        rewards_generated = list()
        
        for t in range(self.time_steps):
            q_values = self.Q_value_compute.predict(np.asarray([current_state]))[0]
            action_selected = np.argmax(q_values)
            action_encoded = tf.keras.utils.to_categorical(action_selected, bandit.k)
            
            reward = bandit.generate_reward(action_selected)
            bandit_mean = bandit.mean_sd_list[action_selected][0]
            rewards_generated.append( max_reward - bandit_mean)
            
            bandit_mean  = current_state[ action_selected ][0]
            bandit_count = current_state[ action_selected ][1]
            
            next_state = np.copy( current_state )
            next_state[ action_selected ][0] = (bandit_mean * bandit_count + reward) / (bandit_count + 1)
            next_state[ action_selected ][1] = bandit_count + 1
            
            current_state = np.copy( next_state )    
        return rewards_generated
