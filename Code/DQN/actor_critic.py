# -*- coding: utf-8 -*-
"""
Actor Critic model with TensorFlow 2
"""

import tensorflow as tf
import numpy as np
import random 
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from q_network import Solver

class ActorCritic(Solver):
	def __init__(self, bandit, episodes, time_steps, trials, epsilon, beta, decaying_epsilon):
		super().__init__(bandit)
		self.episodes = episodes
		self.time_steps = time_steps
		self.trials = trials
		self.epsilon = epsilon
		self.beta = beta
		self.decaying_epsilon = decaying_epsilon

		##############################################################################################
		layer_init = tf.keras.initializers.VarianceScaling()
		act_fn = tf.nn.relu
		num_bandits = bandit.k
		bandit_inputs = layers.Input( shape = (num_bandits,2) )
		flatten = layers.Flatten()( bandit_inputs )

		####################################     Actor network           #################################################################################
		dense_1a = layers.Dense( units = 100, activation = act_fn , kernel_initializer = layer_init, bias_initializer = layer_init )(flatten)
		dense_2a = layers.Dense( units = 50, activation = act_fn , kernel_initializer = layer_init, bias_initializer = layer_init )(dense_1a)
		Qa_vals = layers.Dense( units = num_bandits, activation = None , kernel_initializer = layer_init, bias_initializer = layer_init)(dense_2a)
		self.Qa_value_compute = tf.keras.Model( inputs = bandit_inputs, outputs = Qa_vals )

		#####################################    Critic Network              #############################################################################

		dense_1c = layers.Dense( units = 100, activation = act_fn , kernel_initializer = layer_init, bias_initializer = layer_init )(flatten)
		dense_2c = layers.Dense( units = 50, activation = act_fn , kernel_initializer = layer_init, bias_initializer = layer_init )(dense_1c)
		Qc_vals = layers.Dense( units = num_bandits, activation = None , kernel_initializer = layer_init, bias_initializer = layer_init)(dense_2c)
		self.Qc_value_compute = tf.keras.Model( inputs = bandit_inputs, outputs = Qc_vals )

		#########################################################################################################################################

		action_inputs = layers.Input( shape = (num_bandits,) )

		selected_Qa_value = layers.Dot( axes = 1 )( [ Qa_vals, action_inputs ] )
		self.Qa_value_selected = tf.keras.Model( inputs = [ bandit_inputs, action_inputs ], outputs = selected_Qa_value )
		self.Qa_value_selected.compile( optimizer = 'adam', loss = 'mean_squared_error' )

		selected_Qc_value = layers.Dot( axes = 1 )( [ Qc_vals, action_inputs ] )
		self.Qc_value_selected = tf.keras.Model( inputs = [ bandit_inputs, action_inputs ], outputs = selected_Qc_value )
		self.Qc_value_selected.compile( optimizer = 'adam', loss = 'mean_squared_error' )
		#########################################################################################################################################

	def train_on_one_pass( self, bandit, update = False):

		num_bandits = bandit.k
		current_state = np.zeros((num_bandits, 2))
		
		for t in range(self.time_steps):   

			if self.decaying_epsilon:
				self.epsilon = self.epsilon/(t+1)

			# Choose Action based on Actor network
			if random.random() < self.epsilon:
				action_selected = np.random.randint(num_bandits)
			else:
				q_values = self.Qa_value_compute.predict(np.asarray([current_state]))[0]
				action_selected = np.argmax( q_values )
			
			action_encoded = np.asarray( [ tf.keras.utils.to_categorical( action_selected, num_bandits ) ] )
			reward = bandit.generate_reward(action_selected)
			
			bandit_mean  = current_state[ action_selected ][0]
			bandit_count = current_state[ action_selected ][1]
			
			next_state = np.copy(current_state)
			next_state[ action_selected ][0] = (bandit_mean * bandit_count + reward) / (bandit_count + 1)
			next_state[ action_selected ][1] = bandit_count + 1
			
			# Update Critic network
			next_q_value = np.max( self.Qc_value_compute.predict( np.asarray( [ next_state ] ) )[ 0 ] )

			q_update_value = reward + self.beta * next_q_value
			state_input = np.asarray([current_state])
			
			self.Qc_value_selected.fit( [ state_input, action_encoded ],  np.asarray( [ q_update_value ] ) , epochs = 1, verbose = False )  
			
			current_state = np.copy(next_state)  

		if update:
			self.Qa_value_compute.set_weights(self.Qc_value_compute.get_weights())


	def generate_sample_regret_trajectory(self, bandit):

		max_reward = bandit.get_max_mean()
		num_bandits = bandit.k
		current_state = np.zeros((num_bandits, 2))
		rewards_generated = list()
		
		for t in range(self.time_steps):
			Qa_val = self.Qa_value_compute.predict( np.asarray( [ current_state ] ) )[ 0 ] 
			Qc_val = self.Qc_value_compute.predict( np.asarray( [ current_state ] ) )[ 0 ] 
			# Confirm once
			# q_values = (Qa_val + Qc_val)/2
			q_values = Qa_val
			action_selected = np.argmax( q_values )
			action_encoded = tf.keras.utils.to_categorical( action_selected, num_bandits )
			
			reward = bandit.generate_reward(action_selected)
			bandit_mean = bandit.get_mean(action_selected)
			rewards_generated.append( max_reward - bandit_mean )
			
			bandit_mean  = current_state[ action_selected ][0]
			bandit_count = current_state[ action_selected ][1]
			
			next_state = np.copy( current_state )
			next_state[ action_selected ][0] = (bandit_mean * bandit_count + reward) / (bandit_count + 1)
			next_state[ action_selected ][1] = bandit_count + 1
			
			current_state = np.copy(next_state)    
		return rewards_generated