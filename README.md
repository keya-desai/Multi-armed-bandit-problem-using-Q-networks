# CS-605 Bandit problem using Deep Q-Network (DQN)

This independent study is performed under Professor Charles Cowan, Rutgers University. 


Multi-armed bandit problem in reinforcement learning refers to devising a technique to optimize maximum reward gained at each step where the bandits follow a particular distribution. As part of our independent study, we understand the bandit problems, and the traditional methods for solving it such as UCB, and epsilon - greedy. In this work, we apply the reinforcement algorithms of classical Q-learning model and its variants of Deep Q-learning, Double DQN, Actor - critic, and deep Q-learning with replay buffer to the problem of multi-armed bandits. We present our experiments with various set ups and report the robustness of the models across different distributions of bandits. 


The code contains the following:
1. DQN : DQN model in Tensorflow 2 with variants such as Double DQN, Actor-critic
2. PolicyGradient : Basic Policy gradient model for bandit problem with loss = -log(pi)*A
3. Iterative Q learning approach
4. Multi-armed bandits : Mathematical approaches for bandit problems such as UCB, Epsilon-greedy, Naive approach
5. dqn_bandits_tf1_session_graph: DQN model for bandit problem in Tensorflow 1 
