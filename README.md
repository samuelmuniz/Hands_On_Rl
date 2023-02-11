# BE 1 REINFORCE LEARNING wit GYM libraries

## GYM
Gym is a toolkit for developing and comparing reinforcement learning algorithms. It provides a variety of environments, such as classic control and toy text problems, that serve as a testbed for reinforcement learning algorithms. These environments are used to evaluate and compare the performance of different algorithms in a standardized way. The library has a simple interface that allows developers to easily create new environments, which makes it a versatile tool for experimentation and research in the field of reinforcement learning.

## pyTORCH
It provides a variety of tools and libraries for building, training, and testing deep neural networks. PyTorch is widely used in the fields of computer vision, natural language processing, and reinforcement learning. In reinforcement learning, PyTorch can be used to build neural networks that can be used as function approximators for value or policy functions.

## a2c_sb3_cartpole.py

This file is the implementation of the reinforcing learnin in the GYM model with the following specs :

"Setup the agent as a simple neural network with:
    - One fully connected layer with 128 units and ReLU activation followed by a dropout layer
    - One fully connected layer followed by softmax activation
Repeat 500 times:
    Reset the environment
    Reset the buffer
    Repeat until the end of the episode:
        Compute action probabilities 
        Sample the action based on the probabilities and store its probability in the buffer 
        Step the environment with the action
        Compute and store in the buffer the return using gamma=0.99 
    Normalize the return
    Compute the policy loss as -sum(log(prob) * return)
    Update the policy using an Adam optimizer and a learning rate of 5e-3
"

## A2C ALGORITHM

Advantage Actor-Critic (A2C) is a reinforcement learning algorithm that combines ideas from both the Actor-Critic and Q-Learning algorithms. It is an on-policy algorithm, which means that it learns from the actions taken in the environment and updates its policy accordingly.

In the A2C algorithm, the "Actor" part of the algorithm represents the policy, which maps states to actions. The "Critic" part of the algorithm represents the value function, which estimates the expected future reward for a given state or state-action pair. The advantage function is used to estimate the difference between the expected reward and the value of a state-action pair.

## ENVIROMENTS

In this BE we'll use two GYM enviroments in order to apply the A2C algorithm:

### CartPole-v1 (a2c_sb3_cartpole.py)

CartPole-v1 is a classic reinforcement learning environment that is widely used for testing and evaluating different reinforcement learning algorithms. It is included in the OpenAI Gym library, which provides a standardized interface for evaluating reinforcement learning algorithms.

### PandaReachJointsDense-v2 (a2c_sb3_panda_reach.py)

In the PandaReachJointsDense-v2 environment, a robot arm is tasked with reaching a target location in a three-dimensional workspace. The arm is equipped with seven joints and can move in a variety of ways to reach the target. The state of the environment is defined by the position and velocity of the arm joints, as well as the position of the target. The action space consists of the torques applied to each of the seven joints at each time step.

## Platforms
For a matter of simplicity, we'll use the following online platforms to apply the A2C algorithms:

### Hugging Face Hub

Hugging Face Hub is a platform developed by Hugging Face, a company that provides NLP tools and services. It is designed to make it easy for researchers, developers, and data scientists to use state-of-the-art NLP models in their projects.

### Weights and Bias

Weights & Biases is an online platform that provides tools for deep learning researchers, data scientists, and engineers. It is designed to make it easier to develop, train, and monitor deep learning models.

