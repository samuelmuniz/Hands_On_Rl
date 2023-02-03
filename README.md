# Hands-On Reinforcement Learning

In this hands-on project, we will first implement a simple RL algorithm and apply it to solve the CartPole-v1 environment. Once we become familiar with the basic workflow, we will learn to use various tools for machine learning model training, monitoring, and sharing, by applying these tools to train a robotic arm.

## To be handed in

This work must be done individually. The expected output is a repository named `hands-on-rl` on https://gitlab.ec-lyon.fr. It must contain a `README.md` file that explains **briefly** the successive steps of the project. Throughout the subject, you will find a 🛠 symbol indicating that a specific production is expected.
The last commit is due before 11:59 pm on Monday, February 13, 2023. Subsequent commits will not be considered.

## Introduction to Gym

Gym is a framework for developing and evaluating reinforcement learning environments. It offers various environments, including classic control and toy text scenarios, to test RL algorithms.

### Installation

```sh
pip install gym==0.21
```

### Usage

Here is an example of how to use Gym to solve the `CartPole-v1` environment:

```python
import gym

# Create the environment
env = gym.make("CartPole-v1")

# Reset the environment and get the initial observation
observation = env.reset()

for _ in range(100):
    # Select a random action from the action space
    action = env.action_space.sample()
    # Apply the action to the environment 
    # Returns next observation, reward, done signal (indicating
    # if the episode has ended), and an additional info dictionary
    observation, reward, done, info = env.step(action)
    # Render the environment to visualize the agent's behavior
    env.render() 
```

## REINFORCE

The REINFORCE algorithm (also known as Vanilla Policy Gradient) is a policy gradient method that optimizes the policy directly using gradient descent. The following is the pseudocode of the REINFORCE algorithm:

```txt
Setup the CartPole environment
Setup the agent as a simple neural network with:
    - One fully connected layer with 128 units and ReLU activation followed by a dropout layer
    - One fully connected layer followed by softmax activation
Repeat 500 times:
    Reset the environment
    Reset the buffer
    Repeat until the end of the episode:
        Compute and store in the buffer the action probabilities 
        Sample the action based on the probabilities
        Step the environment with the action
        Compute and store in the buffer the return using gamma=0.99 
    Normalize the return
    Compute the policy loss as -sum(log(prob) * return)
    Update the policy using an Adam optimizer and a learning rate of 5e-3
```

🛠 Use PyTorch to implement REINFORCE and solve the CartPole environement. Share the code in `reinforce.py`, and share a plot showing the return accross episodes in the `README.md`.

## Familiarization with a complete RL pipeline: Application to training a robotic arm

In this section, you will use the Stable-Baselines3 package to train a robotic arm using RL. You'll get familiar with several widely-used tools for training, monitoring and sharing machine learning models.

### Get familiar with Stable-Baselines3

Stable-Baselines3 (SB3) is a high-level RL library that provides various algorithms and integrated tools to easily train and test reinforcement learning models.

#### Installation

```sh
pip install stable-baselines3[extra]
```

> ⚠️ If you use zsh as a shell, you'll need to use extra quote: `stable-baselines3"[extra]"`

#### Usage

Use the Stable-Baselines3 documentation and implement a code to solve the CartPole environment.

🛠 Store the code in `cartpole_sb3.py`. Unless otherwise state, you'll work upon this file for the next sections.

### Get familiar with Hugging Face Hub

Hugging Face Hub is a platform for easy sharing and versioning of trained machine learning models. With Hugging Face Hub, you can quickly and easily share your models with others and make them usable through the API. For example, see the trained A2C agent for CartPole: https://huggingface.co/sb3/a2c-CartPole-v1. Hugging Face Hub provides an API to download and upload SB3 models.

#### Installation of ̀ huggingface_sb3`

```sh
pip install huggingface_sb3 pyglet==1.5.1
```

#### Upload the model on the Hub

Follow the Hugging Face Hub documentation to upload the previously learned model to the Hub.

🛠 Link the trained model in the `README.md` file.

### Get familiar with Weights & Biases

Weights & Biases (W&B) is a tool for machine learning experiment management. With W&B, you can track and compare your experiments, visualize your model training and performance, and collaborate with your team.

#### Installation


```shell
pip install wandb
```

Use the documentation of Stable-Baselines3 and Weights & Biases to track the CartPole training. Make the run public.

🛠 Share the link of the wandb run in the `README.md` file.

### Get familiar with panda-gym

Panda-gym is a collection of environments for robotic simulation and control. It provides a range of challenges for training robotic agents in a simulated environment. In this section, you will get familiar with one of the environments provided by panda-gym, the PandaReachJointsDense-v2.

#### Installation

```shell
pip install panda_gym==2.0.0
```

#### Train, track, and share

Use the Stable-Baselines3 package to train A2C model on the `PandaReachJointsDense-v2` environment. 500k timesteps should be enough. Track the environment with Weights & Biases. Once the training is over, upload the trained model on the Hub.

🛠 Share all the code in `panda_gym_sb3.py`. Share the link of the wandb run and the trained model in the `README.md` file.

## Contribute

This tutorial may contain errors, inaccuracies, typos or areas for improvement. Feel free to contribute to its improvement by opening an issue.

## Author

Quentin Gallouédec