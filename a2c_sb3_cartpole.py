import gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import matplotlib.pyplot as plt
from huggingface_sb3 import push_to_hub

### Find File address directory PATH

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
print(dir_path)

#Initialize model configs
config = {
    "policy_type": "MlpPolicy", #Policy object that implements actor critic, using a MLP (2 layers of 64)
    "total_timesteps": 25000, #Time steps of 1/25000 s
    "env_name": "CartPole-v1", #enviroment Cartpole-V1
}

#wandb initialization
run = wandb.init(
    project="sb3-cartpole1", # Choose cartpole1 model
    config=config, # Upload configs
    sync_tensorboard=True,  # auto-upload sb3-carpole1's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

#create environment
def make_env():
    environment = gym.make(config["env_name"])
    environment = Monitor(environment)  # record stats such as returns
    return environment

#upload environment 
environment = DummyVecEnv([make_env])
model = A2C(config["policy_type"], environment, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

#Stop model learning
run.finish()
# Save the trained model database produced
model.save("a2cSb3Cartpole.zip")
# Load the trained model database produced
model = A2C.load("a2cSb3Cartpole.zip")


