import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


# Enable NVIDIA CPU parallel processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
if device.type == 'cuda':
    print("CUDA enable")
    dtype = torch.float32

torch.set_default_dtype(dtype)

# Setup the CartPole environment in the GYM library
env = gym.make("CartPole-v1")

# Setup the agent as a simple neural network 
# One fully connected layer with 128 units and ReLU activation followed by a dropout layer
# One fully connected layer followed by softmax activation

class NeuralNetwork(nn.Module):

    def __init__(self, inputSize, hiddenSize, numbClasses):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hiddenSize, numbClasses)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim = 1)
        return x


# initialize the NN with 
# inputSize  = 4
# hiddenSize = 128
# numbClasses = 2
hiddenSize = 128
policy = NeuralNetwork(
                inputSize  = env.observation_space.shape[0], 
                hiddenSize = hiddenSize,
                numbClasses = env.action_space.n
            )
optimizer = optim.Adam(policy.parameters(), lr=5e-3)

# number of episodes 
episodes = 500
# discount value
gamma = 0.99
#list acumulative rewards by episode
rewardsCounter = []

#Initiate the loop without useless variable and shows its progress with "tqdm"
for _ in tqdm(range(episodes)):

    # Reset the environment and the buffer
    state = env.reset() 
    logProb = []
    #reward list by episode
    rewards = []
    done = False

    while not done:

        # Compute probabilites of move <left> or <right>
        state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
        #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # probability of move <left> or <right>
        actionProb = policy(state) 
        
        # Stores actions probabilities in the buffer
        m = torch.distributions.Categorical(actionProb)
        action = m.sample() # which side is the machine moving to 
        logProb.append(m.log_prob(action)) 
        
        # Step the environment with the action
        state, reward, done, _ = env.step(action.item())
        rewards.append(reward)

        # env.render() 

    rewardsCounter.append(len(rewards))

    # Compute and store the return (outcome)
    outcome = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        outcome.insert(0, R)
    outcome = torch.tensor(outcome)
    
    # Normalize the return outcome
    outcome = (outcome - outcome.mean()) / (outcome.std() + 1e-8)
    
    # Compute the policy loss
    policyLoss = []
    for log_prob, R in zip(logProb, outcome):
        policyLoss.append(-log_prob * R)
    policyLoss = torch.cat(policyLoss).sum()
    
    # Update the policy value for each iteration
    optimizer.zero_grad()
    policyLoss.backward()
    optimizer.step()

# Plot the accumlative rewards thoughout each episodes
plt.plot(rewardsCounter)
plt.title("CartPole")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

