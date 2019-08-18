# Report

## The learning algorithm

The basics of this algorithm is described in https://arxiv.org/pdf/1509.02971.pdf

Nearly everything is used from my other DDPG repository: https://github.com/SibHusky/DDPG-Reacher-ContinuousControl
The main thing was to tune the hyperparameters.

---
MU = 0.0  
THETA = 0.175  
SIGMA = 0.01  

EPSILON = 1.0   
EPSILON_END = 0.001  (strictly speaking no hyperparameter, hard coded)  
EPSILON_DECAY = 0.995  
  
BUFFER_SIZE = 5e5   
BATCH_SIZE = 265  
GAMMA = 0.99  
TAU = 0.001  

UPDATE_EVERY = 20 (Learning frequence, in that case: the learning process runs every 100 timesteps)  
UPDATE_TIMES = 10 (how often the weights are updated by the learning process)  
START_LEARNING = 40 (The learning starts when the memory buffer has (BATCH_SIZE * START_LEARNING) entrys)  

The neural networks
---
The state is represented as vector. So a vanilla DNN is used.
Both networks use an Adam optimizer with a learning rate of 0.001.

The actor:
- batch normalization layer
- full connected layer 1: 33  --> 128 + relu activation function
  (33 is the size of the state vector)
- batch normalization layer
- full connected layer 2: 128 --> 128 + relu activation function
- batch normalization layer
- full connected layer 3: 128 --> 4 + tanh activation function
  (4 is the size of the action vector)
  

The critic:
- batch normalization layer
- full connected layer 1: 33 --> 128 + relu activation function
- full connected layer 2: 128 + 4 --> 128 + relu activation function
  (in this layer the actions are added. that's why +4)
- full connected layer 3: 128 --> 1



## Result and plots
