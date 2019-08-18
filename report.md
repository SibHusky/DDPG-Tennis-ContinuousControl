# Report

## The learning algorithm

The basics of this algorithm is described in https://arxiv.org/pdf/1509.02971.pdf

Nearly everything is used from my other DDPG repository: https://github.com/SibHusky/DDPG-Reacher-ContinuousControl
The main thing was to tune the hyperparameters.

Hyperparameters
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

Episode 100	  Average100 Score: 0.01	Score: 0.10  
Episode 200  	Average100 Score: 0.01	Score: 0.00  
Episode 300  	Average100 Score: 0.01	Score: 0.00  
Episode 400  	Average100 Score: 0.00	Score: 0.00  
Episode 500  	Average100 Score: 0.00	Score: 0.00  
Episode 600  	Average100 Score: 0.01	Score: 0.00  
Episode 700	  Average100 Score: 0.01	Score: 0.00  
Episode 800  	Average100 Score: 0.00	Score: 0.00  
Episode 900	  Average100 Score: 0.00	Score: 0.00  
Episode 1000	Average100 Score: 0.02	Score: 0.10  
Episode 1100	Average100 Score: 0.06	Score: 0.00  
Episode 1200	Average100 Score: 0.04	Score: 0.10  
Episode 1300	Average100 Score: 0.10	Score: 0.09  
Episode 1400	Average100 Score: 0.11	Score: 0.20  
Episode 1500	Average100 Score: 0.15	Score: 0.20  
Episode 1559	Average100 Score: 0.51	Score: 2.10  
Environment solved! 	Average100 Score: 0.51	Episodes: 1559  

| <img src="https://github.com/SibHusky/DDPG-Tennis-ContinuousControl/blob/master/media/final_result_plot.png" width="391" height="262" /> |
|---|

| <img src="https://github.com/SibHusky/DDPG-Tennis-ContinuousControl/blob/master/media/tennis_untrained.gif" width="480" height="270" /> | <img src="https://github.com/SibHusky/DDPG-Tennis-ContinuousControl/blob/master/media/tennis_trained.gif" width="480" height="270" />  |
|---|---|
| Untrained agents | Trained agents |


## Ideas for the future work
- intensive hyperparameter tuning
- tune the neural network

