# Navigation Project Report

TODO: change learning rates
TODO: rerun training set1 to get results.

## Framework description

* **DDPG**: DDPG or Deep Deterministic Policy Gradients is a model independent and off-policy actor-critic algorithm that uses deep neural networks to learn policies in high-dimensional, continuous action and state spaces. It consists of two networks. The Actor network is a policy based model that will take the state as input and directly output a policy, as in the optimal stochastic action probability distribution associated to that state. The Critic model is a value based model that has a similar structure to the **DQN framework** it will take the state as input and construct a Q valued function for the action space for that state. In order to deal with instability or divergence of the network weights, this framework is supported by the concepts of **Experience replay** and **Fixed Q-Targets** (by using a local and target network) developed by DeepMind.
* **OU Noise**:

## Implementation structure

The project is structured as follows:

* model.py : this file defines the model classes that contain the deep learning models' structures used to train the agents. It consists of 1 fully connected layer followed by a normalisation layer and then 2 fully connected layers. Apart from the normalised layer, it has the same structure as the one used to solve the OpenAI gym Pendulum environment.
* agent.py : this file defines the agent class that contains the functions used to train the agent.
* .ipynb files : Those files set up the environment and allow to train and watch the trained agents perform.
* .pth files: Those files saved the deep learning model weights in order to use a trained agent at any time without the need to retrain.
    
## Hyperparameters

Two different sets of hyperparameters were used to train the agents. THey are reported in Table 1 and 2 below. The next section will compare the difference in performance between the two sets of parameters.

  | Hyperparameter                      | Value |
  | ----------------------------------- | ----- |
  | Number of episodes                  | 1000  |
  | Average score to finish training    | 30.0  |
  | Max timesteps                       | 1000  |
  | Replay Buffer size                  | 1e5   |
  | Batch size                          | 128   |
  | Gamma                               | 0.99  |
  | Tau                                 | 1e-3  |
  | Learning rate Actor                 | 1e-4  |
  | Learning rate Critic                | 1e-3  |
  | Weight decay                     	| 0     |
  | Fully connected layers size         | 128   |
  | Mu_ou                               | 0     |
  | Theta_ou                            | 0.15  |
  | Sigma_ou                            | 0.1   |

The training will stop when the maximum number of episodes is reached or when the average score reaches 30.0.

## Results

| Hyperparameters set 1                                        | Hyperparameters set 2                         | Hyperparameters set 3                                         |
| ------------------------------------------ | ---------------------------------- | --------------------------------------------------- |
| ![set-1](report_files/dqn_performance.png)              | ![set-2](report_files/double_dqn.png)      | ![set-3](report_files/dueling_dqn.png)          |

The Double DQN is the fastest to converge at 493 episodes. The DQN is second at 537 episodes and the Dueling DQN comes last at 573 episodes.

## Discussion and Further improvements


