# Navigation Project Report

TODO: change learning rates

## Framework description

* **DDPG**:
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

The training will stop when the maximum number of episodes is reached or when the average score reaches 13.0.

## Results

| Hyperparameters set 1                                        | Hyperparameters set 2                         | Hyperparameters set 3                                         |
| ------------------------------------------ | ---------------------------------- | --------------------------------------------------- |
| ![set-1](report_files/dqn_performance.png)              | ![set-2](report_files/double_dqn.png)      | ![set-3](report_files/dueling_dqn.png)          |

The Double DQN is the fastest to converge at 493 episodes. The DQN is second at 537 episodes and the Dueling DQN comes last at 573 episodes.

## Discussion and Further improvements


