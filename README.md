# L2RPN-using-A3C

Reinforcement Learning using the Actor-Critic framework for the L2RPN challenge (https://l2rpn.chalearn.org/ & https://competitions.codalab.org/competitions/22845#learn_the_details-overview). The agent trained using this code was one of the winners of the challenge. The code uses the pypownet environment (https://github.com/MarvinLer/pypownet). The code is released under a license of LGPLv3.

# Requirements
*   Python >= 3.6
*   Keras
*   pypownet 
*   Virtual Environment (conda/venv) Recommended

Pypownet Installation and Documentation : https://github.com/MarvinLer/pypownet

# Explaination of Files
- PDF Files
    - Amar_L2RPN_IJCNN_git.pdf - Presentation of the method at IJCNN-2019 in the L2RPN workshop. Summarizes the idea beind the approach and the training methodology. 
- Numpy Files
    - valid_actions_array_uniq.npz - matrix of valid unique actions 
    - valid_actions_masking_subid_perm.npz - matrix that maps the substation-Ids to the unique valid actions to be used for masking the output of the actor
- Python Files
    - valid_switching_controls.py - python file that creates the numpy files explained above
    - pypow_14_a3c_final.py - python file used to train the actor & critic neural networks using A3C 
- Chronic Datasets in public_data folder
    - datasets - Original chronics data given by the L2RPN contest
    - datasets_sub_4 - Subsampled chronics from the original data by 4
    - datasets_sub_7 - Subsampled chronics from original data by 7
    - you can create other subsamples by modifying the value of the 'sub_sample' value in the matlab file create_sub_files.m
    
# Usage
## Training your own A3C model
```
python pypow_14_a3c_final.py
```
This will create two new files 
  - pypow_14_a3c_actor.h5 - The weights of the actor neural network 
  - pypow_14_a3c_critic.h5 - The weights of the critic neural network

## Key Hyper-Parameter Tuning for Training 
To speed up the learning, the enviornment difficulty level is slowly increased and the following hyper-parameters in the code can be used to make the environment difficult or easy
  - game_level_global - chooses subsampled data so that the agents can see data from farther in the dataset
  - game_over_mode_global - controls the behavior of the lines in the environment
  - chronic_loop_mode_global - controls how the envirnment 'Reset' function will behave 
  

# License information
Copyright 2019 Amarsagar Reddy Ramapuram Matavalam

This Source Code is subject to the terms of the GNU Lesser General Public License v3.0. If a copy of the LGPL-v3 was not distributed with this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.html.
