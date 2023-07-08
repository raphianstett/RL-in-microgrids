from MA3_learning import MA_QLearning
from MA3_learning import Baseline_MA3
from learning import QLearning

from environment import State
from environment import MDP as SMDP

from MA3_environment import MDP
from MA3_environment import Policy
from MA3_environment import Reward

from data import RealData
from MA_data import Data_3
from MA_data import Data_2

import matplotlib.pyplot as plt
import numpy as np
import os

### TRAIN MODELS
from MA3_variations.MA3_learning_with_diff import MA_QLearning as dMA_QLearning
from MA3_variations.MA3_environment_with_diff import MDP as dMDP3
from MA3_variations.MA3_environment_with_diff import Policy as dPolicy

# train MA on different models
def train_MA3(iterations):
    training_data, test_data = Data_2.split_data(Data_3.get_data(), 7)
    
    
    subfolder_name = 'Q_3MARL'
    os.makedirs(subfolder_name, exist_ok=True)
    for i,n in enumerate(iterations):
        
        # # 3 bins
        mdp3 = MDP(1000, 500, 500, 250, 12000, 3,3)
        Q_A3,Q_B3,Q_C3, rewards_per_episode = MA_QLearning.iterate(training_data,n,mdp3)

        # # 5 bins
        mdp5 = MDP(1000, 500, 500, 250, 12000, 5,5)
        Q_A5,Q_B5,Q_C5, rewards_per_episode = MA_QLearning.iterate(training_data,n,mdp5)

        # # 7 bins
        mdp7 = MDP(1000, 500, 500, 250, 12000, 7,7)
        Q_A7,Q_B7,Q_C7, rewards_per_episode = MA_QLearning.iterate(training_data,n, mdp7)
        
        # # 10 bins
        mdp10 = MDP(1000, 500, 500, 250, 12000, 10,10)
        Q_A10,Q_B10,Q_C10, rewards_per_episode = MA_QLearning.iterate(training_data,n, mdp10)


        # model with difference
        dmdp = dMDP3(1000,500,500,250,12000)
        dQ_A, dQ_B,dQ_C, rewards_per_episode = dMA_QLearning.iterate(training_data,n,dmdp)
        
        # # Define the file path within the subfolder
        file_path_A = os.path.join(subfolder_name, 'Q_A3_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B3_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C3_' + str(n)+ '.csv')
        np.savetxt(file_path_A, Q_A3, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B3, delimiter=',', fmt='%d')
        np.savetxt(file_path_C, Q_C3, delimiter=',', fmt='%d')


        file_path_A = os.path.join(subfolder_name, 'Q_A5_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B5_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C5_' + str(n)+ '.csv')
        np.savetxt(file_path_A, Q_A5, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B5, delimiter=',', fmt='%d')
        np.savetxt(file_path_C, Q_C5, delimiter=',', fmt='%d')

        file_path_A = os.path.join(subfolder_name, 'Q_A7_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B7_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C7_' + str(n)+ '.csv')        
        np.savetxt(file_path_A, Q_A7, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B7, delimiter=',', fmt='%d')
        np.savetxt(file_path_C, Q_C7, delimiter=',', fmt='%d')
        
        file_path_A = os.path.join(subfolder_name, 'Q_A10_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'Q_B10_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'Q_C10_' + str(n)+ '.csv')        
        np.savetxt(file_path_A, Q_A10, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, Q_B10, delimiter=',', fmt='%d')
        np.savetxt(file_path_C, Q_C10, delimiter=',', fmt='%d')
        
        file_path_A = os.path.join(subfolder_name, 'dQ_A_' + str(n)+ '.csv')
        file_path_B = os.path.join(subfolder_name, 'dQ_B_' + str(n)+ '.csv')
        file_path_C = os.path.join(subfolder_name, 'dQ_C_' + str(n)+ '.csv')
        np.savetxt(file_path_A, dQ_A, delimiter=',', fmt='%d')
        np.savetxt(file_path_B, dQ_B, delimiter=',', fmt='%d')
        np.savetxt(file_path_C, dQ_C, delimiter=',', fmt='%d')



# train_MA3([5,10,15])