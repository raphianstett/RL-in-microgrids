# Bachelor Thesis: Reinforcement Learning for Energy Management in Microgrids

This repository contains the code for my bachelor thesis. In the main folder the files for the three original microgrid models can be found. In the folder 'Variations' the models with adapted MDPs can be found. 

In general, the mcirogrid models are composed of the files environment, learning and testing. The environment files contain the MDP, the state space, the reward and the optimal policy evaluation. The learning files contain the Q-learning algorithm and the files for the original models also the baseline. The 2MARL and 3MARL models inherit the classes MDP and State from the single agent models. In the data.py file, the household data is imported and converted into the suitable data format for the RL algorithm. In the MA_data.py file, the data for households B and C is generated and brought into the correct format. 

To test the Q-learning algorithm, plots can be generated in the testing.py, MA2_testing.py, MA3_testing and test_model_comparison.py files. The first three test the performances on each microgrid. The test_model_comparison file generates the plots from the "Comparison of all Models" section from the thesis, where the microgrid models are compared. 
For most of the models and test cases, pre-trained Q-tables are available and stored in the SARL, 2MARL, 3MARL subfolders. Generated plots are automatically saved in the plots subfolder. Separate descriptions of the test cases and where Q-tables are available are noted in the respective files.

Sidenote: I used a slightly different connotation of the state-action value-function Q(S,A) in the code. The learning rate there does not only influences the temporal difference term but also the previous value of the state-action pair with (1-alpha). However, this has no effect on the interpretation of the learning rate in the thesis. 
