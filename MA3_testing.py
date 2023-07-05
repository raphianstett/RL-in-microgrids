from MA3_learning import MA_QLearning
from MA3_learning import Baseline_MA3
from learning import QLearning

from environment import State
from environment import MDP as SMDP
from MA3_environment import Reward
from MA3_environment import MDP
from MA3_environment import Policy
from MA3_environment import Reward

from data import RealData
from MA_data import Data_3
from MA_data import Data_2

import matplotlib.pyplot as plt
import numpy as np
from learning import Baseline

data = Data_3()
data2 = Data_2()
df = data.get_data()
# print(df)
df_summer = RealData.get_summer(df)
# print(df_summer)
df_training, test = RealData.split_data(df_summer, 7)

        
mdp = MDP(1000, 500, 500, 250, 12000, 7,7)
costs_A,costs_B,costs_C, actions_A, actions_B, actions_C, battery, diffb, diffb2 = Baseline_MA3.find_baseline(test, mdp)
print(np.sum(costs_A))
print(np.sum(costs_B))
print(np.sum(costs_C))

plt.plot(battery[:100])
plt.show()
print("without battery A: " + str(mdp.get_total_costs(test["Production_A"] - test["Consumption_A"])))
print("without battery B: " + str(mdp.get_total_costs(test["Production_B"] - test["Consumption_B"])))
print("without battery C: " + str(- np.sum(test["Consumption_C"])))


# mdp = MDP(1000, 1000, 500, 500, 12000, 7,7)

# # test B data on single agent
# # mdp_B = SMDP(1000, 1000, 200, 500, 60, 7,7)
# # Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(df_bs_B,500, mdp_B)
# # reward_B, policy_B, battery_B, dis_B, loss_B, states_B = SMDP.find_policy(mdp_B, Q, df_bs_B)

# #Q_A,Q_B,Q_C, rewards_per_episode, changed, all_rewards, battery = MA_QLearning.iterate(df_training,10000, mdp)

# print("actions A changed with check actions: " + str(changed))
# # print(all_rewards)
# plt.plot(rewards_per_episode)
# plt.show()

# costs_A, costs_B, costs_C, policy_A, policy_B, policy_C, battery_A, battery_B = Policy.find_policies(mdp, Q_A, Q_B, Q_C, test)

# plt.hist(policy_A)
# plt.show()
# plt.hist(policy_B)
# plt.show()
# plt.hist(policy_C)
# plt.show()
# # baseline_rewards_A, baseline_states, baseline_policy_A, baseline_bat, difference= Baseline.find_baseline_policy(df_bs_A, mdp)
# # baseline_rewards_B, baseline_states, baseline_policy_B, baseline_bat, difference= Baseline.find_baseline_policy(df_bs_B, mdp)
# # baseline_rewards_B, baseline_states, baseline_policy_B, baseline_bat, difference= Baseline.find_baseline_policy(df_bs_B, mdp)



# # print("Baseline A:        " + str(np.sum(baseline_rewards_A)))
# # print("Baseline B:        " + str(np.sum(baseline_rewards_B)))
# # print("Baseline C:")

# print("Agent A: " + str(np.sum(costs_A)))
# print("Agent B: " + str(np.sum(costs_B)))
# print("Agent C: " + str(np.sum(costs_C)))


# print("without battery A: " + str(mdp.get_total_costs(test["Production_A"] - test["Consumption_A"])))
# print("without battery B: " + str(mdp.get_total_costs(test["Production_B"] - test["Consumption_B"])))
# print("without battery C: " + str(- np.sum(test["Consumption_C"])))


# plt.plot(test["Production_A"] - test["Consumption_A"][:240], color = "grey", linestyle = "dashdot")
# plt.plot(test["Production_B"] - test["Consumption_B"][:240], color = "lightgrey", linestyle = "dashdot")
# plt.plot(battery_A[:240], color = "green")
# plt.show()

# plt.plot(test["Production_A"] - test["Consumption_A"][1000:1240], color = "grey", linestyle = "dashdot")
# plt.plot(test["Production_B"] - test["Consumption_B"][1000:1240], color = "lightgrey", linestyle = "dashdot")
# plt.plot(battery_A[1000:1240], color = "green")
# plt.show()

