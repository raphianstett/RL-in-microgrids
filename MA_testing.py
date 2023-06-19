from MA2_learning import MA_QLearning
from learning import QLearning

from environment import State
from environment import MDP as SMDP
from MA2_environment import Reward
from MA2_environment import MDP
from data import RealData
from MA_data import Data
import matplotlib.pyplot as plt
import numpy as np
from learning import Baseline
data = Data()
df = data.get_data()
df_training, test = RealData.split_data(df, 7)

x, df_bs_B = RealData.split_data(data.get_data_B(), 7)
x, df_bs_A = RealData.split_data(RealData.get_real_data(), 7)
mdp_A = MDP(1000, 500, 200, 500, 60, 5,5)
mdp_B = MDP(1000, 500, 200, 500, 60, 5,5)

# test B data on single agent
# mdp_B = SMDP(1000, 1000, 200, 500, 60, 7,7)
# Q, rewards_per_episode, all_rewards, actions, states_id, states, battery = QLearning.iterate(df_bs_B,500, mdp_B)
# reward_B, policy_B, battery_B, dis_B, loss_B, states_B = SMDP.find_policy(mdp_B, Q, df_bs_B)

Q_A,Q_B, rewards_per_episode = MA_QLearning.iterate(df_training,800, mdp_A, mdp_B)

plt.plot(rewards_per_episode)
plt.show()

reward_A, policy_A, battery_A, dis_A, loss_A, states_A = MDP.find_policy(mdp_A, Q_A, test, "A")
reward_B, policy_B, battery_B, dis_B, loss_B, states_B = MDP.find_policy(mdp_B, Q_B, test, "B")

plt.hist(policy_A)
plt.show()
plt.hist(policy_B)
plt.show()

baseline_rewards_A, baseline_states, baseline_policy_A, baseline_bat, difference= Baseline.find_baseline_policy(df_bs_A, mdp_A)
baseline_rewards_B, baseline_states, baseline_policy_B, baseline_bat, difference= Baseline.find_baseline_policy(df_bs_B, mdp_B)


print("Baseline A:        " + str(np.sum(baseline_rewards_A)))
print("Baseline B:        " + str(np.sum(baseline_rewards_B)))

print("Agent A connected: " + str(np.sum(reward_A)))
print("Agent B connected: " + str(np.sum(reward_B)))

print("without battery A: " + str(mdp_A.get_total_costs(df_bs_A["Production"] - df_bs_A["Consumption"])))
print("without battery B: " + str(mdp_B.get_total_costs(df_bs_B["Production"] - df_bs_B["Consumption"])))

plt.plot(battery_A[:240], color = "red")
plt.plot(battery_B[:240], color = "green")
plt.show()
