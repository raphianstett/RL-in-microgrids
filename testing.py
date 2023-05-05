from value_iteration import ValueIteration
from data import RealData

dat = RealData.get_real_data()


Q_table_sol, rewards_per_episode, all_rewards, actions, states_id, states, battery = ValueIteration.value_iteration(dat,100)

