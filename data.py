import numpy as np
import pandas as pd

class StepFunctions:
    def generate_step_consumption():
        
        high_early = [400]*6
        low = [250]*6 
        high_late = [400]*3
        cons_step = [100]*7 + high_early + low + high_late + [100]*2
        #cons_step = [x + 300 for x in cons_step]
        return cons_step


    # generate step production
    # use 1500 Wh as production from 9 am to 5 pm

    def generate_step_production():
        none_early = [0]*8
        high = [1500]*8
        none_late = [0]*8
        return none_early + high + none_late

    # print(generate_step_production())
    # print(len(generate_step_production()))
    # print(len(generate_step_consumption()))
    # plt.plot(generate_step_consumption())
    # plt.show()
    cons = np.array(generate_step_consumption())
    prod = np.array(generate_step_production())
    diff = prod - cons
    diff = [0 if x > 0 else x for x in diff]
    # print(diff)
    # print(np.sum(diff))
    # plt.plot(diff)
    # plt.show()

    # import test data for training and testing
    def get_test_data():
        d = {'Consumption': StepFunctions.generate_step_consumption(), 'Production': StepFunctions.generate_step_production(), 'Time': [*range(0,24,1)]}
        dat = pd.DataFrame(d, columns = ['Consumption', 'Production', 'Time'])

        # data checking

        prod = dat["Production"]
        cons = dat["Consumption"]
        return dat
    
class RealData:
    def get_real_data():
        dat = pd.read_csv("household_with_pv_new.csv")
        
        dat = {'Consumption': dat["Consumption"], 'Production': dat["Production"], 'Time': dat["Time"], 'Date': dat["Date"], 'Purchased': dat["Purchased"]}
        dat = pd.DataFrame(dat, columns=['Consumption', 'Production', 'Time', 'Date', 'Purchased'])

        return dat
    

d = RealData.get_real_data()


# print(d["Purchased"].sum()) 
# = 1697651.0