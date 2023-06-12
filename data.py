import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

np.set_printoptions(threshold=np.inf)


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
        dat['Date'] = pd.to_datetime(dat['Date'])
        dat = RealData.mark_weekend_days(dat)
        return dat
    
    def cut_production(df):
        prod = df["Production"]
        df["Production"] = [min(x, 4000) for x in prod]
        return df

    def mark_weekend_days(df):
        df['is_weekend'] = df['Date'].apply(lambda x: int(x.weekday() >= 5))
        return df

    def split_data_test(data, days): 
        test_data = pd.DataFrame(columns = ['Consumption','Production', 'Time','Date','Purchased'], index = range(0,12*days*24))
        days_in_month = np.array([30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31])
        for i in range(12):
            for j in range(days*24):
             
                which = 0 if i == 0 else days_in_month[[*range(0,i,1)]].sum()
                #print(which)
                # print(days_in_month[[*range(0,i,1)]].sum())
                #print([0,*range(0,i,1)])
                idx = (which * 24 + j)
                row = data.loc[idx]
                #print(row)
                test_data.loc[i * days * 24 + j] = row
                data = data.drop(idx)
            data = data.set_axis(range(0, len(data)), axis = 'index')
            test_data = test_data.set_axis(range(0, len(test_data)), axis = 'index')    
        return data, test_data

    def extract_days(group, days):
        return group.head(days)

    def split_data(df, days):
        df['Date'] = pd.to_datetime(df['Date'])
        grouped = df.groupby(df['Date'].dt.to_period('M'))
        n = days * 24
        new_df = pd.concat([RealData.extract_days(group, n) for _, group in grouped])
        new_df = new_df.reset_index(drop=True)

        # Remove the extracted rows from the original DataFrame
        df = df.drop(new_df.index)

        # Reset the index of the updated DataFrame
        df = df.reset_index(drop=True)
        return df, new_df

   

    def equalObs(x, nbin):
        nlen = len(x)
        return np.interp(np.linspace(0, nlen, nbin + 1),
                        np.arange(nlen),
                        np.sort(x))

    #create histogram with equal-frequency bins 
    def get_bin_boundaries(data, nbins):
        n, bins, patches = plt.hist(data, RealData.equalObs(data, nbins), edgecolor='black')
        return bins

    def get_prod_nonzeros(prod):
        p = []
        for x in prod:
            if x != 0:
                p.append(x)
            else: 
                continue
        return p
    def get_summer(df):
        return df.drop(range(2930, 7298))
    def get_winter(df):
        df = df.drop(range(0,2929))
        df = df.drop(range(7299), len(df))
        return df

# month * days * hours
training_data, test = RealData.split_data(RealData.get_real_data(), 7)
# print(training_data)
# print(training_data["Production"] - training_data["Consumption"])


