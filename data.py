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
        dat = pd.DataFrame(dat, columns=['Consumption', 'Production', 'Date', 'Time'])
        dat['Date'] = pd.to_datetime(dat['Date'])
        dat = RealData.mark_weekend_days(dat)
        #dat = pd.DataFrame(dat, columns=['Consumption', 'Production', 'Time', 'is_weekend'])
        return dat
    def get_data():
        dat = RealData.get_real_data()
        dat = pd.DataFrame(dat, columns= ['Consumption', 'Production', 'Time'])
        return dat.to_numpy()
    
    def cut_production(df):
        prod = df["Production"]
        df["Production"] = [min(x, 4000) for x in prod]
        return df

    def mark_weekend_days(df):
        df['is_weekend'] = df['Date'].apply(lambda x: int(x.weekday() >= 5))
        return df

    def extract_days(group, days):
        return group.head(days)

    def get_training_test(days, get_summer, get_winter):
        df = RealData.get_real_data()
        if get_summer:
            df = RealData.get_summer_pd(df)
        if get_winter:
            df = RealData.get_winter_pd(df)
        df['Date'] = pd.to_datetime(df['Date'])
        grouped = df.groupby(df['Date'].dt.to_period('M'))
        n = days * 24
        new_df = pd.concat([RealData.extract_days(group, n) for _, group in grouped])
    
        # Remove the extracted rows from the original DataFrame
        df = df.drop(new_df.index)
        
        # reset index now
        new_df = new_df.reset_index(drop=True)
        # Reset the index of the updated DataFrame
        df = df.reset_index(drop=True)
        df = df.drop('Date', axis = 1)
        new_df = new_df.drop('Date', axis = 1)
        np.round(df).astype(int)
        np.round(new_df).astype(int)
        
        return df.to_numpy(), new_df.to_numpy()

   

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
    # function for pd dataframe
    def get_summer_pd(df):
        df.drop(df.index[2928:7298], inplace=True)
        df.reset_index(drop = True, inplace = True)
        return df
    
    # function for numpy array
    def get_summer(arr):
        arr = np.delete(arr, np.s_[2928:7298], axis=0)
        return arr

    # function for pd dataframe
    def get_winter_pd(df):
        df.drop(df.index[0:2928], inplace = True)
        df.drop(df.index[4368: len(df)], inplace = True)
        df.reset_index(drop = True, inplace = True)
        return df
    
    #function for numpy array

    def get_winter(arr):
        arr = np.delete(arr, np.s_[0:2928], axis=0)
        arr = np.delete(arr, np.s_[4368:], axis=0)
        return arr


# month * days * hours


# print(training_data["Production"] - training_data["Consumption"])

#print(RealData.train])

