import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data:

    # returns pandas dataframe
    def get_data_pd():
        dat = pd.read_csv("Data/household_with_pv.csv")
        dat = {'Consumption': dat["Consumption"], 'Production': dat["Production"], 'Time': dat["Time"], 'Date': dat["Date"], 'Purchased': dat["Purchased"]}
        dat = pd.DataFrame(dat, columns=['Consumption', 'Production', 'Date', 'Time'])
        dat['Date'] = pd.to_datetime(dat['Date'])
        dat = Data.mark_weekend_days(dat)
        return dat
    
    # returns numpy array for RL training
    def get_data():
        dat = Data.get_data_pd()
        dat = pd.DataFrame(dat, columns= ['Consumption', 'Production', 'Time'])
        return dat.to_numpy()
    

    def mark_weekend_days(df):
        df['is_weekend'] = df['Date'].apply(lambda x: int(x.weekday() >= 5))
        return df

    def extract_days(group, days):
        return group.head(days)

    # returns two dataframes with training and test split
    # days sets how many days are extracted from each month for testing
    # it can be set if only summer or winter data should be used
    def get_training_test(days, get_summer, get_winter):
        df = Data.get_data_pd()

        if get_summer:
            df = Data.get_summer_pd(df)
        if get_winter:
            df = Data.get_winter_pd(df)
        df['Date'] = pd.to_datetime(df['Date'])
        grouped = df.groupby(df['Date'].dt.to_period('M'))
        n = days * 24
        new_df = pd.concat([Data.extract_days(group, n) for _, group in grouped])
    
        df = df.drop(new_df.index)
        
        
        new_df = new_df.reset_index(drop=True)
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

    #create histogram with equal-frequency bins, is used to determine bin edges for discretisation
    def get_bin_boundaries(data, nbins):
        n, bins, patches = plt.hist(data, Data.equalObs(data, nbins), edgecolor='black')
        return bins

    def get_prod_nonzeros(prod):
        p = []
        for x in prod:
            if x != 0:
                p.append(x)
            else: 
                continue
        return p
    
    # function to get summer from pd dataframe
    def get_summer_pd(df):
        df.drop(df.index[2928:7298], inplace=True)
        df.reset_index(drop = True, inplace = True)
        return df
    
    # function to get summer from  numpy array
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

    # plots data for one week starting from start for winter and summer
    def plot_data(start):
        data = Data.get_data_pd()
        
        data = Data.get_summer_pd(data)
        fig = plt.figure(figsize = (10,5))
        ax = fig.add_subplot(1,2,1)
        ax.plot(data["Production"][start:start+168], color = "royalblue", linestyle = "dashed", label = "Production")
        ax.plot(data["Consumption"][start:start+168], color = "yellowgreen", linestyle = 'solid', label = "Demand")
        ax.set_title("Summer")
        ax.set_xticks(np.arange(12,200,24), np.arange(0,8,1))
        ax.set_xlabel('Days')
        ax.set_ylabel('Value in Wh')
        ax.legend()
        data = Data.get_winter_pd(data)
        
        ax = fig.add_subplot(1,2,2)
        ax.plot(data["Production"][start:start+168], color = "royalblue", linestyle = "dashed", label = "Production")
        ax.plot(data["Consumption"][start:start+168], color = "yellowgreen", linestyle = 'solid', label = "Demand")
        ax.set_title("Winter")
        ax.set_xticks(np.arange(12,180,24), np.arange(1,8,1))
        ax.set_xlabel('Days')
        ax.set_ylabel('Value in Wh')
        ax.legend()
        plt.tight_layout()
        plt.savefig('plots/SARL/Data.png', dpi = 300)
