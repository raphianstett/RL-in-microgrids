from data import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data_2:
    def get_data():
        dat = pd.read_csv("Data/MA2_data.csv")
        dat = {'Consumption_A': dat["Consumption_A"],'Consumption_B': dat["Consumption_B"], 'Production_A': dat["Production_A"],'Production_B': dat["Production_B"], 'is weekend': dat["is_weekend"], 'Date': dat["Date"],'Time': dat["Time"]}
        dat['Date'] = pd.to_datetime(dat['Date'])
        dat = pd.DataFrame(dat, columns=['Consumption_A','Consumption_B', 'Production_A','Production_B', 'Date','Time','is weekend'])        
        return dat
    
    # function to generate data for 2MARL
    def write_data(self):
        gen = DataGeneration()
        df = Data.get_data_pd()
        df["Consumption_B"] = gen.add_noise(gen.shuffle_consumption(df), 0.1)
        df["Production_B"] = gen.scale(df["Production"], 0.8)
        df["Date"] = gen.get_days().repeat(24)
        df = df.rename(columns = {"Consumption" : "Consumption_A", "Production" : "Production_A"})
        df = df[["Consumption_A", "Consumption_B", "Production_A", "Production_B", "is_weekend", "Date", "Time"]]
        df.to_csv("Data/MA2_data.csv", index = False)
        

    # for test and training split with numpy
    def get_training_test(days, get_summer, get_winter):
        df = Data_2.get_data()
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
    
    # for test and training split with pd dataframe as input
    def split_data(df,days):
        
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
    
    def get_data_B():
        dat = Data_2.get_data()
        gen = DataGeneration()
        dat_B = pd.DataFrame(dat, columns=['Consumption_B', 'Production_B', 'Time', 'is weekend']) 
        dat_B['Date'] = gen.get_days().repeat(24)
        dat_B = dat_B.rename(columns = {"Consumption_B" : "Consumption", "Production_B" : "Production"})
        return dat_B
    
# class to generate data for households B and C 
class DataGeneration:
    def get_days(self):
        start_date = '2021-06-01'
        end_date = '2022-05-31'
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        return date_range

    def get_weekend(self,dates):
        is_weekend = [None]*365
        for i in range(len(dates)):
            if dates[i].weekday() < 5:  # 0-4 represents Monday to Friday (weekday)
                is_weekend[i] = 0
            else:  # 5-6 represents Saturday and Sunday (weekend)
                is_weekend[i] = 1
        return is_weekend

    # shuffles weekends and weekdays separately and inside each month
    def shuffle_days(self):
        dates = self.get_days()
        
        df = pd.DataFrame({"Date" : dates, "is weekend": self.get_weekend(dates)})
        is_weekend = self.get_weekend(dates)
        weekdays_df = df[df['Date'].apply(lambda x: x.weekday() < 5)]
        weekends_df = df[df['Date'].apply(lambda x: x.weekday() >= 5)]
        
        # Shuffle weekdays and weekends within each month
        shuffled_weekdays_df = weekdays_df.groupby(pd.Grouper(key='Date', freq='M')).apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
        shuffled_weekends_df = weekends_df.groupby(pd.Grouper(key='Date', freq='M')).apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
        shuffled_dates = pd.DataFrame(columns = ["Date", "is weekend"])
        
        idx_w = 0
        idx_d = 0
        for i in range(len(is_weekend)):
            if is_weekend[i]:
                new_row = shuffled_weekends_df.iloc[idx_w].to_frame().T
                idx_w +=1
            else:
                new_row = shuffled_weekdays_df.iloc[idx_d].to_frame().T
                idx_d +=1
            shuffled_dates = pd.concat([shuffled_dates, new_row])
        shuffled_dates.reset_index(drop = True)
        return shuffled_dates

    # shuffles consumption according to dates
    def shuffle_consumption(self,df):
        shuffled_dates = self.shuffle_days()
        df_copy = df.copy()
        df_copy.set_index("Date", inplace = True)

        shuffled_cons = pd.DataFrame(columns = ["Consumption", "Production", "Time", "Purchased", "is_weekend"])
        i = 0
        for i in range(365):
            d = shuffled_dates.iloc[i,0]
            c = df_copy.loc[d]
            shuffled_cons = pd.concat([shuffled_cons, c])
        df.reset_index(drop = True)
        return list(shuffled_cons["Consumption"])

    # adds gaussian noise to vector with set standard deviation
    def add_noise(self,vec,std):
        mean = 1
        vec = list(vec)
        for x in vec:
            noise = np.random.normal(mean, std, len(vec))
            x *= noise  
        return vec

    def scale(self,prod, scaling):
        prod = [np.floor(x * scaling) for x in prod]
        return prod

    def plot_data(self,df):
        hours = np.arange(500, 741) 
        days = (hours - 500) / 24  
        x_ticks = days
        x_ticklabels = hours

        plt.rcParams["font.size"] = 12
        plt.rcParams["font.family"] = "Arial"

        plt.figure(figsize=(10, 5))

        plt.plot(df["Consumption_B"][:168], color="red", linestyle="-", label="Consumption_A")
        plt.plot(df["Production_B"][:168], color="green", linestyle="dashdot", label="Consumption_B")

        plt.xlabel("Days")
        plt.xticks([], [])
        plt.ylabel("Value in Wh")
        plt.title("Consumption and Production")
        plt.legend()


class Data_3:
    def write_data():
        gen = DataGeneration()
        df = Data.get_data_pd()
        
        df["Consumption_B"] = gen.add_noise(gen.shuffle_consumption(df), 0.1)
        df["Production_B"] = gen.scale(df["Production"], 0.8)
        df["Date"] = gen.get_days().repeat(24)
        df["Consumption_C"] = gen.scale(gen.add_noise(gen.shuffle_consumption(df), 0.2), 0.5)
        df["Production_C"] = [0]*(24*365)
        df["Date"] = gen.get_days().repeat(24)
        df = df.rename(columns = {"Consumption" : "Consumption_A", "Production" : "Production_A"})
        
        df = df[["Consumption_A", "Consumption_B", "Consumption_C", "Production_A", "Production_B", "Production_C", "is_weekend", "Date", "Time"]]
        
        df.to_csv("Data/MA3_data.csv", index = False)
        
    def get_data():
        dat = pd.read_csv("Data/MA_3_data.csv")
        dat = {'Consumption_A': dat["Consumption_A"],'Consumption_B': dat["Consumption_B"], 'Consumption_C': dat["Consumption_C"], 'Production_A': dat["Production_A"],'Production_B': dat["Production_B"], 'Production_C': dat["Production_C"], 'is weekend': dat["is_weekend"], 'Date': dat["Date"],'Time': dat["Time"]}
        dat['Date'] = pd.to_datetime(dat['Date'])
        dat = pd.DataFrame(dat, columns=['Consumption_A','Consumption_B', 'Consumption_C', 'Production_A','Production_B','Production_C', 'is weekend', 'Date','Time'])        
        return dat
    
    
    def plot_differences():
        dat = Data_3.get_data()
        d_s = Data.get_summer_pd(dat)
        d_w = Data.get_winter_pd(dat)

        d_A = d_s["Production_A"] - d_s["Consumption_A"]
        d_B = d_s["Production_B"] - d_s["Consumption_B"]
        d_C = d_s["Consumption_C"] 

        plt.plot(d_A[168:336], color = "tab:red", label = "A", linestyle = "solid")
        plt.plot(d_B[168:336], color = "royalblue", label = "B", linestyle = "dashdot")
        plt.plot(d_C[168:336], color ="darkslategrey", label = "C", linestyle = "dashdot")
        plt.xlabel('Days')
        plt.ylabel('Value in Wh')
        plt.xticks(np.arange(180,348,24), np.arange(1,8,1))
        plt.legend()
        # plt.title('Differences between production and demand')
        plt.savefig("MA3_data.png", dpi = 300)
        plt.show()
