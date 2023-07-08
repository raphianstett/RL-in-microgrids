from data import RealData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data_2:
    def get_data():
        dat = pd.read_csv("MA_data.csv")
        # print(dat[:100])
        dat = {'Consumption_A': dat["Consumption_A"],'Consumption_B': dat["Consumption_B"], 'Production_A': dat["Production_A"],'Production_B': dat["Production_B"], 'is weekend': dat["is_weekend"], 'Date': dat["Date"],'Time': dat["Time"]}
        dat['Date'] = pd.to_datetime(dat['Date'])
        dat = pd.DataFrame(dat, columns=['Consumption_A','Consumption_B', 'Production_A','Production_B', 'is weekend', 'Date','Time'])        
        return dat
    
    def write_data(self):
        df = RealData.get_real_data()
        df["Consumption_B"] = self.add_noise(self.shuffle_consumption(df), 0.1)
        
        df["Production_B"] = self.scale(df["Production"], 0.8)
        df["Date"] = self.get_days().repeat(24)
        df = df.rename(columns = {"Consumption" : "Consumption_A", "Production" : "Production_A"})
        
        df = df[["Consumption_A", "Consumption_B", "Production_A", "Production_B", "is_weekend", "Date", "Time"]]
        
        df.to_csv("MA_data.csv", index = False)
        # print(df)

    def get_training_test(days, get_summer, get_winter):
        df = Data_2.get_data()
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
    
    def split_data(df,days):
        
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
    
    def get_days():
        start_date = '2021-06-01'
        end_date = '2022-05-31'
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        #dates = date_range.repeat(24)
        return date_range

    def get_weekend(self,dates):
        is_weekend = [None]*365
        for i in range(len(dates)):
            if dates[i].weekday() < 5:  # 0-4 represents Monday to Friday (weekday)
                is_weekend[i] = 0
            else:  # 5-6 represents Saturday and Sunday (weekend)
                is_weekend[i] = 1
        return is_weekend

    def shuffle_days(self):
        dates = self.get_days()
        
        df = pd.DataFrame({"Date" : dates, "is weekend": self.get_weekend(dates)})
        is_weekend = self.get_weekend(dates)
        weekdays_df = df[df['Date'].apply(lambda x: x.weekday() < 5)]
        weekends_df = df[df['Date'].apply(lambda x: x.weekday() >= 5)]
        
        # Shuffle weekdays and weekends within each month
        shuffled_weekdays_df = weekdays_df.groupby(pd.Grouper(key='Date', freq='M')).apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
        shuffled_weekends_df = weekends_df.groupby(pd.Grouper(key='Date', freq='M')).apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
        
        # concat, so that the structure of weekends is the same
        shuffled_dates = pd.DataFrame(columns = ["Date", "is weekend"])
        
        idx_w = 0
        idx_d = 0
        for i in range(len(is_weekend)):
            if is_weekend[i]:
                #print("Weekend: " + str(is_weekend[i]))
                new_row = shuffled_weekends_df.iloc[idx_w].to_frame().T
                idx_w +=1
            else:
                #print("not weekend" + str(is_weekend[i]))
                new_row = shuffled_weekdays_df.iloc[idx_d].to_frame().T
                idx_d +=1
            shuffled_dates = pd.concat([shuffled_dates, new_row])
        
        #print(shuffled_df[:24])
        # Reset the index of the shuffled DataFrame
        # shuffled_dates = shuffled_dates.reset_index(drop=True)
        shuffled_dates.reset_index(drop = True)
        return shuffled_dates

    def shuffle_consumption(self,df):
        shuffled_dates = self.shuffle_days()
       
        df_copy = df.copy()

        # shuffled_dates.set_index("Date", inplace = True)
        # print(shuffled_dates)
        df_copy.set_index("Date", inplace = True)
        
        
        # print(df)

        shuffled_cons = pd.DataFrame(columns = ["Consumption", "Production", "Time", "Purchased", "is_weekend"])
        i = 0
        for i in range(365):
            d = shuffled_dates.iloc[i,0]
            # print(d)
            c = df_copy.loc[d]
            # print(c)
            shuffled_cons = pd.concat([shuffled_cons, c])
        df.reset_index(drop = True)
        return list(shuffled_cons["Consumption"])

    # print(shuffle_consumption(df))

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

    def get_data_B():
        dat = Data_2.get_data()
        dat_B = pd.DataFrame(dat, columns=['Consumption_B', 'Production_B', 'is weekend', 'Time']) 
        dat_B['Date'] = Data_2.get_days().repeat(24)
        dat_B = dat_B.rename(columns = {"Consumption_B" : "Consumption", "Production_B" : "Production"})
        return dat_B
        

    
    def plot_data(self,df):
        hours = np.arange(500, 741)  # Assuming each value represents a day
        days = (hours - 500) / 24  # Convert days to hours
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

        plt.show()

class Data_3:
    def write_data():
        data2 = Data_2()
        df = RealData.get_real_data()
        
        df["Consumption_B"] = data2.add_noise(data2.shuffle_consumption(df), 0.1)
        df["Production_B"] = data2.scale(df["Production"], 0.8)
        df["Date"] = data2.get_days().repeat(24)
        df["Consumption_C"] = data2.scale(data2.add_noise(data2.shuffle_consumption(df), 0.2), 0.5)
        df["Production_C"] = [0]*(24*365)
        df["Date"] = data2.get_days().repeat(24)
        df = df.rename(columns = {"Consumption" : "Consumption_A", "Production" : "Production_A"})
        
        df = df[["Consumption_A", "Consumption_B", "Consumption_C", "Production_A", "Production_B", "Production_C", "is_weekend", "Date", "Time"]]
        
        df.to_csv("MA_3_data.csv", index = False)
        # print(df[:100])
        # return df
    
    def get_data():
        dat = pd.read_csv("MA_3_data.csv")
        # print(dat[:100])
        dat = {'Consumption_A': dat["Consumption_A"],'Consumption_B': dat["Consumption_B"], 'Consumption_C': dat["Consumption_C"], 'Production_A': dat["Production_A"],'Production_B': dat["Production_B"], 'Production_C': dat["Production_C"], 'is weekend': dat["is_weekend"], 'Date': dat["Date"],'Time': dat["Time"]}
        dat['Date'] = pd.to_datetime(dat['Date'])
        dat = pd.DataFrame(dat, columns=['Consumption_A','Consumption_B', 'Consumption_C', 'Production_A','Production_B','Production_C', 'is weekend', 'Date','Time'])        
        return dat
    
    
    def plot_differences():
        dat = Data_3.get_data()
        d_s = RealData.get_summer_pd(dat)
        d_w = RealData.get_winter_pd(dat)

        d_A = d_s["Production_A"] - d_s["Consumption_A"]
        d_B = d_s["Production_B"] - d_s["Consumption_B"]
        d_C = d_s["Consumption_C"] 

        plt.plot(d_A[:168], color = "tab:red", label = "A", linestyle = "solid")
        plt.plot(d_B[:168], color = "royalblue", label = "B", linestyle = "dashdot")
        plt.plot(d_C[:168], color ="darkslategrey", label = "C", linestyle = "dashdot")
        plt.xlabel('Days')
        plt.ylabel('Value in Wh')
        plt.xticks(np.arange(12,180,24), np.arange(1,8,1))
        plt.legend()
        plt.title('Differences between production and demand')
        plt.savefig("MA3_data.png", dpi = 300)
        plt.show()

    