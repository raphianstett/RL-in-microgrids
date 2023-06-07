from data import RealData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Data:
    def get_data():
        df = RealData.get_real_data()
        df["Consumption_B"] = Data.add_noise(Data.shuffle_consumption(df), 0.1)
        # print(df[:10])
        df["Production_B"] = Data.scale_production(df["Production"], 0.8)
        df = df.rename(columns = {"Consumption" : "Consumption_A", "Production" : "Production_A"})
        # # print(df)
        df = df[["Consumption_A", "Consumption_B", "Production_A", "Production_B", "is_weekend", "Time"]]


    def get_days():
        start_date = '2021-06-01'
        end_date = '2022-05-31'
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        return date_range

    def get_weekend(dates):
        is_weekend = [None]*365
        for i in range(len(dates)):
            if dates[i].weekday() < 5:  # 0-4 represents Monday to Friday (weekday)
                is_weekend[i] = 0
            else:  # 5-6 represents Saturday and Sunday (weekend)
                is_weekend[i] = 1
        return is_weekend

    def shuffle_days():
        dates = Data.get_days()
        
        df = pd.DataFrame({"Date" : dates, "is weekend": Data.get_weekend(dates)})
        is_weekend = Data.get_weekend(dates)
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

    def shuffle_consumption(df):
        shuffled_dates = Data.shuffle_days()
        print(df[:12])
        df_copy = df.copy()

        # shuffled_dates.set_index("Date", inplace = True)
        # print(shuffled_dates)
        df_copy.set_index("Date", inplace = True)
        print(df_copy[:14])
        print(df[:13])
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

    def add_noise(vec,std):
        mean = 1
        vec = list(vec)
        for x in vec:
            noise = np.random.normal(mean, std, len(vec))
            x *= noise  
        return vec

    def scale_production(prod, scaling):
        prod = [x * scaling for x in prod]
        return prod


    
    def print_data(df):
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


