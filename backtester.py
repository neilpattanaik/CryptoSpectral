import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_zscore(time_series):
    return (time_series-time_series.mean())/np.std(time_series)

def get_movingaverage(time_series, days):
    return pd.Series.to_numpy(pd.Series(time_series).rolling(window = days, center = False).mean())

def get_movingstd(time_series, days):
    return pd.Series.to_numpy(pd.Series(time_series).rolling(window = days, center = False).std())

class Trader:
    def trade_pair(self, coin1, coin2, zenter,zexit,profitpercent):
        series1, series2 = self.testing_timeseries_df[coin1], self.testing_timeseries_df[coin2]
        price_ratio_series = series1/series2
        sample_ma = price_ratio_series.rolling(window=self.samplewindow,
                               center=False).mean()
        sample_std = price_ratio_series.rolling(window=self.samplewindow,
                        center=False).std()
        test_ma = price_ratio_series.rolling(window=self.testwindow,
                               center=False).mean()
        zscore = (test_ma - sample_ma)/sample_std
        qty_coin1, qty_coin2= 0, 0
        positions = {1:[], 2:[]}
        if not min(series1) or not min(series2):
            return

        for i in range(len(price_ratio_series)):
        # Short position
            if zscore.iloc[i] > zenter:
                qty_coin1 -= 1 / (2* series1.iloc[i])
                qty_coin2 += 1 / (2* series2.iloc[i])
                positions[1].append([-1 / (2* series1.iloc[i]), series1.iloc[i], i])
                positions[2].append([1 / (2* series2.iloc[i]), series2.iloc[i], i])

        # Long position
            elif zscore.iloc[i] < -zenter:
                qty_coin1 += 1 / (2* series1.iloc[i])
                qty_coin2 -= 1 / (2* series2.iloc[i])
                positions[1].append([1 / (2* series1.iloc[i]), series1.iloc[i]])
                positions[2].append([-1 / (2* series2.iloc[i]), series2.iloc[i]])

        # Trade exit criterion
            elif abs(zscore.iloc[i]) < zexit:
                self.realized_pnl += qty_coin1*series1.iloc[i] + qty_coin2*series2.iloc[i] 
                #print('Position cleared. Clearing Profit:', str(qty_coin1*series1.iloc[i] + qty_coin2*series2.iloc[i]))
                qty_coin1, qty_coin2 = 0,0
                positions[1],positions[2]=[],[]
            del_list=[]
            for j in range(len(positions[1])):
                if positions[1][j][0] > 0:
                    if (series1.iloc[i]*positions[1][j][0] - positions[2][j][0]*(positions[2][j][1] - series2.iloc[i]))/(positions[1][j][1]*positions[1][j][0]) -1 >= profitpercent:
                        del_list.append(j)
                        qty_coin1 -= positions[1][j][0]
                        qty_coin2 -= positions[2][j][0]
                        self.realized_pnl += positions[1][j][0]*series1.iloc[i] + positions[2][j][0]*series2.iloc[i] 
                elif (series2.iloc[i]*positions[2][j][0] - positions[1][j][0]*(positions[1][j][1] - series1.iloc[j]))/(positions[2][j][1]*positions[2][j][0]) -1 >= profitpercent:
                    del_list.append(j)
                    qty_coin1 -= positions[1][j][0]
                    qty_coin2 -= positions[2][j][0]
                    self.realized_pnl += positions[1][j][0]*series1.iloc[i] + positions[2][j][0]*series2.iloc[i] 
            for j in reversed(range(len(del_list))):
                del_index = del_list.pop(j)
                positions[1].pop(del_index)
                positions[2].pop(del_index)
        for i in reversed(range(len(positions[1]))):
            self.unrealized_pnl += positions[1][i][0]*(series1.iloc[-1] - positions[1][i][1])
            self.unrealized_pnl += positions[2][i][0]*(series2.iloc[-1] - positions[2][i][1])
            positions[1].pop(i)
            positions[2].pop(i)

    def __init__(self, samplewindow, testwindow, pairs, dataset, start_time, end_time, zenter, zexit, profitpercent):
        self.samplewindow, self.testwindow, self.pairs, self.open_positions, self.tradecount = samplewindow, testwindow, pairs, {},0
        self.realized_pnl, self.unrealized_pnl = 0,0

        self.testing_timeseries_df = pd.DataFrame(index=(np.asarray(dataset._timestamps[start_time:-end_time]).astype(float)).astype('datetime64[s]'))
        names_in_pair = []
        for pair in pairs:
            for coin in pair:
                names_in_pair.insert(0, list(coin)[0])
                self.testing_timeseries_df[names_in_pair[0]] = dataset.price_matrix[list(coin.values())[0],start_time:-end_time]
            self.trade_pair(names_in_pair.pop(), names_in_pair.pop(),zenter, zexit,profitpercent)
        self.percent = (self.realized_pnl+self.unrealized_pnl)/100
        
