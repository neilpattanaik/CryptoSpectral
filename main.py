import cryptodata
import backtester
import algorithms
import spectralpairing
import numpy as np
import matplotlib.pyplot as plt
import random


def test_trading_strategy(tradeplots = False, sample_window = 24, test_window=1, data_start=1950, test_start_period=720, test_end_period=1, number_of_coins=100, dimensions=15, clusters=10, zenter=1, zexit=0, profitpercent=.05):
    pairs, return_matrix, price_matrix, dataset = spectralpairing.spectral_pairing(number_of_coins,data_start, test_start_period,"hour", dimensions, clusters)

    if tradeplots:
        for pair in pairs:
            price_ratios = price_matrix[list(pair[0].values())[0],:]/price_matrix[list(pair[1].values())[0],:]
            wide_ratios_mavg, wide_ratios_stds = backtester.get_movingaverage(price_ratios, sample_window), backtester.get_movingstd(price_ratios, sample_window)
            short_ratios_mavg= backtester.get_movingaverage(price_ratios, test_window)
            window_zsore = (short_ratios_mavg-wide_ratios_mavg)/wide_ratios_stds

            
            fig, (axs1, axs2, axs3) = plt.subplots(3)
            fig.suptitle(list(pair[0].keys())[0]+'/'+list(pair[1].keys())[0]+' Price Ratio Versus Time')
            time = (np.asarray(dataset._timestamps).astype(float)).astype('datetime64[s]')
            plt.plot(figsize=(15,7)) 

            # Plot 1: Time Series of Price Ratio with overall avg, moving avg
            axs1.plot(time, price_ratios)
            axs1.plot(time, short_ratios_mavg)
            axs1.plot(time, wide_ratios_mavg)
            axs1.axhline((price_ratios).mean(), color='red', linestyle='--')
            axs1.axvline(x = np.asarray([dataset._timestamps[data_start-test_start_period]]).astype(float).astype('datetime64[s]')[0], color = 'purple', label = 'begin trading')
            axs1.legend(['Ratio','5d Ratio MA', '60d Ratio MA', 'Mean Ratio'])
    

            # Plot 2: Ratios and buy and sell signals from z score
            

            plt.ylabel('Ratio')
            plt.xlabel('Time')
            #plt.show()

    trader = backtester.Trader(sample_window, test_window, pairs,dataset, data_start-test_start_period, test_end_period, zenter, zexit, profitpercent)
    return trader.percent, trader.realized_pnl, trader.unrealized_pnl
            
print(test_trading_strategy())