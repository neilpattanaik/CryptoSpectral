import algorithms # Import dataprocessing.py 
import cryptodata # Import cryptodata.py 
import numpy as np 
from statsmodels.tsa.stattools import coint
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd

def halflife(pairsdictlist,price_matrix):
    df = pd.DataFrame() 
    df['timeseries'] = pd.Series(price_matrix[list(pairsdictlist[0].values())[0],:])/pd.Series(price_matrix[list(pairsdictlist[1].values())[0],:])
    df['ylag'] = df['timeseries'].shift(1)
    df['deltaY'] = df['timeseries'] - df['ylag']
    results = smf.ols('deltaY ~ ylag', data=df).fit()
    lam = results.params['ylag']
    halflife=-np.log(2)/lam
    return 1 <= halflife and halflife <= 168


def cointegration_check(clustered_data_nameindexpairs, training_data_matrix):
    cointegrated_pairs = []
    for cluster in clustered_data_nameindexpairs:
        keys = list(cluster.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                # Uses Engle-Granger test
                p_value = coint(training_data_matrix[cluster[keys[i]],:], training_data_matrix[cluster[keys[j]],:])[1]
                # If p-value is low, then null hypothesis (i.e., that i,j are not cointegrated) is unlikely and can be rejected.
                if p_value <= 0.01:
                    cointegrated_pairs.append([{keys[i]:cluster[keys[i]]}, {keys[j]:cluster[keys[j]]}])
    return cointegrated_pairs

def hurst_exponent(pair, training_data_matrix):
    with np.errstate(divide='raise'):
        try:
            time_series = training_data_matrix[list(pair[0].values())[0],:] / training_data_matrix[list(pair[1].values())[0],:]
        except FloatingPointError:
            return 1
    lags = range(2,100)
    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


def spectral_pair_search(training_data_matrix, price_matrix, coins, dimensions, clusters):
    """Returns a list of lists, with each sublist containing dictionaries with 2 <coin name string>:<row index in data matrix> entries corresponding
    to a pair.
    Takes in a data matrix, list of coin names corresponding to their order in the data matrix, the number of dimensions for PCA,
    and the number of clusters for spectral clustering."""
    # Reduce dimensionality and cluster coins
    pca_reduced_datamatrix, explained_variances = algorithms.PCA(training_data_matrix[:], dimensions)
    clustered_data_indices = algorithms.spectral_clustering(pca_reduced_datamatrix, clusters = clusters)
    clustered_data_nameindexpairs = []
    for cluster in clustered_data_indices:
        cluster_dict = {}
        for i in range(len(cluster)):
           cluster_dict[coins[cluster[i]]] = cluster[i]
        if cluster_dict:
            clustered_data_nameindexpairs.append(cluster_dict)
    
    # Check for pairs in each cluster that are cointegrated
    cointegrated_pairs = cointegration_check(clustered_data_nameindexpairs, price_matrix)
    cointegrated_stationary_pairs = list(filter(lambda x: hurst_exponent(x, price_matrix) < 0.5, cointegrated_pairs))
    cointegrated_stationary_pairs_halflifereverting = list(filter(lambda x: halflife(x, price_matrix), cointegrated_stationary_pairs))
    return cointegrated_stationary_pairs_halflifereverting


def spectral_pairing(number_of_coins, data_start, test_start_period, interval, dimensions, clusters):
    dataset = cryptodata.CryptoDataset(number_of_coins, data_start, interval)
    data_matrix,price_matrix= dataset.data_matrix, dataset.price_matrix        
    return spectral_pair_search(data_matrix[:,:data_start-test_start_period],price_matrix[:,:data_start-test_start_period], dataset.coins, dimensions, clusters), data_matrix, price_matrix, dataset
