import requests
import os
import sqlite3
from datetime import datetime, timezone
import time
from math import sqrt
import numpy as np


class CryptoDataset:
    """CryptoDataset is instantiated with 3 parameters:
    - number_of_coins (integer between 1 and 100)
    - timeframe (integer between 1 and 720 for hourly interval, or any integer greater than 1 for daily/weekly)
    - interval (string, must be "hour", "day", or "week") 

    Has three attributes that should be accessed by other classes:
    - .data_matrix is the nxp matrix (n coins, p timestamps) of the stacked normalized return series
    - .coins is the list of coin names ordered as in the data matrix
    - .creation_date is the date (UTC) that the CryptoDataset was created
    - .historical_coin_data is a dictionary with string keys and chronological (ascending) list of prices value pairs. i.e, {'BTC':[69432.53, 72532.45,...]}
    """
    def get_sql():
        # Returns sqlite3 cursor object and database connect object
        db = sqlite3.connect('historicaldata.db')
        cursor = db.cursor()
        return cursor, db
        
    def set_api_key(self):
        # Sets the CryptoCompare API Key as an environmental variable and attribute of the CryptoDataset object
        try:
            self._api_key = os.environ['API_KEY']
        except:
            os.environ['API_KEY'] = input('Enter CryptoCompare API Key: ')
            self._api_key = os.environ['API_KEY']
    
    def get_coin_list(self):
        """ Gets the symbols of top 100 coins by current market cap and saves to a table called coinlist_<Month_dd_yyyy>>. All dates/times are in UTC.
        Then, updates object's coinlist."""

        cursor, db = CryptoDataset.get_sql()
        self.creation_date = datetime.now(timezone.utc).strftime("%b_%d_%Y")
        table_name = "coinlist_"+self.creation_date

        # If table for today's coinlist doesn't exist in database, then call CryptoCompare API to get it
        if not cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)).fetchall():
            coin_list_full_data = requests.get('https://min-api.cryptocompare.com/data/top/mktcapfull', params={'limit':100, 'tsym':'USD', 'api_key':self._api_key}).json()['Data']
            names_and_market_cap_ranks = [(coin_list_full_data[i]['CoinInfo']['Name'], i) for i in range(len(coin_list_full_data))]
            cursor.execute("CREATE TABLE " + table_name + "(name, cap_rank);")
            cursor.executemany("INSERT INTO " + table_name + "(name, cap_rank) VALUES(?, ?);", names_and_market_cap_ranks)
            db.commit()

        # Queries the first <number_of_coins> coins from today's coinlist, and saves the list of tuples [('name_1',), ('name_2',), ...] to coin_list attribute
        self.coin_list = cursor.execute("SELECT name FROM "+table_name+" ORDER BY cap_rank LIMIT ?;", (self.number_of_coins,)).fetchall()
    
    def get_coin_historical_data(self, coin, timestamp, table_name):
        """Takes Coin & Timestamp and makes sure data at necessary interval is present upto and including timestamp"""

        cursor, db = CryptoDataset.get_sql()
        seconds_per_interval = 3600 if table_name == "hour" else 86400 if table_name == "day" else 604800
        coin_api_data = requests.get("https://min-api.cryptocompare.com/data/v2/histo"+self._api_endpoint, params={'limit':min(max((self._start_time-timestamp)/seconds_per_interval, 1), 2000), 'tsym':'USD', 'api_key':self._api_key, 'fsym':coin[0], 'aggregate':self._aggregate, 'aggregatePredictableTimePeriods':'false'}).json()['Data']
        last_received_timestamp =  coin_api_data['TimeFrom']
        while last_received_timestamp > timestamp:
            new_coin_api_data = requests.get("https://min-api.cryptocompare.com/data/v2/histo"+self._api_endpoint, params={'toTs':last_received_timestamp,'limit':min(max((1, last_received_timestamp-timestamp)/seconds_per_interval), 2000), 'tsym':'USD', 'api_key':self._api_key, 'fsym':coin[0], 'aggregate':self._aggregate, 'aggregatePredictableTimePeriods':'false'}).json()['Data']
            coin_api_data['Data'].append(new_coin_api_data['Data'])
            last_received_timestamp = new_coin_api_data['TimeFrom']
        for datapoint in coin_api_data['Data']:
            if datapoint['time'] >= timestamp:
                cursor.execute("UPDATE " + table_name + " SET t" + str(datapoint['time'])+"="+str(datapoint['close'])+" WHERE name=?;", coin)
                db.commit()


    def get_historical_data(self):
        """Checks if database has sufficient historical data for the given timeframe and interval for each coin in coinlist.
        If any coin lacks sufficient data, gets necessary data from the CryptoCompare API and adds it to the corresponding table (hourTable, dayTable, weekTable).
        Creates the necessary table if it does not exist. Dictionary {'name':[<closing price at time k=0,...,timeframe>]} of closing prices is bound to historical_closing_data attribute,
        where time k=timeframe corresponds to the current time (i.e, if interval=day, time at k=timeframe is current, time at k=timeframe-1 is yesterday, etc)."""
        
        cursor, db = CryptoDataset.get_sql()
        table_name = self.interval
        self.historical_coin_data = {}


        # Creates table for interval if it doesn't exist
        if not cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)).fetchall():
            cursor.execute("CREATE TABLE " + table_name + "(name TEXT PRIMARY KEY);")
            db.commit()
        
        # Checks if each coin is in table, and adds it if not. 
        cursor.executemany("INSERT OR IGNORE INTO " + table_name + "(name) VALUES(?)", self.coin_list)
        db.commit()
        

        # Makes a list of necessary timestamps
        unixtime = time.time() # Current unix time
        if table_name == 'day':
            self._start_time = unixtime - unixtime % 86400
            self._timestamps = list(reversed([int(self._start_time - (86400 * i)) for i in range(self.timeframe)])) # First entry is most recent unix midnight timestamp, each subsequent entry is previous midnight
        elif table_name == 'hour':
            self._start_time = unixtime - unixtime % 3600
            self._timestamps = list(reversed([int(self._start_time - (3600 * i)) for i in range(self.timeframe)])) # First entry is most recent hour's unix timestamp, each subsequent entry is previous hour's stamp
        else: # week case
            self._start_time = unixtime - unixtime % 86400
            self._timestamps = list(reversed([int(self._start_time - (604800 * i)) for i in range(self.timeframe)])) # First entry is most recent midnight's unix timestamp, each subsequent entry is timestamp of 7 days prior's midnight 

        # Check if all necessary timestamp columns exist. If they don't, then add them. 
        existing_timestamp_cols = [column[1] for column in cursor.execute('PRAGMA table_info('+table_name+');').fetchall()]
        for timestamp in self._timestamps:
            if not 't'+str(timestamp) in existing_timestamp_cols:
                cursor.execute("ALTER TABLE " + table_name + " ADD COLUMN t" + str(timestamp) + " REAL;")
        
        
        # Checks if coin has sufficient data, and gets the price/time data from the API if necessary.
        self._api_endpoint, self._aggregate = "hour" if table_name == "hour" else "day", 7 if table_name == "week" else 1  # Weekly data comes from day data with aggregate of 7
        for timestamp in self._timestamps:
            for coin_needing_data in cursor.execute("SELECT name FROM " + table_name + " WHERE t" + str(timestamp) + " IS NULL;").fetchall():
                if coin_needing_data in self.coin_list:
                    self.get_coin_historical_data(coin_needing_data, timestamp, table_name)

        # Once each coin has price data for each timestamp, get the price data in chronoligical order and save to dictionary
        coin_time_series = []
        for coin in self.coin_list:
            coin_time_series.clear()
            for timestamp in self._timestamps:
                coin_time_series.append(cursor.execute("SELECT t" +str(timestamp) + " FROM " +table_name+" WHERE name=?", coin).fetchall()[0])
            self.historical_coin_data[coin[0]] = list(map(lambda x: float(x[0]), coin_time_series))

    def __init__(self, number_of_coins, timeframe, interval):
        cursor, db = CryptoDataset.get_sql()
        # Ensures parameters are valid
        if interval not in ["hour", "day", "week"]:
            raise Exception("Interval is not valid. Must be hour, day, or week.")
        elif not type(number_of_coins is int) or 0 >= number_of_coins or 100 < number_of_coins:
            raise Exception("Invalid coin number")
        elif not type(timeframe) is int:
            raise Exception("Non-integer timeframe")

        # Ensures API Key is set-up
        self.set_api_key()
        # Assigns attributes upon instantiation
        self.number_of_coins, self.timeframe, self.interval = number_of_coins, timeframe, interval
        # Gets coinlist and assigns it to coin_list attribute
        self.get_coin_list()
        # Gets historical data for coinlist and timeframe
        self.get_historical_data()
        # Build Data Matrix (nxp, where n is number of coins and p is number of timestamps)
        returns_matrix_object =DataSetMatrix(self, True)
        self.data_matrix = returns_matrix_object.data_matrix
        # Set .coins attr
        self.coins = returns_matrix_object.order_list
        
        price_matrix_object = DataSetMatrix(self, False)
        self.price_matrix = price_matrix_object.data_matrix
        
        

class NormalizedReturnSeries:
    """Instantiated with an list of prices in chronoligical ascending order, creates a normalized return series (list) accessible
    by .series attribute"""
    def build_return_series(self):
        # Constructs the UnNormalized Return Series
        i = 0
        while i < len(self.series)-1:
            price_at_t, price_at_t_minus_1 = self.series[i+1], self.series.pop(i) or 1
            self.series.insert(i, (price_at_t-price_at_t_minus_1)/price_at_t_minus_1)
            i += 1
        self.series.pop()

    def get_mean(self):
        self.mean = sum(self.series)/len(self.series)
    
    def get_standard_deviation(self):
        self.standard_deviation = sqrt(sum([pow(R - self.mean, 2) for R in self.series])/(len(self.series)-1))

    def normalize(self):
        for i in range(len(self.series)):
            self.series[i] = (self.series[i] - self.mean)/self.standard_deviation

    def __init__(self, input_list):
        # Creates UnNormalized Return Series
        self.series = input_list[:]
        self.build_return_series()

        # Normalize the Return Series
        self.get_mean()
        self.get_standard_deviation()
        self.normalize()

class DataSetMatrix:
    """A DataSetMatrices object is instantiated a CryptoDataset object. It has the standardized Covariance matrix
    generated from the return series for each coin in the dataset. This class has three public attributes:
    - .correlation_matrix is an np array representing a symmetric correlation matrix w/ <number_of_coins> rows and columns. To access this attr, 
    MUST call .build_correlation_matrix() method first.
    - .data_matrix is the matrix (np array) of the stacked normalized return series
    - .order_list is the list of string coin names ordered as in the rows of the data matrix"""

    def build_data_matrix(self, returns):
        """From a CryptoDataset, builds a data matrix of the stacked normalized return series. .order_list is a tuple showing the order (top to bottom)
        of the coins stacked in the data matrix"""
        arr_list =  []
        if returns:
            self.order_list = list(self.dataset.historical_coin_data.keys())
            last_index, del_list = self.dataset.number_of_coins-1, []
            for i in range(len(self.order_list)):
                if self.dataset.historical_coin_data[self.order_list[i]][0] and self.dataset.historical_coin_data[self.order_list[i]][last_index]:
                    arr_list.append(NormalizedReturnSeries(self.dataset.historical_coin_data[self.order_list[i]]).series)
                else:
                    del_list.append(i)
            for index in sorted(del_list, reverse=True):
                self.order_list.pop(index)
        else:
            self.order_list = list(self.dataset.coins)
            for i in range(len(self.order_list)):
                arr_list.append(self.dataset.historical_coin_data[self.order_list[i]])
        self.data_matrix = np.array(arr_list)


        

    def __init__(self, dataset, returns):
        self.dataset = dataset
        self.build_data_matrix(returns)

