import pca
import cryptodata

def identify_pairs(number_of_coins, timeframe, interval, k):
    dataset = cryptodata.CryptoDataset(number_of_coins, timeframe, interval)
    pca.apply_PCA(dataset, k)


identify_pairs(10,10,"day", 3)
print("done")