from math import sqrt
import numpy as np
class NormalizedReturnSeries:
    """Instantiated with an list of prices in chronoligical ascending order, creates a normalized return series (list) accessible
    by .series attribute"""
    def build_return_series(self):
        # Constructs the UnNormalized Return Series
        i = 0
        while i < len(self.series)-1:
            price_at_t, price_at_t_minus_1 = self.series[i+1], self.series.pop(i)
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

class DataSetMatrices:
    """A DataSetMatrices object is instantiated a CryptoDataset object. It has the standardized Covariance matrix
    generated from the return series for each coin in the dataset. This class has two public attributes:
    - .correlation_matrix is an np array representing a symmetric correlation matrix w/ <number_of_coins> rows and columns. To access this attr, 
    MUST call .build_correlation_matrix() method first.
    - .data_matrix is the matrix (np array) of the stacked normalized return series"""

    def build_data_matrix(self):
        """From a CryptoDataset, builds a data matrix of the stacked normalized return series. .order_list is a tuple showing the order (top to bottom)
        of the coins stacked in the data matrix"""

        self.order_list = list(self.dataset.historical_coin_data.keys())
        self.data_matrix = np.array([NormalizedReturnSeries(self.dataset.historical_coin_data[coin]).series for coin in self.order_list])

    def build_correlation_matrix(self):
        """Creates Empirical Correlation Matrix accessible with .correlation_matrix attribute"""
        dim_correleation_matrix, divisor = len(self.order_list), self.data_matrix.shape[1] - 1
        self.correlation_matrix = (self.data_matrix @ self.data_matrix.T)/(self.data_matrix.shape[1] - 1)

    def __init__(self, dataset):
        self.dataset = dataset
        self.build_data_matrix()
        self.build_correlation_matrix()

def apply_PCA(dataset, k):
    """Returns dimension reduced on dataset using k top eigenvalues/vectors of the correlation matrix."""
    matrices = DataSetMatrices(dataset)
    data_matrix, correlation_matrix = matrices.data_matrix, matrices.correlation_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    feature_vector = np.delete(eigenvectors[:,eigenvalues.argsort()[::-1]], [i for i in range(k, len(eigenvalues))], axis=1)
    proj_data_matrix = data_matrix.dot(feature_vector)
    print(proj_data_matrix)
    





