import numpy as np
import os

class Scaler:
    def __init__(self):
        self.train_matrix = None
        self.axis = None
        self.min = None
        self.max = None
        self.mu = None
        self.sigma = None

    def fit(self, train_matrix, axis, ddof=0):
        """
        Initializes attributes.

        Parameters:
        train_matrix (ndarray): Matrix containing training data, with samples along axis=0.
        axis (int or tuple of ints): Axes along which to calculate min, max, mean and standard deviation.
        ddof (int): Delta degrees of freedom, relevant only if scale_type='standardize' in transform (parameter passed to numpy.std).

        Returns:
        None
        """
        self.train_matrix = train_matrix
        self.axis = axis
        self.min = np.min(train_matrix, axis=axis)
        self.max = np.max(train_matrix, axis=axis)
        self.mu = np.mean(train_matrix, axis=axis)
        self.sigma = np.std(train_matrix, axis=axis, ddof=ddof)

    def transform(self, matrix, scale_type):
        """
        Normalizes or standardizes input matrix using the initialized using fit. Make sure matrix is normalized/standardized using attributes initialized using train_matrix.

        Parameters:
        matrix (ndarray): Matrix to be normalized/standardized.
        scale_type (str): Can be 'normalize' or 'standardize' depending on whether to normalize or standardize, respectively.

        Returns:
        ndarray: Normalized/standardized matrix.
        """

        if self.min is None or self.max is None or self.mu is None or self.sigma is None:
            raise ValueError('fit must be called on train_matrix before transform can be called.')

        if scale_type == 'normalize':
            return self._normalize(matrix, self.min, self.max)

        elif scale_type == 'standardize':
            return self._standardize(matrix, self.mu, self.sigma)

    def _normalize(self, matrix, min_, max_):
        """
        Normalizes matrix along one or more axes using minimum and maximum values as min_ and max_ respectively.

        For regular structured data with samples along axis=0 and features along axis=1, provide axis=0 as parameter to find min and max along the samples axis.

        For time series data with shape (num_samples, num_timesteps, num_features) pass axis=(0, 1) to find min and max across all samples and all timesteps.

        Parameters:
        matrix (ndarray): NumPy array with any ndim, to be normalized.
        min_ (float or ndarray with ndim=1): Minimum value to use for normalization.
        max_ (float or ndarray with ndim=1): Maximum value to use for normalization.

        Returns:
        ndarray: Normalized matrix.
        """

        matrix_normalized = (matrix - min_) / (max_ - min_)

        # The following assertion will hold for normalized train matrix but not for normalized test matrix.
        # assert ((matrix_normalized >= 0) & (matrix_normalized <= 1)).all()

        return matrix_normalized

    def _standardize(self, matrix, mu, sigma):
        """
        Standardizes matrix along one or more axes using mean and standard deviation values as mu and sigma respectively.

        For regular structured data with samples along axis=0 and features along axis=1, provide axis=0 as parameter to find mean and standard deviation along the samples axis.

        For time series data with shape (num_samples, num_timesteps, num_features) pass axis=(0, 1) to find mean and standard deviation across all samples and all timesteps.

        Parameters:
        matrix (ndarray): NumPy array with any ndim, to be standardized.
        mu (float or ndarray with ndim=1): Mean values to use for standardization.
        sigma (float or ndarray with ndim=1): Standard deviation values to use for standardization.

        Returns:
        ndarray: Standardized matrix.
        """

        return (matrix - mu) / sigma

if __name__ == '__main__':
    dirpath = os.path.join('..', 'data_rnn', 'data_breath')

    train_matrix = np.load(os.path.join(dirpath, 'train_X.npy'))
    test_matrix = np.load(os.path.join(dirpath, 'test_X.npy'))

    sc = Scaler()
    sc.fit(train_matrix, (0, 1))

    train_matrix_normalized = sc.transform(train_matrix, 'normalize')
    train_matrix_standardized = sc.transform(train_matrix, 'standardize')
    test_matrix_normalized = sc.transform(test_matrix, 'normalize')
    test_matrix_standardized = sc.transform(test_matrix, 'standardize')

    assert len(sc.min) == len(sc.max) == len(sc.mu) == len(sc.sigma) == train_matrix.shape[-1]

    ROUND = 3
    np.set_printoptions(suppress=True)

    print('min:', np.round(sc.min[:6], ROUND))
    print('max:', np.round(sc.max[:6], ROUND))
    print('mu:', np.round(sc.mu[:6], ROUND))
    print('sigma:', np.round(sc.sigma[:6], ROUND))

    print('train:', np.round(train_matrix[0, 100, :6], ROUND))
    print('train_n:', np.round(train_matrix_normalized[0, 100, :6], ROUND))
    print('train_s:', np.round(train_matrix_standardized[0, 100, :6], ROUND))

    print('test:', np.round(test_matrix[0, 100, :6], ROUND))
    print('test_n:', np.round(test_matrix_normalized[0, 100, :6], ROUND))
    print('test_s:', np.round(test_matrix_standardized[0, 100, :6], ROUND))