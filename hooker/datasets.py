
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import requests

import random
import math

__author__ = "Guillaume Chevalier"
__license__ = "MIT"
__version__ = "2017-03"

DEFAULT_SEED = 123456

class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]
        if hasattr(self, 'track_ids'):
            self.track_ids = self.track_ids[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch

    # Python 3.x compatibility
    def __next__(self):
        return self.next()
    
class MultiTSDataProvider(DataProvider):
    
    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, n_pred=60, n_cond=60, stride_length=10,
                 scaler=StandardScaler(), base_dir = '../data/', n_ts=100):
        self.n_pred = n_pred # number of values to predict
        self.n_cond = n_cond # number of values on which to condition predictions
        full_df = pd.read_csv(base_dir+'train_1.csv', nrows=n_ts)
        dates = [c for c in full_df.columns if c !='Page']
        val_dates = dates[-(n_pred+n_cond):]
        self.val_dates = val_dates
        if which_set == 'train':
            inputs = full_df.drop(['Page'] + val_dates, axis=1).fillna(0).values
        elif which_set == 'val':
            inputs = full_df[val_dates].fillna(0).values
          # each ts should have 0 mean and unit variance
        if scaler:
            # need to change the way this behaves
            inputs = scaler.fit_transform(inputs)
            print(np.max(inputs))
        print(inputs.shape)
        
        if which_set == 'train':
            window_length = n_cond + n_pred
            n_windows = int(np.floor(np.divide(inputs.shape[1] - window_length,
                                               stride_length) + 1))
            start_index = 0
            window_array = np.ndarray((inputs.shape[0], n_windows, window_length))
            for i in range(n_windows):
                window_array[:,i,:] = inputs[:, start_index:start_index+window_length]
                start_index += stride_length
            print(window_array.shape)
            window_array = window_array.reshape((-1,window_length,1))
            print(window_array.shape)
            print('{} overlapping windows of length {} in training date range'.format(n_windows, window_length))
        #         inputs = w.reshape((-1, window_length, inputs.shape[-1])) # reshape inputs to (n_sample, length, n_features)
            targets = window_array[:,self.n_cond:,:] # shifted one along from inputs - so there is one target val which is never input, into either decoder or encoder
            inputs = np.pad(window_array[:,:-1,:],((0,0), (1,0), (0,0)),mode='constant')
            
        #         inputs, targets = inputs[:,:self.n_cond,:], inputs[:,self.n_cond:,:]
            inputs = inputs.reshape((inputs.shape[0], -1))
            targets = targets.reshape((targets.shape[0], -1))
        
        elif which_set == 'val':
            targets = inputs[:, self.n_cond:]
            inputs = np.pad(inputs[:,:self.n_cond], ((0,0), (1,n_pred-1)), mode='constant') # decoder inputs is now 1 useful num followed by a bunch of zeros, in principle
        
        print(inputs.shape, targets.shape)
        super(MultiTSDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)


def generate_x_y_data_v1(isTrain, batch_size):
    """
    Data for exercise 1.

    returns: tuple (X, Y)
        X is a sine and a cosine from 0.0*pi to 1.5*pi
        Y is a sine and a cosine from 1.5*pi to 3.0*pi
    Therefore, Y follows X. There is also a random offset
    commonly applied to X an Y.

    The returned arrays are of shape:
        (seq_length, batch_size, output_dim)
        Therefore: (10, batch_size, 2)

    For this exercise, let's ignore the "isTrain"
    argument and test on the same data.
    """
    seq_length = 10

    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        rand = random.random() * 2 * math.pi

        sig1 = np.sin(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, seq_length * 2))
        sig2 = np.cos(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, seq_length * 2))
        x1 = sig1[:seq_length]
        y1 = sig1[seq_length:]
        x2 = sig2[:seq_length]
        y2 = sig2[seq_length:]

        x_ = np.array([x1, x2])
        y_ = np.array([y1, y2])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batch_x, batch_y


def generate_x_y_data_two_freqs(isTrain, batch_size, seq_length):
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sig1 = amp_rand * np.sin(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        )
        )

        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() * 1.2

        sig1 = amp_rand * np.cos(np.linspace(
            seq_length / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seq_length / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seq_length * 2
        )
        ) + sig1

        x1 = sig1[:seq_length]
        y1 = sig1[seq_length:]

        x_ = np.array([x1])
        y_ = np.array([y1])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batch_x, batch_y


def generate_x_y_data_v2(isTrain, batch_size):
    """
    Similar the the "v1" function, but here we generate a signal with
    2 frequencies chosen randomly - and this for the 2 signals. Plus,
    the lenght of the examples is of 15 rather than 10.
    So we have 30 total values for past and future.
    """
    return generate_x_y_data_two_freqs(isTrain, batch_size, seq_length=15)


def generate_x_y_data_v3(isTrain, batch_size):
    """
    Similar to the "v2" function, but here we generate a signal
    with noise in the X values. Plus,
    the lenght of the examples is of 30 rather than 10.
    So we have 60 total values for past and future.
    """
    seq_length = 30
    x, y = generate_x_y_data_two_freqs(
        isTrain, batch_size, seq_length=seq_length)
    noise_amount = random.random() * 0.15 + 0.10
    x = x + noise_amount * np.random.randn(seq_length, batch_size, 1)

    avg = np.average(x)
    std = np.std(x) + 0.0001
    x = x - avg
    y = y - avg
    x = x / std / 2.5
    y = y / std / 2.5

    return x, y


def loadCurrency(curr, window_size):
    """
    Return the historical data for the USD or EUR bitcoin value. Is done with an web API call.
    curr = "USD" | "EUR"
    """
    # For more info on the URL call, it is inspired by :
    # https://github.com/Levino/coindesk-api-node
    r = requests.get(
        "http://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-17&end=2017-03-03&currency={}".format(
            curr
        )
    )
    data = r.json()
    time_to_values = sorted(data["bpi"].items())
    values = [val for key, val in time_to_values]
    kept_values = values[1000:]

    X = []
    Y = []
    for i in range(len(kept_values) - window_size * 2):
        X.append(kept_values[i:i + window_size])
        Y.append(kept_values[i + window_size:i + window_size * 2])

    # To be able to concat on inner dimension later on:
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    return X, Y


def normalize(X, Y=None):
    """
    Normalise X and Y according to the mean and standard deviation of the X values only.
    """
    # # It would be possible to normalize with last rather than mean, such as:
    # lasts = np.expand_dims(X[:, -1, :], axis=1)
    # assert (lasts[:, :] == X[:, -1, :]).all(), "{}, {}, {}. {}".format(lasts[:, :].shape, X[:, -1, :].shape, lasts[:, :], X[:, -1, :])
    mean = np.expand_dims(np.average(X, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(X, axis=1) + 0.00001, axis=1)
    # print (mean.shape, stddev.shape)
    # print (X.shape, Y.shape)
    X = X - mean
    X = X / (2.5 * stddev)
    if Y is not None:
        assert Y.shape == X.shape, (Y.shape, X.shape)
        Y = Y - mean
        Y = Y / (2.5 * stddev)
        return X, Y
    return X


def fetch_batch_size_random(X, Y, batch_size):
    """
    Returns randomly an aligned batch_size of X and Y among all examples.
    The external dimension of X and Y must be the batch size (eg: 1 column = 1 example).
    X and Y can be N-dimensional.
    """
    assert X.shape == Y.shape, (X.shape, Y.shape)
    idxes = np.random.randint(X.shape[0], size=batch_size)
    X_out = np.array(X[idxes]).transpose((1, 0, 2))
    Y_out = np.array(Y[idxes]).transpose((1, 0, 2))
    return X_out, Y_out

X_train = []
Y_train = []
X_test = []
Y_test = []


def generate_x_y_data_v4(isTrain, batch_size):
    """
    Return financial data for the bitcoin.

    Features are USD and EUR, in the internal dimension.
    We normalize X and Y data according to the X only to not
    spoil the predictions we ask for.

    For every window (window or seq_length), Y is the prediction following X.
    Train and test data are separated according to the 80/20 rule.
    Therefore, the 20 percent of the test data are the most
    recent historical bitcoin values. Every example in X contains
    40 points of USD and then EUR data in the feature axis/dimension.
    It is to be noted that the returned X and Y has the same shape
    and are in a tuple.
    """
    # 40 pas values for encoder, 40 after for decoder's predictions.
    seq_length = 40

    global Y_train
    global X_train
    global X_test
    global Y_test
    # First load, with memoization:
    if len(Y_test) == 0:
        # API call:
        X_usd, Y_usd = loadCurrency("USD", window_size=seq_length)
        X_eur, Y_eur = loadCurrency("EUR", window_size=seq_length)

        # All data, aligned:
        X = np.concatenate((X_usd, X_eur), axis=2)
        Y = np.concatenate((Y_usd, Y_eur), axis=2)
        X, Y = normalize(X, Y)

        # Split 80-20:
        X_train = X[:int(len(X) * 0.8)]
        Y_train = Y[:int(len(Y) * 0.8)]
        X_test = X[int(len(X) * 0.8):]
        Y_test = Y[int(len(Y) * 0.8):]

    if isTrain:
        return fetch_batch_size_random(X_train, Y_train, batch_size)
    else:
        return fetch_batch_size_random(X_test,  Y_test,  batch_size)
