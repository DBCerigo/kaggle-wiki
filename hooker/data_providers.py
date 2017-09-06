import numpy as np
import pandas as pd

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
    
    def __init__(self, full_scaled_inputs, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None, n_pred=60, n_cond=60, stride_length=10,
                 base_dir = '../data/', n_ts=None, echo_input=False):
        self.n_pred = n_pred # number of values to predict
        self.n_cond = n_cond # number of values on which to condition predictions
        if n_ts is None:
            selected_inputs = full_scaled_inputs
        else:
            self.sampled_rows = np.random.choice(full_scaled_inputs.shape[0], n_ts, replace=False)
            selected_inputs = full_scaled_inputs[self.sampled_rows, :]
        if which_set == 'train':
            inputs = selected_inputs[:,:-(n_pred+n_cond)]
        elif which_set == 'val':
            inputs = selected_inputs[:,-(n_pred+n_cond):]
          # each ts should have 0 mean and unit variance
        if stride_length == 0:
            stride_length = n_cond
        print(inputs.shape)
        
        if which_set == 'train':
            if stride_length > 0:
                window_length = n_cond + n_pred
                n_windows = int(np.floor(np.divide(inputs.shape[1] - window_length,
                                                   stride_length) + 1))
                start_index = 0
                window_array = np.ndarray((inputs.shape[0], n_windows, window_length))
                for i in range(n_windows):
                    window_array[:,i,:] = inputs[:, start_index:start_index+window_length]
                    start_index += stride_length
                inputs = window_array.reshape((-1,window_length))
                print('{} overlapping windows of length {} in training date range'.format(n_windows, window_length))
                inputs = window_array[:,:n_cond]
                targets = window_array[:,n_cond:]
            elif stride_length == -1:
                targets = inputs[:,-n_pred:]
                inputs = inputs[:,-(n_cond+n_pred):-n_pred]
        
        elif which_set == 'val':
            targets = inputs[:, self.n_cond:]
            inputs = inputs[:,:self.n_cond] # decoder inputs is now 1 useful num followed by a bunch of zeros, in principle
        
        print(inputs.shape, targets.shape)
        if echo_input:
            targets = inputs
        super(MultiTSDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)