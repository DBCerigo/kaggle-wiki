import os
import datetime
import numpy as np

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.layers.core import Dense
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib.legacy_seq2seq import tied_rnn_seq2seq, rnn_decoder
from tensorflow.contrib.rnn import DropoutWrapper

from utils import clock
from helpers import InferenceHelper
from datasets import generate_x_y_data_v3


class BaseSeq2Seq:

    def __init__(self, n_cond, n_pred):
        self.n_cond = n_cond
        self.n_pred = n_pred

    def train_sine(self, epochs=100, report_error_avg=10, batch_size=100):
        # input_batch, target_batch = val_data.next()
        for e in range(epochs):
            running_error = 0.
            val_error = 0.
            mean_running_error = 0.
            with clock():
                for i in range(report_error_avg):
                    input_batch, target_batch = generate_x_y_data_v3(True, batch_size)
                    input_batch = input_batch.reshape(input_batch.shape[0],-1).T
                    target_batch = target_batch.reshape(target_batch.shape[0],-1).T
                    #     print(input_batch.shape, target_batch.shape)
                    feed_dict = {self.train: True}
                    # each decoder input is batch size x 1
                    feed_dict = self.feed_vals(input_batch, target_batch, feed_dict)
                    _, err = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    mean_preds = np.mean(input_batch[:,:], axis=1).reshape(-1,1)
                    mean_errs = np.abs(mean_preds - target_batch)
                    batch_mean_err = np.sum(np.mean(mean_errs, axis=0))
                    running_error += err
                    mean_running_error += batch_mean_err
                running_error /= report_error_avg
                mean_running_error /= report_error_avg
            print("""End of epoch {0}: running error average = {1:.3f}
                     mean error average = {2:.3f}""".format(e + 1, running_error, mean_running_error))


    def train(self,
              train_data, 
              valid_data, 
              epochs=20,
              keep_prob=0.7,
              logdir='tf-log',
              save=False,
              close_session=False):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stats_writer = tf.summary.FileWriter(os.path.join(logdir,
                                                          timestamp),
                                             graph=self.graph)
        model_dir = 'checkpoints/{}-{}-{}'.format(timestamp, self.n_cond, self.n_pred)
        if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        for e in range(epochs):
            running_error, val_error = 0., 0.
            median_running_error = 0.
            with clock():
                for input_batch, target_batch in train_data:
        #     print(input_batch.shape, target_batch.shape)
                    feed_dict = {self.is_train: True, self.keep_prob: keep_prob}
                    # each decoder input is batch size x 1
                    feed_dict = self.feed_vals(input_batch, target_batch, feed_dict)
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    median_preds = np.median(input_batch[:,:], axis=1).reshape(-1,1)
                    median_errs = np.abs(median_preds - target_batch)
                    batch_median_err = np.mean(median_errs)
                    running_error += err
                    median_running_error += batch_median_err
                running_error /= train_data.num_batches
                median_running_error /= train_data.num_batches
                if save:
                    self.saver.save(self.sess, model_dir+'/model.ckpt'.format(timestamp), global_step=e+1)

            for input_batch, target_batch in valid_data:
                feed_dict = {self.is_train: False, self.keep_prob: 1.0}
                feed_dict = self.feed_vals(input_batch, target_batch, feed_dict)
                val_err = self.sess.run(self.loss, feed_dict=feed_dict)
                # this time we don't need to feed in either decoder_inputs or targets
                val_error += val_err
            val_error /= valid_data.num_batches

            summary = tf.Summary(value=[
                                tf.Summary.Value(tag="train_error", simple_value=running_error), 
                                tf.Summary.Value(tag="valid_error", simple_value=val_error), 
                            ])
            # http://stackoverflow.com/questions/37902705/how-to-manually-create-a-tf-summary
            stats_writer.add_summary(summary, e)

            print("""End of epoch {0}: running error average = {1:.3f}
                     median error average = {2:.3f}
                     val error average = {3:.3f}""".format(e + 1, running_error, median_running_error, val_error))

        if close_session:
            self.sess.close()

    def restore_session(self, meta_file):
        self.sess = tf.Session(graph=self.graph)
        l = meta_file.split('/')
        filename = l[-1]
        dirname = '/'.join(l[:-1])
        if not len(dirname) or dirname[0] != '/':
            dirname='./' + dirname
        new_saver = tf.train.import_meta_graph(meta_file)
        new_saver.restore(self.sess, tf.train.latest_checkpoint(dirname))

    def view_preds(self, inputs, targets, n_view=10, one_step=False):
        preds = self.predict(inputs, targets, one_step)
        for i in range(n_view):
            fig = plt.figure()
            print(i)
            plt.plot(inputs[i,:], color='blue')
            plt.plot([self.n_cond+j for j in range(self.n_pred)], targets[i,:], color='blue')
            plt.plot([self.n_cond+j for j in range(self.n_pred)], preds[i,:], color='red')
            plt.show()


class MixedSeq2Seq(BaseSeq2Seq):

    def __init__(self, n_cond, n_pred, hidden_dim,
                 n_layers=2,
                 input_dim=1,
                 learning_rate=0.01,
                 output_dim=1,
                 cell_type='GRU'):
        """
        Construct graph
        """
        super().__init__(n_cond, n_pred)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.is_train = tf.placeholder(tf.bool)


            self.inputs = tf.placeholder(tf.float32, shape=(None, n_cond, input_dim))
            
            # Decoder: expected outputs
            self.expected_sparse_output = [
                tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
                  for t in range(n_pred)
            ]

            self.go_sym = tf.placeholder(tf.float32, shape=(None, output_dim), name="GO")

            # Give a "GO" token to the decoder. 
            # You might want to revise what is the appended value "+ enc_inp[:-1]". 
            #     dec_inp = [ tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO") ] + [tf.zeros_like(v) for v in enc_inp[:-1]]
            #     dec_inp = [ tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")] + enc_inp[1:]
            self.dec_inp = [self.go_sym] + self.expected_sparse_output[:-1] # feed previous target as next input
            self.keep_prob = tf.placeholder(tf.float32)
            # Create a `layers_stacked_count` of stacked RNNs (GRU cells here). 
            self.cells = []
            for i in range(n_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    if cell_type == 'GRU':
                        self.cells.append(DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_dim), output_keep_prob=self.keep_prob))
                    elif cell_type == 'LSTM':
                        self.cells.append(DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_dim), output_keep_prob=self.keep_prob))
            self.cell = tf.nn.rnn_cell.MultiRNNCell(self.cells)

            enc_outputs, enc_state = tf.nn.dynamic_rnn(self.cell, self.inputs, dtype=tf.float32) # returns outputs, final_state

            # For reshaping the input and output dimensions of the seq2seq RNN: 
            self.w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
            self.b_out = tf.Variable(tf.random_normal([output_dim]))

            self.output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
            def looper(output, i):
                return self.output_scale_factor * (tf.matmul(output, self.w_out) + self.b_out)

            dec_outputs, dec_memory = tf.cond(self.is_train, lambda: rnn_decoder(self.dec_inp, enc_state, self.cell),
                                              lambda: rnn_decoder(self.dec_inp, enc_state, self.cell, loop_function=looper))
            # but without the "norm" part of batch normalization hehe. 
            self.reshaped_outputs = [self.output_scale_factor*(tf.matmul(i, self.w_out) + self.b_out) for i in dec_outputs]

            output_loss = 0
            for _y, _Y in zip(self.reshaped_outputs, self.expected_sparse_output):
            #         output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))
                output_loss += tf.reduce_mean(tf.abs(_y - _Y)) # average loss across batch for single timestep
            self.loss = output_loss / n_pred

            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            tf.summary.scalar('loss', self.loss)

            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

    def feed_vals(self, inps, targs, feed_dict):
        feed_dict[self.inputs] = np.pad(inps, ((0,0),(1,0)), mode='constant')[:,:-1].reshape(inps.shape[0], inps.shape[1], 1)
        feed_dict[self.go_sym] = inps[:, self.n_cond-1].reshape(-1,1)
        for i in range(self.n_pred):
            feed_dict[self.expected_sparse_output[i].name] = targs[:,i].reshape(-1,1)
        return feed_dict

    def predict(self, inputs, targets):
        feed_dict = {self.is_train: False, self.keep_prob: 1.0}
        preds = self.sess.run(self.reshaped_outputs, feed_dict=self.feed_vals(inputs, targets, feed_dict))
        # a list with preds for each time step
        pred_array = np.ndarray((inputs.shape[0], self.n_pred))
        for i,p in enumerate(preds):
            pred_array[:,i] = p.reshape(-1)
        return pred_array


class LegacySeq2Seq(BaseSeq2Seq):

    def __init__(self, n_cond, n_pred, hidden_dim,
                 n_layers=2,
                 input_dim=1,
                 learning_rate=0.01,
                 output_dim=1,
                 cell_type='GRU'):
        """
        Construct graph
        """
        super().__init__(n_cond, n_pred)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.is_train = tf.placeholder(tf.bool)
            # Encoder: inputs
            self.enc_inp = [
                tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
                   for t in range(n_cond-1)
            ]

            self.enc_inp.insert(0, tf.zeros_like(self.enc_inp[0], dtype=np.float32))

            # Decoder: expected outputs
            self.expected_sparse_output = [
                tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
                  for t in range(n_pred)
            ]

            self.go_sym = tf.placeholder(tf.float32, shape=(None, output_dim), name="GO")

            # Give a "GO" token to the decoder. 
            # You might want to revise what is the appended value "+ enc_inp[:-1]". 
            #     dec_inp = [ tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO") ] + [tf.zeros_like(v) for v in enc_inp[:-1]]
            #     dec_inp = [ tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")] + enc_inp[1:]
            self.dec_inp = [self.go_sym] + self.expected_sparse_output[:-1] # feed previous target as next input
            self.keep_prob = tf.placeholder(tf.float32)
            # Create a `layers_stacked_count` of stacked RNNs (GRU cells here). 
            self.cells = []
            for i in range(n_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    if cell_type == 'GRU':
                        self.cells.append(DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_dim), output_keep_prob=self.keep_prob))
                    elif cell_type == 'LSTM':
                        self.cells.append(DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_dim), output_keep_prob=self.keep_prob))
            self.cell = tf.nn.rnn_cell.MultiRNNCell(self.cells)

            # For reshaping the input and output dimensions of the seq2seq RNN: 
            self.w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
            self.b_out = tf.Variable(tf.random_normal([output_dim]))

            self.output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
            def looper(output, i):
                return self.output_scale_factor * (tf.matmul(output, self.w_out) + self.b_out)

            dec_outputs, dec_memory = tf.cond(self.is_train, lambda: tied_rnn_seq2seq(self.enc_inp, self.dec_inp, self.cell),
                                              lambda: tied_rnn_seq2seq(self.enc_inp, self.dec_inp, self.cell, loop_function=looper))
            # but without the "norm" part of batch normalization hehe. 
            self.reshaped_outputs = [self.output_scale_factor*(tf.matmul(i, self.w_out) + self.b_out) for i in dec_outputs]

            output_loss = 0
            for _y, _Y in zip(self.reshaped_outputs, self.expected_sparse_output):
            #         output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))
                output_loss += tf.reduce_mean(tf.abs(_y - _Y)) # average loss across batch for single timestep
            self.loss = output_loss / n_pred

            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            tf.summary.scalar('loss', self.loss)

            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()


    
    def feed_vals(self, inps, targs, feed_dict):
        for i in range(self.n_cond-1):
            feed_dict[self.enc_inp[i+1].name] = inps[:,i].reshape(-1,1)
        feed_dict[self.go_sym] = inps[:, self.n_cond-1].reshape(-1,1)
        for i in range(self.n_pred):
            feed_dict[self.expected_sparse_output[i].name] = targs[:,i].reshape(-1,1)
        return feed_dict


    def run_batch(self):
        pass

    def predict(self, inputs, targets):
        feed_dict = {self.is_train: False, self.keep_prob: 1.0}
        preds = self.sess.run(self.reshaped_outputs, feed_dict=self.feed_vals(inputs, targets, feed_dict))
        # a list with preds for each time step
        pred_array = np.ndarray((inputs.shape[0], self.n_pred))
        for i,p in enumerate(preds):
            pred_array[:,i] = p.reshape(-1)
        return pred_array

class DynamicSeq2Seq(BaseSeq2Seq):

    def __init__(self, n_cond, n_pred, hidden_dim,
                 n_layers=2,
                 input_dim=1,
                 learning_rate=0.01,
                 output_dim=1,
                 cell_type='GRU',
                 batch_size=100):
        """
        Construct graph
        TrainingHelper just iterates over the dec_inputs passed to it
        But in general a helper will take sample ids passed by basic decoder and used these to pick inputs
        BasicDecoder just implements a step function which produces outputs and sample ids at each step
            the outputs are the result of applying the rnn cell followed by an optional output layer

        what I need is a version of GreedyEmbeddingHelper,
            (A helper for use during inference.
             Uses the argmax of the output (treated as logits) and passes the
             result through an embedding layer to get the next input.)


        """
        super().__init__(n_cond, n_pred)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.is_train = tf.placeholder(tf.bool)
            cells = []
            self.keep_prob = tf.placeholder(tf.float32)
            for i in range(n_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    if cell_type=='GRU':
                        cells.append(DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_dim), output_keep_prob=self.keep_prob))
                    elif cell_type=='LSTM':
                        cells.append(DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_dim), output_keep_prob=self.keep_prob))
                    # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            
            self.inputs = tf.placeholder(tf.float32, shape=(None, n_cond, input_dim))
            self.go_sym = tf.placeholder(tf.float32, shape=(None, 1, input_dim))
            self.targets = tf.placeholder(tf.float32, shape=(None, n_pred, input_dim))
            
            
            dec_input = tf.concat([self.go_sym, self.targets[:,:-1,:]], 1)
            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32) # returns outputs, state
            
            # one of the features of the dynamic seq2seq is that it can handle variable length sequences
            # but to do this you need to pad them to equal length then specify the lengths separately
            # with constant lengths we still need to specify the lengths for traininghelper, but n.b. they're all the same
            sequence_lengths = tf.constant(n_pred, shape=(batch_size,))

            train_helper = tf.contrib.seq2seq.TrainingHelper(dec_input, sequence_lengths)

            def sampler(time, outputs, state):
                # this isn't necessary, but just do it to get the types right
                sample_ids = math_ops.cast(
                  math_ops.argmax(outputs, axis=-1), tf.int32)
                return sample_ids

            def looper(time, outputs, state, sample_ids):
                 # next_inputs_fn: callable that takes `(time, outputs, state, sample_ids)`
                 # and emits `(finished, next_inputs, next_state)`.
                next_time = time + 1
                finished = next_time >= sequence_lengths
                next_inputs = tf.reshape(outputs, (batch_size, input_dim)) # collapse the time axis
                # I think this is the right thing to do based on looking at the shape of the outputs of TrainingHelper.initialize
                return (finished, outputs, state)

            inf_helper = tf.contrib.seq2seq.CustomHelper(lambda: (array_ops.tile([False], [batch_size]), tf.reshape(self.go_sym, (batch_size, input_dim))) ,
                                                         sampler,
                                                         looper) # initialize fn, sample fn, next_inputs fn

            # initialize_fn: callable that returns `(finished, next_inputs)`
            # for the first iteration.
            # sample_fn: callable that takes `(time, outputs, state)`
            # next_inputs_fn - see note on looper
            #https://github.com/tensorflow/tensorflow/issues/11540

            output_layer = Dense(1, activation=None)

            train_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cell,
                helper=train_helper,
                initial_state=enc_state,
                output_layer=output_layer)

            inf_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cell,
                helper=inf_helper,
                initial_state=enc_state,
                output_layer=output_layer)
            
            outputs, states, sequence_lengths = tf.cond(self.is_train,
                lambda: tf.contrib.seq2seq.dynamic_decode(decoder=train_decoder),
                lambda: tf.contrib.seq2seq.dynamic_decode(decoder=inf_decoder))
            # outputs, states, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=train_decoder)
            # here outputs is an instance of class BasicDecoderOutput, with attrs rnn_output, sample_ids
        
            self.preds = outputs.rnn_output    
            self.loss = tf.reduce_mean(tf.abs(self.preds - self.targets))
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            tf.summary.scalar('loss', self.loss)

            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

    def feed_vals(self, inps, targs, feed_dict):
        feed_dict[self.inputs] = np.pad(inps, ((0,0),(1,0)), mode='constant')[:,:-1].reshape(inps.shape[0], inps.shape[1], 1)
        feed_dict[self.go_sym] = inps[:,-1].reshape(-1,1,1)
        feed_dict[self.targets] = targs.reshape(targs.shape[0], targs.shape[1], 1)
        return feed_dict

    def predict(self, inputs, targets, one_step=False):
        feed_dict = {self.is_train: one_step, self.keep_prob: 1.0}
        preds = self.sess.run(self.preds, feed_dict=self.feed_vals(inputs, targets, feed_dict))
        return preds