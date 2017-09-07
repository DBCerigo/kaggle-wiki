import os
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import tied_rnn_seq2seq
from tensorflow.contrib.rnn import DropoutWrapper

from utils import clock



class BaseSeq2Seq:

    def __init__(self, n_cond, n_pred):
        self.n_cond = n_cond
        self.n_pred = n_pred

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
                    batch_median_err = np.sum(np.mean(median_errs, axis=0))
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
            self.loss = output_loss

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

    