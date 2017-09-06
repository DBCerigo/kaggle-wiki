class LegacySeq2Seq:

	def __init__(self, n_cond, n_pred, hidden_dim,
				 n_layers=2,
				 input_dim=1,
				 output_dim=1, cell_type='GRU'):
		"""
		Construct graph
		"""
		self.graph = tf.Graph()
        with self.graph.as_default():
			self.train = tf.placeholder(tf.bool)
			# Encoder: inputs
			self.enc_inp = [
			    tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
			       for t in range(n_cond-1)
			]

			self.enc_inp.insert(0, tf.zeros_like(enc_inp[0], dtype=np.float32))

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
			self.dec_inp = [go_sym] + expected_sparse_output[:-1] # feed previous target as next input

			# Create a `layers_stacked_count` of stacked RNNs (GRU cells here). 
			self.cells = []
			for i in range(n_layers):
			    with tf.variable_scope('RNN_{}'.format(i)):
			    	if cell_type == 'GRU':
			        	self.cells.append(tf.nn.rnn_cell.GRUCell(hidden_dim))
			        elif cell_type == 'LSTM':
			        	self.cells.append(tf.nn.rnn_cell.BasicLSTMCell(hidden_dim))
			self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)

			# For reshaping the input and output dimensions of the seq2seq RNN: 
			self.w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
			self.b_out = tf.Variable(tf.random_normal([output_dim]))

			self.output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
			def looper(output, i):
			    return self.output_scale_factor * (tf.matmul(output, self.w_out) + self.b_out)

			dec_outputs, dec_memory = tf.cond(train, lambda: tied_rnn_seq2seq(self.enc_inp, self.dec_inp, self.cell),
			                                  lambda: tied_rnn_seq2seq(self.enc_inp, self.dec_inp, self.cell, loop_function=looper))
			# but without the "norm" part of batch normalization hehe. 
			self.reshaped_outputs = [self.output_scale_factor*(tf.matmul(i, self.w_out) + self.b_out) for i in dec_outputs]

            output_loss = 0
		    for _y, _Y in zip(self.reshaped_outputs, self.expected_sparse_output):
			#         output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))
	        output_loss += tf.reduce_mean(tf.abs(_y - _Y)) # average loss across batch for single timestep
		    self.loss = output_loss

		    self.optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum)
		    self.train_op = optimizer.minimize(self.loss)

		    tf.summary.scalar('loss', self.loss)

            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

    def run_batch(self):
    	pass

    def train(self):
    	pass

    def restore_session(self):
    	pass

class WikiTSSeq2Seq()

class SignalChallSeq2Seq(LegacySeq2Seq)
	
	def __init__(self, n_cond, hidden_dim, n_pred=60):
		super().__init__(n_cond, n_pred, hidden_dim,
						 input_dim=1, output_dim=1,
						 cell_type='GRU')

