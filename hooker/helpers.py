from tensorflow.contrib.seq2seq import CustomHelper

class InferenceHelper(CustomHelper):

	def __init__(self, start_inputs):
		self._start_inputs = start_inputs
		super().__init__(None, None, None)