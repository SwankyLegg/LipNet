from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import numpy as np

class CTCLayer(Layer):
	def __init__(self, name='ctc', **kwargs):
		super(CTCLayer, self).__init__(name=name, **kwargs)
		
	def call(self, inputs):
		y_pred, labels, input_length, label_length = inputs
		# the 2 is critical here since the first couple outputs of the RNN
		# tend to be garbage:
		y_pred = y_pred[:, 2:, :]
		return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
	
	def compute_output_shape(self, input_shape):
		return (None, 1)  # (batch_size, 1)