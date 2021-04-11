#################################################################################################
# author: alexanderchf1@gmail.com - Alexander Y. Choquenaira Florez
# note: in many places, the code could be shorter, but that would just make it less comprehensible
# comments are not revised, have them with caution
#################################################################################################

import tensorflow as tf
import numpy as np
from MyEmbedding import MyEmbedding
from MyMultiHeadAttention import MyMultiHeadAttention
#from MyDecoderLayer import MyDecoderLayer
#from MyModel import MyModel

def point_wise_FFN(dff, d_model):
	# dff = d_inner
	return tf.keras.Sequential([
		tf.keras.layers.Dense(dff, activation='relu', name='pwf_dense1'),  # (batch_size, seq_len, dff)    
		tf.keras.layers.Dense(d_model, name='pwf_dense2'),  # (batch_size, seq_len, d_model)    
	])
	
class MyDecoderLayer(tf.keras.layers.Layer):
	def __init__(self, n_proj, d_model, d_transf, dff):
		super(MyDecoderLayer, self).__init__()
		#self._name      = name

		self.attention = MyMultiHeadAttention(n_proj, d_model, d_transf)
		self.ffn       = point_wise_FFN(dff, d_model)    

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='dl_layernorm1')
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='dl_layernorm2')

		self.dropout1   = tf.keras.layers.Dropout(0.1, name='dl_dropout1')
		self.dropout2   = tf.keras.layers.Dropout(0.1, name='dl_dropout2')
  
	def call(self, inp_q, inp_k, inp_v, training = True):
		output_attn = self.attention(inp_q, inp_k, inp_v, training)
		#print('output_attn: ', output_attn.shape)    

		output_1 = self.dropout1(output_attn, training = training)
		#output_1 = self.layernorm1(output_1 + inp_q)  # (batch_size, target_seq_len, d_model)
		output_1 = inp_q + self.layernorm1(output_1)  # (batch_size, target_seq_len, d_model)

		output_fnn = self.ffn(output_1)
		output_2   = self.dropout2(output_fnn, training = training)
		#output_3   = self.layernorm2(output_2 + output_1)
		output_3   = output_1 + self.layernorm2(output_2)
		#print('output_3: ', output_3.shape)    
			
		return output_3