#################################################################################################
# author: alexanderchf1@gmail.com - Alexander Y. Choquenaira Florez
# note: in many places, the code could be shorter, but that would just make it less comprehensible
# comments are not revised, have them with caution
#################################################################################################

import tensorflow as tf
import numpy as np
from MyEmbedding import MyEmbedding
from MyMultiHeadAttention import MyMultiHeadAttention
from MyDecoderLayer import MyDecoderLayer
#from MyModel import MyModel

def my_new_get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
	return pos * angle_rates

def my_new_positional_encoding(num_adm, batch, d_model):
	angle_rads = my_new_get_angles(np.arange(num_adm)[:, np.newaxis],
	                      np.arange(d_model)[np.newaxis, :],
	                      d_model)

	# apply sin to even indices in the array; 2i
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

	###pos_encoding = angle_rads[np.newaxis, ...]
	pos_encoding = np.zeros((num_adm, batch, d_model))
  
	for i in range(num_adm):
		pos_encoding[i] = np.repeat(np.reshape(angle_rads[i], (1, len(angle_rads[i]))), batch, axis=0)    
  
    
	return tf.cast(pos_encoding, dtype=tf.float32)

class MyModel(tf.keras.Model):
	def __init__(self, n_codes, d_emb, n_layers, n_proj, d_model, d_transf, dff):
		super(MyModel, self).__init__()    
		self.n_codes  = n_codes
		self.d_emb    = d_emb
		self.n_layers = n_layers
		self.d_transf = d_transf
		self.n_proj   = n_proj
		self.d_model  = d_model        
		self.dff      = dff		
								
		self.embedding = MyEmbedding(input_dim = n_codes, output_dim = d_emb)
						
                     
		self.dec_layers = [MyDecoderLayer(n_proj, d_model, d_transf, dff)
						  for i in range(n_layers)]

		self.final_layer = tf.keras.layers.Dense(n_codes, name = 'mm_final_layer')

		# adicionar dropout
		self.dropout = tf.keras.layers.Dropout(0.1, name = 'mm_dropout')

		self.dropout_1 = tf.keras.layers.Dropout(0.1, name = 'mm_dropout_1')
  
	def call(self, inp, training = True, target_mask = None):
		# inp   
		#     num de admisiones
		#     batch size
		#     num de codigos
		#           este tensor utiliza representa a los pacientes
		# target_mask 
		#     num de admisiones
		#     batch size
		#           este tensor representa una mascara donde se guardan si el paciente tiene una admision o no
		# mask_attn   
		#     num de admisiones
		#     num de admisiones
		#           este tensor representa una matriz de admisiones pra considerar solamente admisiones futuras

		
		hidden = self.embedding(inp)  # [28 x 32 x d_emb]

		positional = my_new_positional_encoding(hidden.shape[0], hidden.shape[1], hidden.shape[2])
		hidden += positional

		hidden = self.dropout(hidden, training = training)

		tf.debugging.check_numerics(hidden, "SOS hidden")


		for i_layer in range(self.n_layers):
			hidden = self.dec_layers[i_layer](hidden, hidden, hidden, training)  
		
		
		out_decoder = hidden

		
		tf.debugging.check_numerics(out_decoder, "SOS out_decoder")

	
		if target_mask is not None: # target_mask  [28x32]
		
			# Returns a tensor with a length 1 axis inserted at index axis
			target_mask = tf.expand_dims(target_mask, axis = 2)   # [28 x 32 x 1]
			
			       
			out_decoder = tf.math.multiply(out_decoder, target_mask)  # [28 x 32 x d_emb]
			                            
	
		out_final_layer = self.final_layer(out_decoder)
		tf.debugging.check_numerics(out_final_layer, "SOS out_final_layer")

		logits   = tf.nn.softmax(out_final_layer)    
		tf.debugging.check_numerics(logits, "SOS logits")

		return logits
