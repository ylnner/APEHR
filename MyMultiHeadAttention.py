#################################################################################################
# author: alexanderchf1@gmail.com - Alexander Y. Choquenaira Florez
# note: in many places, the code could be shorter, but that would just make it less comprehensible
# comments are not revised, have them with caution
#################################################################################################

import tensorflow as tf
import numpy as np

class MyMultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, n_proj, d_model ,d_transf):
		super(MyMultiHeadAttention, self).__init__()
		#self._name = name

		# assigning values
		self.n_proj   = n_proj
		self.d_model  = d_model
		self.d_transf = d_transf

		# defining q,k,v    
		self.wq = tf.keras.layers.Dense(n_proj * d_transf, name='mha_wq', kernel_initializer='glorot_normal')
		self.wk = tf.keras.layers.Dense(n_proj * d_transf, name='mha_wk', kernel_initializer='glorot_normal')
		self.wv = tf.keras.layers.Dense(n_proj * d_transf, name='mha_wv', kernel_initializer='glorot_normal')
				
		# defining out
		self.out_dense = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_normal', name='mha_out_dense')

		# escala 1 / sqrt(64), es usado en el calculo de la atencion
		self.scale     = 1 / (d_transf ** 0.5)
		self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='mha_layernorm')    
		self.dropout   = tf.keras.layers.Dropout(0.1, name='mha_dropout')
	  
  
	def call(self, q,k,v, training = True, mask_attn = None, wq = None):
		# q [28x32x846]
		# k [28x32x846]
		# v [28x32x846]
		#    num de admisiones
		#    num de pacientes
		#    num de codigos
		# mask_attn    [28x28x1]
		#     num de admisiones
		#     num de admisiones
		#           este tensor representa una matriz de admisiones pra considerar solamente admisiones futuras, creo
			
		head_q = self.wq(q)                
		head_k = self.wk(k)
		head_v = self.wv(v)
							
		# make reshape
		head_q = tf.reshape(head_q, [q.shape[0], q.shape[1], self.n_proj, self.d_transf])
		head_k = tf.reshape(head_k, [k.shape[0], k.shape[1], self.n_proj, self.d_transf])
		head_v = tf.reshape(head_v, [v.shape[0], v.shape[1], self.n_proj, self.d_transf])

		# attn_score [24x24x32x8]
		# esto es q * k
		attn_score = tf.einsum('ibnd,jbnd->ijbn', head_q, head_k)                    

		# esto es (q * k) / sqrt(d)
		attn_score = tf.math.multiply(attn_score, self.scale)    

		#attn_score = mask_fill_inf(attn_score, mask_attn[:,:,:,None])
		#print('mask_attn; ', mask_attn.shape)
		#print('3attn_score: ', attn_score.shape)
						
		attn_prob = tf.nn.softmax(attn_score, axis=1)
		#attn_prob = tf.nn.softmax(attn_score)
		###print('attn_prob: ', attn_prob.shape)

		# attn_vec [24x32x8x64]
		attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, head_v)    

		# attn_vec [24x32x512]
		attn_vec = tf.reshape(attn_vec, [attn_vec.shape[0], attn_vec.shape[1], self.n_proj * self.d_transf])    

		# attn_out [24x32x384]
		attn_out = self.out_dense(attn_vec)    

		# output   = [24x32x384]
		#####output = self.layernorm(attn_out + h)    

		#####output = self.dropout(output, training = training)    
				
		#####return output

		return attn_out