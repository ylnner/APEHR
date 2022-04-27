#################################################################################################
# author: alexanderchf1@gmail.com - Alexander Y. Choquenaira Florez
# note: in many places, the code could be shorter, but that would just make it less comprehensible
# comments are not revised, have them with caution
#################################################################################################

import tensorflow as tf
import numpy as np

class MyEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyEmbedding, self).__init__()

        w_init = tf.initializers.RandomNormal()#GlorotNormal()        
        #w_init = tf.initializers.GlorotNormal()
        self.w = tf.Variable(
            initial_value = w_init(shape=(input_dim, output_dim), dtype = 'float32'),
            trainable = True, 
            name = 'me_w'
        )
                    
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value = b_init(shape=(output_dim), dtype = 'float32'), trainable = True, name = 'me_b'
        )

    def call(self, inp):
        return tf.matmul(inp, self.w) + self.b
