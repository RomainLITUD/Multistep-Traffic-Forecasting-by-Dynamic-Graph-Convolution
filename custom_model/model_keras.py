from custom_model.layers_keras import *
from custom_model.math_utils import *

import numpy as np
import tensorflow as tf
import math
from collections import defaultdict

from keras import Model
from keras.layers import Input, Dense, GRU, TimeDistributed, Lambda, Concatenate, RNN, dot, LeakyReLU
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import keras

def unstack(x):
    x = tf.unstack(x, axis=1)
    return x

def stack(x):
    y = K.stack(x, axis=1)
    return K.squeeze(y, axis=2)

#######################
def create_embed_model(obs_timesteps=10, pred_timesteps=3, nb_nodes=208, k=1, dgc_mode='hybrid', inner_act=None):
    encoder = RNN(DGCRNNCell(k,dgc_mode=dgc_mode), return_state=True)
    decoder = RNN(DGCRNNCell(k,dgc_mode=dgc_mode), return_sequences=True, return_state=True)
    
    unstack_k = Lambda(unstack)
    choice = Scheduled()
    
    input_obs = Input(shape=(obs_timesteps, nb_nodes, 1)) 
    input_gt = Input(shape=(pred_timesteps, nb_nodes, 1)) #(None, T, N, 1)
    encoder_inputs = Lambda(lambda x: K.squeeze(x, axis = -1))(input_obs) # (None, T, N)
    
    encoder_outputs, state_h = encoder(encoder_inputs)
    
    unstacked = unstack_k(input_gt) #[(None, N, 1) x T] list
    
    initial = unstacked[0] #(None, N, 1)
    decoder_inputs = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(initial) #(None, 1, N)
    decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
    state_h = state_h_new
    prediction = []
    decoded_results = decoder_outputs_new
    prediction.append(decoded_results)
    
    if pred_timesteps > 1:       
        for i in range(1,pred_timesteps):
            decoder_inputs = choice([prediction[-1], unstacked[i]])#(None, 208, 1)
            decoder_inputs = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(decoder_inputs)#(None, 1, 208)
            decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
            state_h = state_h_new
            decoded_results = decoder_outputs_new
            prediction.append(decoded_results)
    
    outputs = Lambda(stack)(prediction)
    model = Model([input_obs, input_gt], outputs)

    return model



