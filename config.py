import tensorflow as tf

# this file contains all main parameters regarding model and its training process

# model params
dense_dropout = 0.3
lstm_dropout = 0.3
n_distr = 20
n_mixtures = 10
hidden_size = 400
input_size = 3 # do not edit this line
corpus_size = 58 # do not edit this line
output_size = 6 * n_distr + 1 # do not edit this line

# inference
max_transcription_length = 32
max_steps_inference = 1000
last_index_offset = 0
smoothness = 1.5


