# this file is not supposed to be edited

# model params
n_distr = 20
n_mixtures = 10
hidden_size = 400
input_size = 3
corpus_size = 58
output_size = 6 * n_distr + 1

# inference
max_transcription_length = 32
max_steps_inference = 1000
last_index_offset = 0

# app constants
default_input = 'hello there!'
progress_bar_text = 'Magic happening, please stand by'
ready_text = 'All ready, enjoy'
n_samples = 10 # how many samples of handwritten text will be generated at once


