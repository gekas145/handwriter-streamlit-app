import time
import random
import config
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

def set_seed(seed):
    if seed is None:
        return
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def plot_network_prediction(ax, data):
    data_split = np.split(data, np.where(data[:, -1] == 1)[0] + 1)
    for d in data_split:
        ax.plot(d[:, 0], d[:, 1], color='black')

def encode_transcription(corpus, string_transcription):
    return [str(corpus.get(char, -1)) for char in string_transcription]

def process_output_gaussian(output, logits=False, smoothness=0.0):
    # important dimensions
    nbatch = output.shape[0]

    # extract distributions parameters
    p = output[..., 0]
    if not logits:
        p = tf.sigmoid(p)

    params = tf.reshape(output[..., 1:], (nbatch, -1, config.n_distr, 6))
    
    weights = params[..., 0]
    weights *= 1.0 + smoothness
    if not logits:
        weights = tf.nn.softmax(weights, axis=-1)
    
    means = params[..., 1:3]
    std_dev = params[..., 3:5]
    std_dev -= smoothness
    std_dev = tf.exp(std_dev)
    corr = tf.tanh(params[..., 5])

    return p, weights, means, std_dev, corr

@tf.function
def generate_point_gaussian(output, smoothness=0.0):
    batch_size = output.shape[0]

    p, weights, means, std_dev, corr = process_output_gaussian(output, logits=True, smoothness=smoothness)
    
    means = means[:, 0, ...]
    std_dev = std_dev[:, 0, ...]
    corr = corr[:, 0, ...]
    p = tf.concat([tf.zeros((batch_size, 1)), p], axis=-1)

    # sample one distribution for each batch member
    distr = tf.random.categorical(weights[:, 0, :], 1)
    distr = tf.reshape(distr, (batch_size, 1))

    # get parameters of sampled distribution
    means = tf.gather_nd(means, distr, batch_dims=1)
    std_dev = tf.gather_nd(std_dev, distr, batch_dims=1)
    corr = tf.gather(corr, tf.reshape(distr, (batch_size,)), batch_dims=1)
    cov = corr * tf.math.reduce_prod(std_dev, axis=-1)

    # build covariance matrix for each batch member and get its Cholesky decomposition matrix
    std_dev **= 2
    cov_matrix = tf.stack([tf.stack([std_dev[:, 0], cov], axis=0), tf.stack([cov, std_dev[:, 1]], axis=0)], axis=0)
    cov_matrix = tf.transpose(cov_matrix, (2, 0, 1))
    L = tf.linalg.cholesky(cov_matrix)
    
    # get next point offset coordinates
    point = tf.reshape(L @ tf.random.normal((batch_size, 2, 1)), (batch_size, 2)) + means
    point = tf.concat([point, tf.cast(tf.random.categorical(p, 1), dtype=np.float32)], axis=-1)
    return point[:, np.newaxis, :]

def update_finish_idx(attention_idx, transcriptions_length, current_finish_idx, iteration):
    mask = tf.math.logical_and(attention_idx >= transcriptions_length + config.last_index_offset, current_finish_idx == -1.0)
    mask = tf.cast(mask, dtype=tf.float32)
    return current_finish_idx * (1 - mask) + iteration * mask

def check_finished(finish_idx):
    return tf.math.reduce_all(finish_idx != -1.0)

def clean_finish_idx(finish_idx):
    mask = tf.cast(finish_idx == -1, tf.float32)
    finish_idx = finish_idx * (1 - mask) + tf.tile(tf.cast([config.max_steps_inference], tf.float32), (finish_idx.shape[0],)) * mask
    return tf.cast(finish_idx, tf.int32)

def get_network_prediction(string_transcription, model, denormalizer, corpus, progress_bar, smoothness=0.0, n_samples=1, seed=None):
    set_seed(seed)

    transcriptions = [encode_transcription(corpus, string_transcription) for i in range(n_samples)]

    transcriptions = pad_sequences(transcriptions,
                                   value=-1.0, 
                                   maxlen=config.max_transcription_length,
                                   padding='post',
                                   truncating='post')
    transcriptions = tf.one_hot(transcriptions, config.corpus_size, axis=-1)

    strokes = tf.zeros((transcriptions.shape[0], 1, 3))
    states = None
    transcriptions_length = tf.constant([len(string_transcription)] * n_samples, dtype=tf.float32)
    finish_idx = tf.tile([-1.0], (transcriptions_length.shape[0],))

    for i in range(config.max_steps_inference):
        output, attention_idx, states = model(strokes[:, -1, :][:, np.newaxis, :], transcriptions, 
                                              training=False, initial_state=states)
        point = generate_point_gaussian(output, smoothness=smoothness)
        strokes = tf.concat([strokes, point], axis=1)

        attention_idx = attention_idx[:, 0, 0]
        finish_idx = update_finish_idx(attention_idx, transcriptions_length, finish_idx, i)

        progress_bar.progress((i+1)/config.max_steps_inference, text=config.progress_bar_text)

        if check_finished(finish_idx):
            break
    
    progress_bar.progress(1.0, text=config.ready_text)
    time.sleep(1)
    progress_bar.empty()

    finish_idx = clean_finish_idx(finish_idx)
    strokes = denormalizer(strokes[:, 1:, :]).numpy()
    strokes = [strokes[i, 0:finish_idx[i], :] for i in range(n_samples)]

    return strokes

