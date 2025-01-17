import keras
import tensorflow as tf
import numpy as np
import config

class CustomRNNCell(keras.layers.Layer):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.n_mixtures = config.n_mixtures
        self.dense = keras.layers.Dense(3*self.n_mixtures)
        self.lstm_cell = keras.layers.LSTMCell(config.hidden_size)
        self.state_size = (config.corpus_size, self.n_mixtures, config.hidden_size, config.hidden_size)

    def call(self, input_at_t, states_at_t, constants):
        '''
        states_at_t = [w_{t-1}, kappa_{t-1}, h_{t-1}, c_{t-1}]
        constants   = [transcriptions, enumerated_transcriptions]
        '''

        input = tf.concat((input_at_t, states_at_t[0]), axis=-1)

        y, lstm_states_at_t_plus_1 = self.lstm_cell(input, states_at_t[2:])

        b, U, N = constants[0].shape
        U += 1

        y = self.dense(y)
        y = tf.exp(y)
        y = tf.reshape(y, (b, 3, self.n_mixtures))

        kappa_at_t = y[:, 2, :]
        y = y[:, :2, :]

        kappa_at_t += states_at_t[1]
        kappa_at_t = tf.tile(tf.expand_dims(kappa_at_t, axis=-1), (1, 1, U))

        y = tf.tile(tf.expand_dims(y, axis=-1), (1, 1, 1, U))

        w = y[:, 0, ...] * tf.exp(-y[:, 1, ...] * (kappa_at_t - constants[1])**2)
        w = tf.math.reduce_sum(w, axis=-2)

        attention_index = tf.expand_dims(tf.cast(tf.argmax(w, axis=-1), tf.float32), axis=1)
        U -= 1
        w = w[:, :U]

        w = tf.tile(tf.expand_dims(w, axis=-1), (1, 1, N))
        w = tf.math.reduce_sum(w * constants[0], axis=-2)

        return tf.concat([lstm_states_at_t_plus_1[0], w, attention_index], axis=-1), (w, kappa_at_t[..., 0], *lstm_states_at_t_plus_1)
    

class Network(keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_rnn = keras.layers.RNN(CustomRNNCell(), return_sequences=True, zero_output_for_mask=True, return_state=True)
        self.lstm1 = keras.layers.LSTM(config.hidden_size, return_sequences=True, zero_output_for_mask=True, return_state=True)
        self.lstm2 = keras.layers.LSTM(config.hidden_size, return_sequences=True, zero_output_for_mask=True, return_state=True)
        self.dense = keras.layers.Dense(config.output_size)

    def call(self, strokes, transcriptions, mask=None, initial_state=None, **kwargs):

        if initial_state is None:
            initial_state = [None] * 3

        # prepare input for custom RNN
        indexes_dim = transcriptions.shape[1] + 1
        indexes = tf.range(1, indexes_dim + 1, dtype=np.float32)
        indexes = tf.tile(tf.expand_dims(indexes, 0), (strokes.shape[0], 1))
        indexes = tf.tile(indexes, (1, config.n_mixtures))
        indexes = tf.reshape(indexes, (strokes.shape[0], config.n_mixtures, indexes_dim))

        # pass through custom RNN
        output = self.custom_rnn(strokes, 
                                 constants=(transcriptions, indexes), 
                                 mask=mask, 
                                 initial_state=initial_state[0])
        
        internal_states = [output[1:]]
        output = output[0]
        attention_vectors = output[..., config.hidden_size:config.hidden_size + transcriptions.shape[-1]]
        attention_index = output[..., config.hidden_size + transcriptions.shape[-1]:]
        hidden_outputs = [output[..., :config.hidden_size]]
        
        # pass through 1st LSTM
        output = hidden_outputs[-1]
        output = tf.concat([output, attention_vectors, strokes], axis=-1)
        output = self.lstm1(output, mask=mask, initial_state=initial_state[1])
        hidden_outputs.append(output[0])
        internal_states.append(output[1:])

        # pass through 2nd LSTM
        output = hidden_outputs[-1]
        output = tf.concat([output, attention_vectors, strokes], axis=-1)
        output = self.lstm2(output, mask=mask, initial_state=initial_state[2])
        hidden_outputs.append(output[0])
        internal_states.append(output[1:])

        # pass through Dense and output
        output = tf.concat(hidden_outputs, axis=-1)
        output = self.dense(output)

        return output, attention_index, internal_states
    

class Denormalizer(keras.Model):

    def __init__(self, means, stds, **kwargs):
        super().__init__(**kwargs)
        self.means = tf.Variable(means, dtype=tf.float32, trainable=False, name='means')
        self.stds = tf.Variable(stds, dtype=tf.float32, trainable=False, names='stds')

    def call(self, input, **kwargs):
        x = tf.math.cumsum(input[..., 0] * self.stds[0] + self.means[0], axis=-1)
        y = tf.math.cumsum(input[..., 1] * self.stds[1] + self.means[1], axis=-1)

        return tf.stack([x, -y, input[..., 2]], axis=-1)

