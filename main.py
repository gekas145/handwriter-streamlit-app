import io
import os
import math
import pickle
import tempfile
import functools
import config as c
import numpy as np
import utils as ut
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from network import Network, Denormalizer
from matplotlib.animation import FuncAnimation, PillowWriter

@st.cache_resource
def load_corpus():
    with open('model/corpus.pickle', 'rb') as f:
        corpus = pickle.load(f)
    return corpus

@st.cache_resource
def load_model():
    model = Network()
    model(tf.zeros((1, 1, c.input_size)), tf.zeros((1, 1, c.corpus_size)))
    model.load_weights('model/model.h5')
    return model

@st.cache_resource
def load_denormalizer():
    denormalizer = Denormalizer(tf.zeros(2), tf.zeros(2))
    denormalizer(tf.zeros((1, 1, c.input_size)))
    denormalizer.load_weights('model/denormalizer.h5')
    return denormalizer

@st.cache_data
def get_network_prediction(string_transcription, _model, _denormalizer, _corpus, _smoothness=0.0, _n_samples=1):
    st.session_state.sample_id = 0
    return ut.get_network_prediction(string_transcription, _model, _denormalizer, _corpus, 
                                     smoothness=_smoothness, 
                                     n_samples=_n_samples)

@st.cache_resource
def get_initial_strokes():
    with open('model/initial_stokes.pickle', 'rb') as f:
        strokes = pickle.load(f)
    return [np.array(p) for p in strokes]

def on_text_input_change():
    st.session_state.first_run_flag = False

def get_current_input():
    user_text_input = st.session_state.get('text_input', None)
    current_input = None
    if user_text_input is None:
        current_input = DEFAULT_INPUT
    elif len(user_text_input) > 0:
        current_input = user_text_input
    else:
        current_input = st.session_state.last_valid_input

    return current_input

def animate_network_prediction(network_prediction):
    def animate(idx, ax, strokes, frame_length):
        ut.plot_network_prediction(ax, strokes[0:idx*frame_length, :])
        ax.set_xlim(np.min(strokes[:, 0]), np.max(strokes[:, 0]))
        ax.set_ylim(np.min(strokes[:, 1]), np.max(strokes[:, 1]))

    fps = 10
    frame_length = 15
    frames = math.ceil(network_prediction.shape[0]/frame_length)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    anim = FuncAnimation(fig,
                         functools.partial(animate, 
                                           ax=ax, 
                                           strokes=network_prediction, 
                                           frame_length=frame_length),
                         frames=frames,
                         interval=1000.0/fps,
                         repeat=False)

    return anim

DEFAULT_INPUT = 'hello there'
N_SAMPLES = 10
INITIAL_STROKES = get_initial_strokes()
current_input = get_current_input()
st.session_state.last_valid_input = current_input

if not 'sample_id' in st.session_state:
    st.session_state.sample_id = 0

if not 'first_run_flag' in st.session_state:
    st.session_state.first_run_flag = True

model = load_model()
denormalizer = load_denormalizer()
corpus = load_corpus()

with st.sidebar:

    st.title('Handwriter')
    st.markdown('#')

    if st.button('Regenerate'):
        st.session_state.sample_id = (st.session_state.sample_id + 1) % N_SAMPLES
        if st.session_state.sample_id == 0:
            get_network_prediction.clear()
            st.session_state.first_run_flag = False
    st.markdown('#')
    
    st.slider('Smoothness', 
               min_value=0.0,
               max_value=5.0,
               step=0.1,
               value=1.5,
               key='slider')
    st.markdown('#')

main_frame = st.container()
with main_frame:
    if st.session_state.first_run_flag:
        network_prediction = INITIAL_STROKES
    else:
        network_prediction = get_network_prediction(current_input, model, denormalizer, corpus, 
                                                    _smoothness=st.session_state.slider,
                                                    _n_samples=N_SAMPLES)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ut.plot_network_prediction(ax, network_prediction[st.session_state.sample_id])
    st.pyplot(fig)

    st.text_input('label',
                  on_change=on_text_input_change,
                  placeholder='Type your sentence',
                  label_visibility='collapsed',
                  max_chars=32,
                  key='text_input')
    
png_plot = io.BytesIO()
fig.savefig(png_plot, format='png')

animation = animate_network_prediction(network_prediction[st.session_state.sample_id])
with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as temp_file:
    animation.save(temp_file.name, writer=PillowWriter(fps=10))
    temp_file.seek(0)
    animation_buf = io.BytesIO(temp_file.read())

os.remove(temp_file.name)

with st.sidebar:
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.download_button('Download PNG',
                           data=png_plot,
                           file_name=st.session_state.last_valid_input+'.png',
                           mime="image/png")
    with col2:
        st.download_button('Download GIF',
                           data=animation_buf,
                           file_name=st.session_state.last_valid_input+'.gif',
                           mime="image/gif")


