import io
import os
import tempfile
import functools
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


@st.cache_data
def get_network_prediction(input_string):
    return np.random.normal(size=(2, 10))

def get_current_input():
    user_text_input = st.session_state.get('text_input', None)
    current_input = None
    if user_text_input is None:
        current_input = DEFAULT_INPUT
    elif len(user_text_input) > 0:
        current_input = user_text_input
    else:
        current_input = st.session_state['last_valid_input']

    return current_input

def plot_network_prediction(network_prediction, current_input):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)
    ax.plot(network_prediction[0], network_prediction[1])
    ax.set_title(current_input)

    return fig

def animate_network_prediction(network_prediction, current_input):
    
    def animate(idx, ax, strokes):
        ax.plot(strokes[0, 0:idx], strokes[1, 0:idx], color='black')
        ax.set_xlim(np.min(strokes[0, :]), np.max(strokes[0, :]))
        ax.set_ylim(np.min(strokes[1, :]), np.max(strokes[1, :]))

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)
    ax.set_title(current_input)

    anim = FuncAnimation(fig,
                         functools.partial(animate, ax=ax, strokes=network_prediction),
                         frames=10,
                         repeat=False)

    return anim

DEFAULT_INPUT = 'hello there'
current_input = get_current_input()
st.session_state['last_valid_input'] = current_input

with st.sidebar:

    if st.button('Regenerate'):
        get_network_prediction.clear()

    st.markdown('#')
    
    st.slider('Smoothness', 
               min_value=0.0,
               max_value=5.0,
               step=0.1)
    
    st.markdown('#')

canvas = st.container()
with canvas:
    network_prediction = get_network_prediction(current_input)
    fig = plot_network_prediction(network_prediction, current_input)
    st.pyplot(fig)

    st.text_input('label', 
                  placeholder='Type your sentence',
                  label_visibility='collapsed',
                  max_chars=32,
                  key='text_input')
    
png_plot = io.BytesIO()
fig.savefig(png_plot, format='png')

animation = animate_network_prediction(network_prediction, current_input)
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
                           file_name=st.session_state['last_valid_input']+'.png',
                           mime="image/png")
    with col2:
        st.download_button('Download GIF',
                           data=animation_buf,
                           file_name=st.session_state['last_valid_input']+'.gif',
                           mime="image/gif")


