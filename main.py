import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def plot_network_output():
    x = np.random.normal(size=10)
    y = np.random.normal(size=10)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)
    ax.plot(x, y)
    user_text_input = st.session_state.get('text_input', None)
    if user_text_input is None:
        ax.set_title(default_input_string)
    elif len(user_text_input) > 0:
        ax.set_title(user_text_input)
    return fig, user_text_input

default_input_string = 'hello there'

with st.sidebar:
    st.button('Regenerate')

canvas = st.container()
with canvas:
    fig, user_text_input = plot_network_output()
    x = -1 if user_text_input is None else len(user_text_input)
    st.pyplot(fig)
    st.write(x)

    st.text_input('label', 
                  placeholder='Type your sentence',
                  label_visibility='collapsed',
                  max_chars=32,
                  key='text_input')



