import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data
def get_network_prediction(input_string):
    return np.random.normal(size=(2, 10))

def plot_network_prediction(network_prediction):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)
    ax.plot(network_prediction[0], network_prediction[1])

    user_text_input = st.session_state.get('text_input', None)

    current_input = None
    if user_text_input is None:
        current_input = DEFAULT_INPUT
    elif len(user_text_input) > 0:
        current_input = user_text_input
    else:
        current_input = st.session_state['last_valid_input']

    ax.set_title(current_input)
    return fig, current_input

DEFAULT_INPUT = 'hello there'

with st.sidebar:

    col1, col2 = st.columns(2, gap='medium')
    with col1:
        if st.button('Regenerate'):
            get_network_prediction.clear()
    with col2:
        st.button('Replay')

    st.markdown('#')
    
    st.slider('Smoothness', 
               min_value=0.0,
               max_value=5.0,
               step=0.1)

canvas = st.container()
with canvas:
    network_prediction = get_network_prediction('')
    fig, last_valid_input = plot_network_prediction(network_prediction)
    st.session_state['last_valid_input'] = last_valid_input
    st.pyplot(fig)

    st.text_input('label', 
                  placeholder='Type your sentence',
                  label_visibility='collapsed',
                  max_chars=32,
                  key='text_input')



