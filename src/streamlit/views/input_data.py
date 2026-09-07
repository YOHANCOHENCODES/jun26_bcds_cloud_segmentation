from config import DATA_DIR
from loaders import load_labels

import streamlit as st

st.write('### The cleaned-up labels dataframe')

df = load_labels(DATA_DIR)
st.dataframe(df.head())
