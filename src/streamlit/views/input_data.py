from loaders import load_labels

import streamlit as st

st.write('### The cleaned-up labels dataframe')

df = load_labels()
st.dataframe(df.head())
