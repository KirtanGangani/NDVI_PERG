import streamlit as st
# full width display
st.set_page_config(layout="wide")

import leafmap.foliumap as leafmap

layers = ["Esri.WorldTopoMap", "OpenTopoMap"]
m1 = leafmap.Map()
m1.add_basemap("Esri.WorldTopoMap")

m2 = leafmap.Map()
m2.add_basemap("OpenTopoMap")

cols = st.columns(2)
with cols[0]:
    m1.to_streamlit()
with cols[1]:
    m2.to_streamlit()
# leafmap.linked_maps(rows=1, cols=2, height="400px", layers=layers)