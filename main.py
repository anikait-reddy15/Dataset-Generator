#AI Syntetic Dataset Generator

'''Build a platform that generates synthetic but realistic datasets for machine learning model training while maintaining privacy.
Helpful for industries lacking large datasets due to privacy concerns.'''

#import statements
import json
import os
import streamlit as st

#API key integration
filepath = "kaggleapi.json"
with open(filepath, "r") as f:
    data = json.load(f)
api_key =  data.get("key")

st.set_page_config(page_title="AI Dataset generator", page_icon = ":1234:")
st.header("Dataset Generator")

    
options = ("Health", "Finance", "E-commerce", "Social media", "Environmental science",
               "Education", "Transportation", "Marketing", "Agriculture", "Energy", "Cybersecurity",
               "Governement", "Real estate", "Entertainment", "Stock market") 
    
selected_option = st.selectbox(f"Choose an option : ", options)
text_input =  st.text_input("Explain about the dataset in detail : ")
    
if st.button("Submit"):
    pass

