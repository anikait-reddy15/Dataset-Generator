import json
import streamlit as st
import subprocess

#API key integration
filepath = "kaggleapi.json"
with open(filepath, "r") as f:
    data = json.load(f)
api_key = data.get("key")

st.set_page_config(page_title="AI Dataset generator", page_icon=":1234:")
st.header("Dataset Generator")


text_input = st.text_input("Dataset required : ")  

def kaggle_dataset(text_input):

    cleaned_input = text_input.strip().lower()
    command = ["kaggle", "datasets", "list", "-s", cleaned_input, "--json"]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    datasets = json.loads(process.stdout)
        
    if datasets:
        return datasets[0]
    else:
        return None
            
        

if st.button("Search"):
    if text_input:
        dataset_info = kaggle_dataset(text_input)
        
        if dataset_info:
            st.json(dataset_info)
        else:
            st.write("No Datasets found")
    else:
        st.write("No input given")