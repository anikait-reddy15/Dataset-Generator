import os
import json
import streamlit as st
import subprocess

# Load API key from kaggleapi.json
filepath = "kaggleapi.json"
with open(filepath, "r") as f:
    data = json.load(f)
api_key = data.get("key")

# Ensure the .kaggle directory exists
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Save the API key to the expected location
kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
with open(kaggle_json_path, "w") as f:
    json.dump(data, f)


st.set_page_config(page_title="AI Dataset Generator", page_icon=":1234:")
st.header("Dataset Generator")

text_input = st.text_input("Dataset required: ")

def kaggle_dataset(text_input):
    try:
        cleaned_input = text_input.strip().lower()
        command = ["kaggle", "datasets", "list", "-s", cleaned_input, "--format", "json"]
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        datasets = json.loads(process.stdout)

        return datasets[0] if datasets else None

    except subprocess.CalledProcessError as e:
        st.error(f"Error fetching dataset: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Failed to parse Kaggle API response.")
        return None

if st.button("Search"):
    if text_input:
        dataset_info = kaggle_dataset(text_input)
        if dataset_info:
            st.json(dataset_info)
        else:
            st.write("No datasets found.")
    else:
        st.write("No input given.")
