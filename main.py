import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import tempfile
import threading
from sdv.metadata import SingleTableMetadata 
from sdv.lite import SingleTablePreset

# Load environment variables
load_dotenv()

# Fetch API keys from .env file
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Validate API Keys
if not KAGGLE_USERNAME or not KAGGLE_KEY:
    st.error("Kaggle API credentials are missing in .env file!")
    st.stop()

# Set Kaggle API credentials
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Streamlit UI
st.set_page_config(page_title="AI Dataset Finder", page_icon="📊")
st.header("Dataset Loader for AI Models")

# Initialize session state variables
if "dataset_refs" not in st.session_state:
    st.session_state.dataset_refs = []
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None
if "dataset_df" not in st.session_state:
    st.session_state.dataset_df = None
if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = ""

def download_dataset(dataset_ref):
    """Function to download and load dataset efficiently."""
    cache_path = f"./cached_datasets/{dataset_ref.replace('/', '_')}.csv"
    os.makedirs("./cached_datasets", exist_ok=True)
    
    if os.path.exists(cache_path):
        st.session_state.dataset_df = pd.read_csv(cache_path)
        st.success("Loaded dataset from cache!")
        return
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
            
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            if csv_files:
                first_csv = os.path.join(temp_dir, csv_files[0])
                os.rename(first_csv, cache_path)  # Cache for future use
                st.session_state.dataset_df = pd.read_csv(cache_path)
                st.success("Dataset downloaded and cached!")
            else:
                st.error("No CSV file found in dataset!")
    except Exception as e:
        st.error(f"Error downloading dataset: {e}")

# User Input for dataset search
user_query = st.text_input("What type of dataset do you need?", placeholder="e.g., stock market trends, weather data")

# Search Kaggle Datasets
if st.button("Find Dataset"):
    if user_query:
        st.write("🔍 Searching for datasets on Kaggle...")
        try:
            datasets = api.dataset_list(search=user_query)
            if not datasets:
                st.error("No datasets found. Try another keyword!")
            else:
                st.session_state.dataset_refs = [dataset.ref for dataset in datasets[:5]]
        except Exception as e:
            st.error(f"Error fetching datasets: {e}")
    else:
        st.warning("⚠ Please enter a dataset type.")

# Dataset Selection
if st.session_state.dataset_refs:
    st.session_state.selected_dataset = st.selectbox("Select a dataset:", st.session_state.dataset_refs)

# Load dataset asynchronously
if st.session_state.selected_dataset:
    if st.button("Load Selected Dataset"):
        st.write(f"Downloading dataset: {st.session_state.selected_dataset} ...")
        download_thread = threading.Thread(target=download_dataset, args=(st.session_state.selected_dataset,))
        download_thread.start()
        download_thread.join()

    # Define the directory where datasets are cached
    cache_dir = "cached_datasets"

    # List all files in the cached directory
    files = os.listdir(cache_dir)

    # Find the first CSV file
    csv_files = [f for f in files if f.endswith(".csv")]

    if csv_files:
        dataset_path = os.path.join(cache_dir, csv_files[0])  # Select the first CSV file
        df = pd.read_csv(dataset_path)  # Load dataset into DataFrame
        st.write("Dataset loaded successfully!")
        st.write(df.head())  # Display first few rows
    else:
        st.write("No CSV files found in the cached_datasets directory.")
        
    #Generate metadata from dataset
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    #Train model
    model = SingleTablePreset(name="ctgan", metadata=metadata)
    model.fit(df)

    #Generate Synthetic Data
    num_values = st.number_input("Enter the number of values to be created (max 1000) : ", min_value=1, max_value=1000)
    synthetic_data = model.sample(num_values)
    synthetic_data.to_csv("synthetic_dataset.csv", index=False)

    st.write("Synthetic dataset generated succesfully : synthetic_dataset.csv")