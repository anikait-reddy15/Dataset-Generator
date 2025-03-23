import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import tempfile
import zipfile

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
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None  

def download_dataset(dataset_ref, max_rows=5000):
    """Download, extract, and load dataset."""
    
    temp_dir = tempfile.TemporaryDirectory()
    st.session_state.temp_dir = temp_dir  # Store temp directory
    dataset_path = temp_dir.name  # Correct path reference

    try:
        st.write("🔄 Downloading dataset... Please wait.")
        api.dataset_download_files(dataset_ref, path=dataset_path, unzip=True)  # Unzip during download
        
        # Check files in directory
        files = os.listdir(dataset_path)
        st.write(f"📂 Extracted Files: {files}")  # Debugging output

        # Find CSV files
        csv_files = [f for f in files if f.endswith(".csv")]

        if not csv_files:
            st.error("No CSV files found in the extracted dataset!")
            return
        
        # Select the largest CSV file
        csv_files.sort(key=lambda f: os.path.getsize(os.path.join(dataset_path, f)), reverse=True)
        selected_csv = csv_files[0]
        full_csv_path = os.path.join(dataset_path, selected_csv)

        # Load dataset
        df = pd.read_csv(full_csv_path, nrows=max_rows)
        st.session_state.dataset_df = df
        st.success(f"✅ Dataset '{selected_csv}' loaded successfully!")

    except Exception as e:
        st.error(f"⚠ Error downloading dataset: {e}")

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

# Load dataset synchronously
if st.session_state.selected_dataset:
    if st.button("Load Selected Dataset"):
        with st.spinner("Downloading and loading dataset..."):
            download_dataset(st.session_state.selected_dataset)

    # Display dataset
    if st.session_state.dataset_df is not None:
        df = st.session_state.dataset_df
        st.write("📊 **Dataset Preview:**")
        st.write(df.head())  # Show first few rows

# Cleanup temporary directory (if exists)
if "temp_dir" in st.session_state and st.session_state.temp_dir:
    st.session_state.temp_dir.cleanup()
    st.session_state.temp_dir = None  # Clear temp dir reference
