import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from io import BytesIO
import zipfile
import tempfile
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

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
if "ctgan_model" not in st.session_state:
    st.session_state.ctgan_model = None

# User Input for dataset search
user_query = st.text_input("What type of dataset do you need?", placeholder="e.g., stock market trends, weather data")

# Search Kaggle Datasets
if st.button("Find Dataset"):
    if user_query:
        st.write("🔍 Searching for datasets on Kaggle...")
        try:
            datasets = api.dataset_list(search=user_query)
            if not datasets:
                st.error("No datasets found for this query. Try another keyword!")
            else:
                st.session_state.dataset_refs = [dataset.ref for dataset in datasets[:5]]
        except Exception as e:
            st.error(f"Error fetching datasets from Kaggle: {e}")
    else:
        st.warning("⚠ Please enter a dataset type.")

# Dataset Selection
if st.session_state.dataset_refs:
    st.session_state.selected_dataset = st.selectbox("Select a dataset:", st.session_state.dataset_refs)

# Load Selected Dataset
if st.session_state.selected_dataset:
    if st.button("Load Selected Dataset"):
        st.write(f"Downloading dataset: {st.session_state.selected_dataset} ...")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                dataset_path = tmp_file.name
                dataset_bytes = api.dataset_download_files(st.session_state.selected_dataset, path=dataset_path, unzip=False)

            # Extract ZIP file
            with zipfile.ZipFile(dataset_path, "r") as zip_file:
                csv_files = [f for f in zip_file.namelist() if f.endswith(".csv")]
                
                if csv_files:  # ✅ 
                    first_csv = csv_files[0]
                    extracted_path = zip_file.extract(first_csv, path=tempfile.gettempdir())  # Extract CSV to temp dir
                    st.session_state.dataset_df = pd.read_csv(extracted_path)  # Load CSV into DataFrame
                    st.success(f"✅ Dataset '{first_csv}' loaded successfully!")
                else:
                    st.error("No CSV file found in the dataset.")  # ✅ Fixed indentation error

        except Exception as e:
            st.error(f"❌ Error downloading or loading dataset: {e}")

# Display Dataset
if st.session_state.dataset_df is not None:
    st.write("### First Few Rows of the Dataset")
    st.dataframe(st.session_state.dataset_df.head())

    # CTGAN Training
    epochs = st.number_input("Enter number of epochs for training:", min_value=1, value=10)
    if st.button("Train CTGAN Model"):
        st.write("🏋️ Training CTGAN model...")
        try:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(st.session_state.dataset_df)
            st.session_state.ctgan_model = CTGANSynthesizer(metadata, epochs=epochs)
            st.session_state.ctgan_model.fit(st.session_state.dataset_df)
            st.success("✅ CTGAN model trained successfully!")
        except Exception as e:
            st.error(f"Error training CTGAN model: {e}")

    # Generate Synthetic Data
    if st.session_state.ctgan_model:
        num_rows = st.number_input("Enter number of synthetic rows to generate:", min_value=1, value=100)
        if st.button("Generate Synthetic Data"):
            st.write("🎲 Generating synthetic data...")
            try:
                synthetic_data = st.session_state.ctgan_model.sample(num_rows)
                st.success(f"✅ Generated {num_rows} synthetic rows!")
                st.write("### First Few Rows of Synthetic Data")
                st.dataframe(synthetic_data.head())
            except Exception as e:
                st.error(f"Error generating synthetic data: {e}")
