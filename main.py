import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import tempfile
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
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None 
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "model" not in st.session_state:
    st.session_state.model = None 
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = None  

def download_dataset(dataset_ref, max_rows=5000):
    """Download, extract, and load dataset."""
    
    temp_dir = tempfile.TemporaryDirectory()
    st.session_state.temp_dir = temp_dir  
    dataset_path = temp_dir.name  

    try:
        st.write("🔄 Downloading dataset... Please wait.")
        api.dataset_download_files(dataset_ref, path=dataset_path, unzip=True)  
        
        # Check files in directory
        files = os.listdir(dataset_path)
        st.write(f"📂 Extracted Files: {files}")  

        # Find CSV files
        csv_files = [f for f in files if f.endswith(".csv")]

        if not csv_files:
            st.error("❌ No CSV files found in the extracted dataset!")
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
                st.error("❌ No datasets found. Try another keyword!")
            else:
                st.session_state.dataset_refs = [dataset.ref for dataset in datasets[:5]]
        except Exception as e:
            st.error(f"⚠ Error fetching datasets: {e}")
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
        st.write(df.head())  

        if st.button("Train Model"):
            with st.spinner("Training model and generating synthetic data..."):
                try:
                    # Generate metadata
                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(df)

                    # Use the correct preset for synthetic data
                    model = SingleTablePreset(name="FAST_ML", metadata=metadata) 
                    model.fit(df)

                    # Store model in session state
                    st.session_state.model = model
                    st.session_state.metadata = metadata
                    st.success("✅ Model trained successfully!")

                except Exception as e:
                    st.error(f"⚠ Error training model: {e}")


        # Synthetic data generation
        if st.session_state.model is not None:
            num_values = st.number_input("Enter the number of synthetic records to generate (max 1000):", min_value=1, max_value=1000)

            if st.button("Generate Synthetic Data"):
                with st.spinner("Generating synthetic data..."):
                    try:
                        synthetic_data = st.session_state.model.sample(num_values)
                        st.session_state.synthetic_data = synthetic_data  

                        # Save synthetic data to CSV
                        synthetic_csv_path = "synthetic_dataset.csv"
                        synthetic_data.to_csv(synthetic_csv_path, index=False)

                        st.success("✅ Synthetic dataset generated successfully!")

                        # Display preview of synthetic data
                        st.write("📊 **Synthetic Dataset Preview:**")
                        st.write(synthetic_data.head())

                    except Exception as e:
                        st.error(f"⚠ Error generating synthetic data: {e}")

        # Provide download link for generated CSV file
        if st.session_state.synthetic_data is not None:
            synthetic_csv_path = "synthetic_dataset.csv"
            
            with open(synthetic_csv_path, "rb") as file:
                st.download_button(
                    label="📥 Download Synthetic Dataset",
                    data=file,
                    file_name="synthetic_dataset.csv",
                    mime="text/csv"
                )

# Cleanup temporary directory (if exists)
if "temp_dir" in st.session_state and st.session_state.temp_dir:
    st.session_state.temp_dir.cleanup()
    st.session_state.temp_dir = None  
