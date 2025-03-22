import os
import pandas as pd
import streamlit as st
import subprocess
import google.generativeai as genai
from dotenv import load_dotenv
from io import StringIO

# Load API keys from .env file
load_dotenv()

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API Keys
if not KAGGLE_USERNAME or not KAGGLE_KEY:
    st.error("Kaggle API credentials are missing in .env file!")
if not GEMINI_API_KEY:
    st.error("Gemini API key is missing in .env file!")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

# Streamlit UI
st.set_page_config(page_title="AI Dataset Finder", page_icon="📊")
st.header("Dataset Finder")

# User Input
user_query = st.text_input("What type of dataset do you need?", placeholder="e.g., stock market trends, weather data")


def find_best_dataset(query):
    """Finds the best Kaggle dataset link using Gemini AI."""
    prompt = f"Find the best Kaggle dataset link for: {query}. Provide only the Kaggle dataset link."

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        dataset_link = response.text.strip()

        if "kaggle.com/datasets/" in dataset_link:
            return dataset_link
        else:
            st.error("Gemini AI did not return a valid Kaggle dataset link.")
            return None

    except Exception as e:
        st.error(f"Error fetching dataset from Gemini AI: {e}")
        return None


def load_kaggle_dataset(dataset_link):
    """Loads dataset from Kaggle into a CSV string variable instead of downloading."""
    if "kaggle.com/datasets/" in dataset_link:
        dataset_id = dataset_link.split("kaggle.com/datasets/")[1]
        
        command = ["kaggle", "datasets", "download", "-d", dataset_id, "--unzip"]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)

            # Find the first CSV file in the directory
            files = [f for f in os.listdir() if f.endswith(".csv")]
            if files:
                df = pd.read_csv(files[0])
                csv_variable = df.to_csv(index=False)  # Store CSV as a variable
                return csv_variable, df
            else:
                st.error("No CSV file found in downloaded dataset.")
                return None, None

        except subprocess.CalledProcessError as e:
            st.error(f"Error downloading dataset: {e}")
            return None, None

    else:
        st.error("Invalid dataset link provided!")
        return None, None


# Search Button
if st.button("Find Dataset"):
    if user_query:
        st.write("🔍 Searching for the best dataset...")

        # Step 1: Get dataset link from Gemini
        dataset_link = find_best_dataset(user_query)
        if dataset_link:
            st.success(f"Found dataset: {dataset_link}")

            # Step 2: Load dataset from Kaggle into a variable
            st.write("Fetching dataset from Kaggle...")
            csv_variable, df = load_kaggle_dataset(dataset_link)

            if csv_variable is not None:
                st.success("Dataset loaded successfully into a variable!")
                st.dataframe(df.head())  # Show first few rows
                
                # Assign dataset to variable
                dataset_csv = csv_variable  # This variable now holds the dataset in CSV format
                
                st.text_area("Dataset in CSV Format:", dataset_csv, height=200)
            else:
                st.error("Failed to load dataset.")
    else:
        st.warning("⚠ Please enter a dataset type.")
