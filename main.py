import os
import json
import pandas as pd
import streamlit as st
import subprocess
import google.generativeai as genai
from dotenv import load_dotenv

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
st.header("Dataset Creator")

# User Input
user_query = st.text_input("What type of dataset do you need?", placeholder="e.g., stock market trends, weather data")


def find_best_dataset(query):
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


def download_kaggle_dataset(dataset_link):
    if "kaggle.com/datasets/" in dataset_link:
        dataset_id = dataset_link.split("kaggle.com/datasets/")[1]
        
        command = ["kaggle", "datasets", "download", "-d", dataset_id, "--unzip"]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)

            # Find the first CSV file in the directory
            files = [f for f in os.listdir() if f.endswith(".csv")]
            if files:
                df = pd.read_csv(files[0])
                return df, files[0]
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

            # Step 2: Download dataset from Kaggle
            st.write("Downloading dataset from Kaggle...")
            df, file_name = download_kaggle_dataset(dataset_link)

            if df is not None:
                st.success(f"Dataset '{file_name}' downloaded successfully!")
                st.dataframe(df.head())  # Show first few rows
                st.download_button("⬇ Download CSV", df.to_csv(index=False), file_name)
            else:
                st.error("Failed to download dataset.")
    else:
        st.warning("⚠ Please enter a dataset type.")
