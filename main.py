import json
import streamlit as st
import subprocess

st.set_page_config(page_title="AI Dataset generator", page_icon=":1234:")
st.header("Dataset Generator")

options = ("Health", "Finance", "E-commerce", "Social media", "Environmental science",
           "Education", "Transportation", "Marketing", "Agriculture", "Energy", "Cybersecurity",
           "Governement", "Real estate", "Entertainment")

selected_option = st.selectbox(f"Choose an option : ", options)
text_input = st.text_input("Explain about the dataset in detail : ")

if st.button("Submit"):
    if text_input:
        try:
            command = ["kaggle", "datasets", "list", "-s", text_input, "--json"]
            process = subprocess.run(command, capture_output=True, text=True, check=True)
            datasets = json.loads(process.stdout)

            st.write(f"Found {len(datasets)} datasets:")
            for dataset in datasets:
                st.write(f"**{dataset['title']}**")
                st.write(f"Owner: {dataset['ownerUserName']}")
                st.write(f"URL: https://www.kaggle.com/datasets/{dataset['ownerUserName']}/{dataset['slug']}")
                st.write("---")
        except subprocess.CalledProcessError as e:
            st.error(f"Error searching datasets: {e}")
        except json.JSONDecodeError:
            st.error("Error decoding Kaggle API response.")
        except FileNotFoundError:
            st.error("Kaggle CLI not installed. Please install it using 'pip install kaggle'.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a search query.")