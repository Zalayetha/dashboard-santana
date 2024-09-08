import requests
import streamlit as st
# Define a function to fetch data from an API


def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch data: {response.status_code}")
        return None


def insert_preprocessing_data(url, data):
    # convert df to json
    json_data = data.to_json(orient='records')
    response = requests.post(url, json=json_data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to insert data: {response.status_code}")
        return None
