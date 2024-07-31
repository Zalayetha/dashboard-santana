import streamlit as st
import pandas as pd
from utils.algorithm_statistic import preprocessing

# dataframe
df = []
statistic_based_ner = st.Page(
    'pages/StatisticBased.py',
    title="Statistic Based NER",
)

# title
st.title("Statistic Based NER")

input_tab, preprocessing_tab, normalization_tab, post_tagging_tab, bio_labeling_tab, naive_bayes_classifier_tab = st.tabs(["Input Data", "Preprocessing", "Normalization",
                                                                                                                           "POS Tagging", "BIO Labeling", "Naive Bayes Classifier"])

with input_tab:
    st.header("Input Data")
    st.markdown("To create your model, you need to prepare your data and upload your CSV file. The file format should be structured like the table below: ")
    df_format = pd.DataFrame(
        ['Ini text pertama', 'ini text kedua', 'ini text ketiga'])

    st.data_editor(df_format, use_container_width=True, hide_index=True)
    uploaded_file = st.file_uploader("Upload New CSV File", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            st.data_editor(
                df,
                use_container_width=True,
                num_rows='dynamic',
                hide_index=True
            )
        except:
            st.error("Wrong type of file, file should be an CSV file.")

with preprocessing_tab:
    st.header("Preprocessing")
    if len(df) == 0:
        st.warning("Please upload your file first in Input Data Tab.")
    else:
        data = preprocessing(df)
st.warning(
    "Note: Please do not go to another page, because your work will be lost")
