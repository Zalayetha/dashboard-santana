import streamlit as st
import pandas as pd
from utils.algorithm_statistic import preprocessing, normalization, postTagging, classify_token, naive_bayes_classifier, model_testing

# dataframe
df = []
statistic_based_ner = st.Page(
    'pages/StatisticBased.py',
    title="Statistic Based NER",
)

# title
st.title("Statistic Based NER")

input_tab, preprocessing_tab, normalization_tab, post_tagging_tab, bio_labeling_tab, naive_bayes_classifer_tab, evaluation_tab = st.tabs(["Input Data", "Preprocessing", "Normalization",
                                                                                                                                          "POS Tagging", "BIO Labeling", "Naive Bayes Classifier", "Model Evaluation"])

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
            st.dataframe(
                df,
                use_container_width=True,
            )
        except:
            st.error("Wrong type of file, file should be an CSV file.")

with preprocessing_tab:
    st.header("Preprocessing")
    if len(df) == 0:
        st.warning("Please upload your file first in Input Data Tab.")
    else:
        df_preprocessing = preprocessing(df)
        st.dataframe(
            df_preprocessing[0],
            use_container_width=True,

        )
with normalization_tab:
    st.header("Normalization")
    if len(df) == 0:
        st.warning("Please upload your file first in Input Data Tab.")
    else:
        data_normalization = normalization(df=df_preprocessing[0])
        st.data_editor(
            data_normalization,
            use_container_width=True,
            num_rows='dynamic',
            hide_index=True
        )

with post_tagging_tab:
    st.header("Post Tagging")
    if len(df) == 0:
        st.warning("Please upload your fiel first in Input Data Tab.")
    else:
        data_post_tag = postTagging(df=data_normalization)
        temp_bef1tag = []
        currentTag = data_post_tag.values.tolist()

        # extract currenttag to other array
        for row in range(len(currentTag)):
            data_row = currentTag[row][1]
            temp_bef1tag.append(data_row)

        bef1tag = []
        for index in range(len(temp_bef1tag)):
            if index == 0:
                bef1tag.append(None)
            else:
                bef1tag.append(temp_bef1tag[index-1])

        df_post_tag = {
            "currentword": [data[0] for data in data_post_tag.values.tolist()],
            "currenttag": [data[1] for data in data_post_tag.values.tolist()],
            "bef1tag": [data for data in bef1tag],
            "token": [classify_token(token=data[0])[1] for data in data_post_tag.values.tolist()]
        }
        st.dataframe(
            pd.DataFrame(data=df_post_tag),
            use_container_width=True,
        )
with bio_labeling_tab:
    st.header("BIO Labeling")
    st.warning(
        "** Changes to columns other than the 'class' column will not affect the naive bayes classify phase.")
    if len(df) == 0:
        st.warning("Please upload your fiel first in Input Data Tab.")
    else:
        label_options = {
            "B-Bencana",
            "I-Bencana",
            "B-Lokasi",
            "I-Lokasi",
            "B-Dampak",
            "I-Dampak",
            "B-Waktu",
            "I-Waktu",
            "O"
        }
        df_bio_labeling = {
            "currentword": [data[0] for data in data_post_tag.values.tolist()],
            "currenttag": [data[1] for data in data_post_tag.values.tolist()],
            "bef1tag": [data for data in bef1tag],
            "token": [classify_token(token=data[0])[1] for data in data_post_tag.values.tolist()],
            "class": ["None"] * len([data[0] for data in data_post_tag.values.tolist()])
        }
        edited_data_labeling = st.data_editor(
            pd.DataFrame(data=df_bio_labeling),
            use_container_width=True,
            column_config={
                "class": st.column_config.SelectboxColumn(label="class", options=label_options, required=True)
            }
        )

with naive_bayes_classifer_tab:
    st.header("Naive Bayes Classifier")
    if len(df) == 0:
        st.warning("Please upload your file first in Input Data Tab")
    else:
        if "None" in edited_data_labeling.loc[:, "class"].tolist():
            st.warning(
                "You cannot run the model before complete the BIO Labeling.")
        else:
            df_classifier = naive_bayes_classifier(edited_data_labeling)
            st.dataframe(
                data=df_classifier,
                use_container_width=True
            )


with evaluation_tab:
    st.header("Model Evaluation")
    if len(df) == 0:
        st.warning("Please upload your file first in Input Data Tab.")
    else:
        if "None" in edited_data_labeling.loc[:, "class"].tolist():
            st.warning(
                "You cannot evaluate the model before complete the BIO Labeling.")

        else:
            st.subheader("Actuals Data")
            actual_table = st.dataframe(
                data=edited_data_labeling,
                use_container_width=True
            )

            st.subheader("Predictions Data")
            predictions_table = st.dataframe(
                data=df_classifier,
                use_container_width=True
            )

            result_test = model_testing(
                predictions=df_classifier.loc[:, "class"].tolist(), actual=edited_data_labeling.loc[:, "class"].tolist())
            st.subheader(f"Accuracy: {result_test['Accuracy']}")
            st.subheader(f"Precision: {result_test['Precision']}")
            st.subheader(f"Recall: {result_test['Recall']}")
            st.subheader(f"F1-Score: {result_test['F1-Score']}")


st.warning(
    "Note: Please do not go to another page, because your work will be lost")
