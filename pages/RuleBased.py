import streamlit as st
import pandas as pd
from utils.algorithm_rule import tokenize, feature_assignment, rules_assignment, model_testing
import pprint
# Data Frame
df = []


def testClicked(predictions, actuals):
    result_test = model_testing(predictions=predictions, actuals=actuals)
    st.header("Result")
    st.subheader(f"Accuracy: {result_test['Accuracy']}")
    st.subheader(f"Precision: {result_test['Precision']}")
    st.subheader(f"Recall: {result_test['Recall']}")
    st.subheader(f"F1-Score: {result_test['F1-Score']}")


rule_based_ner = st.Page(
    'pages/RuleBased.py',
    title="Rule Based INER",
    icon=":material/home:"
)

# title
st.title("Rule Based INER")


input_tab, tokenization_tab, feature_assignment_tab, rule_assignment_tab, name_tagging_tab, testing_tab = st.tabs(
    ["Input Data", "Tokenization", "Feature Assignment", "Rule Assignment", "Manual Name Tagging", "Model Testing"])

with input_tab:
    st.header("Input Data")
    st.markdown(
        "To create your model, you need to prepare your data and upload your CSV file. The file format should be structured like the table below:")
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


with tokenization_tab:
    st.header("Tokenizing")
    if len(df) == 0:
        st.warning("Please upload your file first in Input Data Tab.")
    else:
        token = tokenize(df)

        st.markdown('Result of the tokenizing phase')
        # create appended list of token to display in dataframe
        appended_list_of_token = []
        for col in token:
            for row in col:
                appended_list_of_token.append(row)

        df_token = {
            'Token': [text[0] for text in appended_list_of_token],
            'Token Kind': [text[1] for text in appended_list_of_token]
        }
        st.dataframe(pd.DataFrame(df_token), use_container_width=True)


with feature_assignment_tab:
    st.header("Feature Assignment")
    if len(df) == 0:
        st.warning("Please upload your file first in Input Data Tab.")
    else:
        st.markdown('Result of feature assignment phase')
        feature = feature_assignment(classified_tokens=token)

        # create appended list of feature to display in dataframe
        appended_list_of_feature = []
        for col in feature:
            for row in col:
                appended_list_of_feature.append(row)

        df_feature = {
            'Token': [text[0] for text in appended_list_of_feature],
            'Token Kind': [text[1] for text in appended_list_of_feature],
            'Contextual Features':  [text[2] for text in appended_list_of_feature],
            'Morphological Features':  [text[3] for text in appended_list_of_feature],
            'Part-of-speech Features':  [text[4] for text in appended_list_of_feature],
        }

        st.dataframe(pd.DataFrame(data=df_feature), use_container_width=True)

with rule_assignment_tab:
    st.header("Rule Assignment")
    if len(df) == 0:
        st.warning("Please upload your file first in Input Data Tab.")
    else:
        rule = rules_assignment(classified_partOfSpeech_features=feature)
        df_rule = {
            'Token': [text[0] for text in rule],
            'Token Kind': [text[1] for text in rule],
            'Contextual Features':  [text[2] for text in rule],
            'Morphological Features':  [text[3] for text in rule],
            'Part-of-speech Features':  [text[4] for text in rule],
            'Type of Named Entity': [text[5] for text in rule],
        }
        st.dataframe(pd.DataFrame(data=df_rule), use_container_width=True)

with name_tagging_tab:
    st.header("Manual Name Tagging")
    st.warning(
        "** Changes to columns other than the 'Type of Named Entity' column will not affect the testing phase.")
    if len(df) == 0:
        st.warning("Please upload your file first in Input Data Tab.")
    else:
        # Define options for the dropdown
        label_options = {'O', 'Bencana',
                         'Lokasi', 'Waktu', 'Dampak'}

        df_name_tagging = {
            'Token': [text[0] for text in rule],
            'Token Kind': [text[1] for text in rule],
            'Contextual Features':  [text[2] for text in rule],
            'Morphological Features':  [text[3] for text in rule],
            'Part-of-speech Features':  [text[4] for text in rule],
            'Type of Named Entity': ['None'] * len(rule),
        }
        edited_data_actual = st.data_editor(
            pd.DataFrame(data=df_name_tagging),
            use_container_width=True,
            column_config={
                'Type of Named Entity': st.column_config.SelectboxColumn(label="Typed of Named Entity", options=label_options, required=True)
            }
        )


with testing_tab:
    st.header("Model Testing")
    if len(df) == 0:
        st.warning("Please upload your file first in Input Data Tab.")
    else:

        st.subheader("Actuals Data")
        actuals = edited_data_actual.loc[:, 'Type of Named Entity'].tolist()
        edited_data_actual

        st.subheader("Predictions Data")
        predictions = [text[5] for text in rule]
        st.dataframe(pd.DataFrame(data=df_rule), use_container_width=True)

        if "None" in actuals:
            st.warning(
                "You cannot evaluate the model before complete the manual name tagging.")
        else:
            st.button("Test", on_click=testClicked(
                predictions=predictions, actuals=actuals))
st.warning(
    "Note: Please do not go to another page, because your work will be lost")
