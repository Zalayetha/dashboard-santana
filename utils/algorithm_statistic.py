import pandas as pd
from openpyxl import load_workbook
from openpyxl.workbook import Workbook
import nltk
from nltk import word_tokenize
import re
import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tag import CRFTagger
import pprint
import pycrfsuite
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from api_service.apiService import *
from utils.generate_url import *
nltk.download('punkt')

# classify token


def classify_token(token):
    # Number Checker
    if re.match("^\d+\.?\d*$", token):
        return (token, "Number")

    else:
        return (token, "Word")
# Normalized


def normalized(word):

    # hit api to get normalization dictionary
    url_normalization = generateUrl("NORMALIZATION")
    normalized_word_dict = fetch_data(url=url_normalization)["responseBody"]
    return [normalized_word_dict[term] if term in normalized_word_dict else term for term in word]


# Stopwrods removal and stemming
def stopwordsRemoval(words):
    try:
        indo_stopwords = stopwords.words('indonesian')
    except:
        nltk.download("stopwords")
        indo_stopwords = stopwords.words('indonesian')

    # call the stopword api
    url_stopwords = generateUrl("STOPWORDS")
    extended_stopwords = fetch_data(url=url_stopwords)
    indo_stopwords.extend(extended_stopwords["responseBody"])

    factory = StemmerFactory()

    stemmer = factory.create_stemmer()
    filtered_words = []
    for word in words:
        if word not in indo_stopwords:
            # with stemming
            outputAfterStemming = stemmer.stem(word)
            filtered_words.append(outputAfterStemming)

            # without stemming
            # filtered_words.append(word)
    return filtered_words
# Function to remove single character


def removeSingleChar(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)


# Function to remove punctuation
def removePunctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


# Function to perform tokenizing
def tokenizing(text):
    return word_tokenize(text)


# Preprocessing
def preprocessing(df):

    # Erase missing value
    check_null = df.isna()
    null_index = check_null.any(axis=1)
    df.dropna(inplace=True)

    # Case Folding
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].str.lower()

    # Remove Single Character
    df[0] = df[0].apply(removeSingleChar)

    # Remove Punctuation
    df[0] = df[0].apply(removePunctuation)

    # Perform tokenizing
    df[0] = df[0].apply(tokenizing)

    # Filtering and stemming
    # stemming
    df[0] = df[0].apply(stopwordsRemoval)
    return df


# Normalization Phase
def normalization(df):
    df = df.apply(normalized)
    return df


# Pos Tagging Phase
def postTagging(df):
    crftagger = CRFTagger()
    crftagger.set_model_file(
        'assets/data/all_indo_man_tag_corpus_model.crf.tagger')
    list = crftagger.tag_sents(df)
    listNew = []
    for i in list:
        for j in i:
            listNew.append(j)
    df = pd.DataFrame(listNew)
    return df


# Naive Bayes Classifier
def naive_bayes_classifier(df):
    df_prediction = []
    for i in df.values.tolist():
        data = {
            'currentword': i[0],
            'currenttag': i[1],
            'bef1tag': i[2],
            'token': i[3],
            'class': "?"
        }
        df_prediction.append(data)

    # Step 1: Menghitung probabilitas kelas
    total_data = len(df['class'])
    kelas_counts = {}
    for kelas in df['class']:
        kelas_counts[kelas] = kelas_counts.get(kelas, 0)+1

    probabilitas_kelas = {kelas: count /
                          total_data for kelas, count in kelas_counts.items()}
    total_data = sum(kelas_counts.values())

    # Step 2 : Menghitung probabilitas setiap fitur dengan laplace smoothing
    probabilitas_fitur = {}
    unique_values_per_feature = {fitur: set()
                                 for fitur in df if fitur != "class"}
    for fitur in df:
        if fitur != 'class':
            for value in df[fitur]:
                unique_values_per_feature[fitur].add(value)

    for kelas in probabilitas_kelas:
        probabilitas_fitur[kelas] = {}
        for fitur in df:
            if fitur != 'class':
                fitur_counts = {}
                for i in range(len(df['class'])):
                    if (df["class"][i] == kelas):
                        fitur_counts[df[fitur][i]] = fitur_counts.get(
                            df[fitur][i], 0) + 1
                        total_fitur_in_kelas = sum(fitur_counts.values())
                        # Vocab Size buat laplace smoothing
                        V = len(unique_values_per_feature[fitur])
                        probabilitas_fitur[kelas][fitur] = {value: (count + 1) / (total_fitur_in_kelas + V)
                                                            for value, count in fitur_counts.items()}

    # Step 3: Menghitung probabilitas gabungan
    predicted_class_list = []
    for data in df_prediction:
        probabilitas_gabungan = {}
        for kelas in probabilitas_kelas:
            probabilitas_gabungan[kelas] = probabilitas_kelas[kelas]
            for fitur in data:
                if fitur != "class":
                    value = data[fitur]
                    if value in probabilitas_fitur[kelas][fitur]:
                        probabilitas_gabungan[kelas] *= probabilitas_fitur[kelas][fitur][value]
                    else:
                        # Laplace smoothing jika fitur dari data test tidak dikenal
                        probabilitas_gabungan[kelas] *= 1 / (
                            kelas_counts[kelas] + len(unique_values_per_feature[fitur]))
        # Step 4: Menentukan kelas dengan probabilitas maksimum
        predicted_class = max(probabilitas_gabungan,
                              key=probabilitas_gabungan.get)
        predicted_class_list.append(predicted_class)

    # Return an new dataframe to display it on table
    result_df = []
    list_df = df.values.tolist()
    try:
        for i in range(len(list_df)):
            data = {
                'currentword': list_df[i][0],
                'currenttag': list_df[i][1],
                'bef1tag': list_df[i][2],
                'token': list_df[i][3],
                'class': predicted_class_list[i]
            }
            result_df.append(data)
        return pd.DataFrame(data=result_df)
    except Exception:
        print("Failed to return dataframe")


# model testing
def model_testing(predictions, actual):
    print(f"predictions {predictions}")
    print(f"actual {actual}")
    accuracy = accuracy_score(actual, predictions)
    # or 'micro', 'weighted' based on your class distribution
    precision = precision_score(actual, predictions, average='macro')
    recall = recall_score(actual, predictions, average='macro')
    f1Score = f1_score(actual, predictions, average='macro')
    conf_matrix = confusion_matrix(actual, predictions)
    # Print the evaluation metrics
    result = {
        "Accuracy": f"{accuracy:.1%}",
        "Precision": f"{precision:.1%}",
        "Recall": f"{recall:.1%}",
        "F1-Score": f"{f1Score:.1%}"
    }
    return result
