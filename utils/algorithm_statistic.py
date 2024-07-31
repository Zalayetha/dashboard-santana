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
nltk.download('punkt')


# Normalized
def normalized(word, normalized_word_dict):
    return [normalized_word_dict[term] if term in normalized_word_dict else term for term in word]


# Stopwrods removal and stemming
def stopwordsRemoval(words):
    try:
        indo_stopwords = stopwords.words('indonesian')
    except:
        nltk.download("stopwords")
        indo_stopwords = stopwords.words('indonesian')

    indo_stopwords.extend([
        "yg", "dg", "rt", "dgn", "ny", "d", 'klo',
        'kalo', 'amp', 'biar', 'bikin', 'bilang',
        'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
        'jd', 'jgn', 'sdh', 'aja', 'n', 't',
        'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
        '&amp', 'yah', 'dk', 'dgn', 'ds'
    ])

    txt_stopword = pd.read_csv(
        "assets/data/stopwords_id_satya.txt", names=["stopwords"], header=None)
    indo_stopwords.extend(txt_stopword["stopwords"][0].split(" "))

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
    return df[0]


# Normalization Phase
def normalization(df):
    normalized_word = pd.read_excel("assets/data/normalisasi-V1.xlsx")
    normalized_word_dict = {}

    for index, row in normalized_word.iterrows():
        if row[0] not in normalized_word_dict:
            normalized_word_dict[row[0]] = row[1]

    df = df.apply(normalized(normalized_word_dict))


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
