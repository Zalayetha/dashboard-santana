import pandas as pd
import pprint
from nltk.tokenize import word_tokenize
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
nltk.download("punkt")
# Function to classify tokens


def classify_token(token):
    # Number Checker
    if re.match("^\d+\.?\d*$", token):
        return (token, "NUM")
    # Punctuation Checker
    elif re.match(r'^\W+$', token):  # This checks for any non-alphanumeric character
        if token in ['(', '{', '[']:  # OPUNC Checker
            return (token, "OPUNC")
        elif token in [')', '}', ']']:  # EPUNC Checker
            return (token, "EPUNC")
        else:
            return (token, "OPUNC")
    else:
        return (token, "WORD")


# Features Dictionary List
# ============================
PPRE = "PPRE"
PMID = "PMID"
PSUF = "PSUF"
PTIT = "PTIT"
OPRE = "OPRE"
OSUF = "OSUF"
OPOS = "OPOS"
OCON = "OCON"
LPRE = "LPRE"
LSUF = "LSUF"
LLDR = "LLDR"
POLP = "POLP"
LOPP = "LOPP"
DISASTER = "DISASTER"
TIME = "TIME"
DAY = "DAY"
MONTH = "MONTH"
# 1. Contextual Features Dictionary
contextual_features_dictionary = {
    "PPRE": ["Dr.", "Pak", "K.H.", "bin", "van", "SKom", "SH"],
    "PMID": ["bin", "van", "binti"],
    "PSUF": ["SKom", "SH"],
    "PTIT": ["Menristek", "Mendagri"],
    "OPRE": ["PT.", "Universitas"],
    "OSUF": ["Ltd.", "Company"],
    "OPOS": ["Ketua"],
    "OCON": ["Muktamar", "Rakernas"],
    "LPRE": ["Kota", "Provinsi", "Kabupaten", "Kecamatan", "Kelurahan", "Desa", "kota", "provinsi", "kabupaten", "kecamatan", "kelurahan", "desa"],
    "LSUF": ["Utara", "Selatan", "Barat", "Tengah", "Timur", "City", "utara", "selatan", "barat", "tengah", "timur", "Nugini"],
    "LLDR": ["Gubernur", "Walikota"],
    "POLP": ["oleh", "untuk"],
    "LOPP": ["di", "ke", "dari"],
    # IMPACT TODO ini belum dimasukin ke rule
    "DISASTER": ["banjir", "gempa", "bumi", "tsunami", "gunung", "meletus", "tanah", "longsor", "kekeringan", "angin", "topan", "Banjir", "Gempa", "Bumi", "Tsunami", "Gunung", "Meletus", "Tanah", "Longsor", "Kekeringan", "Angin", "Topan"],
    "IMPACT": ["korban jiwa", "rusak berat", "terdampak", "mengungsi", "kerugian", "hilang", "terluka", "mati", "hancur", "terendam"],
    "TIME": ["hari ini", "kemarin", "minggu ini", "bulan ini", "tahun ini", "pagi", "siang", "sore", "malam"],
    "DAY": ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu", "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"],
    "MONTH": ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember", "januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus", "september", "oktober", "november", "desember"]
}

# 2. Morphological Features Dictionary
TITLE_CASE = "TitleCase"
UPPER_CASE = 'UpperCase'
LOWER_CASE = "LowerCase"
MIXED_CASE = "MixedCase"
CAP_START = 'CapStart'
CHAR_DIGIT = 'CharDigit'
DIGIT = 'Digit'
NUMERIC = "Numeric"
NUMSTR = "NumStr"
ROMAN = "Roman"
TIME_FORM = "TimeForm"
morphological_features_dictionary = {
    'TitleCase': r'^[A-Z][a-z]+$',
    'UpperCase': r'^[A-Z]+$',
    'LowerCase': r'^[a-z]+$',
    'MixedCase': r'^(?=.*[a-z])(?=.*[A-Z])[A-Za-z]+$',
    'CapStart': r'^[A-Z]',
    'CharDigit': r'^(?=.*[a-zA-Z])(?=.*[0-9])[A-Za-z0-9]+$',
    'Digit': r'^\d+$',
    'DigitSlash': r'^\d+/\d+$',
    'Numeric': r'^\d+[\.,]\d+$',
    # This would need a predefined list or more complex natural language processing.
    'NumStr': None,
    'Roman': r'^M{0,4}(CM|CD|XC|XL|IX|IV|I{1,3}|V?I{0,3})$',
    'TimeForm': r'^\d{1,2}[:.]\d{2}$'
}

# 3. Part Of Speech Features Dictionary
ART = "ART"
ADJ = 'ADJ'
ADV = 'ADV'
AUX = 'AUX'
C = 'C'
DEF = 'DEF'
NOUN = 'NOUN'
NOUNP = 'NOUNP'
NUM = 'NUM'
MODAL = 'MODAL'
PAR = 'PAR'
PREP = 'PREP'
PRO = 'PRO'
VACT = 'VACT'
VPAS = 'VPAS'
VERB = 'VERB'
OOV = 'OOV'
partOfSpeech_features_dictionary = {
    "si": "ART", "sang": "ART", "seorang": "ART",
    "indah": "ADJ", "baik": "ADJ", "luar biasa": "ADJ", "buruk": "ADJ", "tinggi": "ADJ",
    "telah": "ADV", "kemarin": "ADV", "segera": "ADV", "baru": "ADV", "nanti": "ADV",
    "harus": "AUX", "bisa": "AUX", "boleh": "AUX",
    "dan": "C", "atau": "C", "lalu": "C", "tetapi": "C", "namun": "C",
    "merupakan": "DEF", "adalah": "DEF", "ialah": "DEF",
    "rumah": "NOUN", "gedung": "NOUN", "kantor": "NOUN", "sekolah": "NOUN", "jalan": "NOUN",
    "ayah": "NOUNP", "ibu": "NOUNP", "guru": "NOUNP", "dokter": "NOUNP", "polisi": "NOUNP",
    "satu": "NUM", "dua": "NUM", "tiga": "NUM", "empat": "NUM", "lima": "NUM",
    "akan": "MODAL", "mungkin": "MODAL", "pasti": "MODAL", "dapat": "MODAL",
    "kah": "PAR", "pun": "PAR", "lah": "PAR", "dong": "PAR",
    "di": "PREP", "ke": "PREP", "dari": "PREP", "atas": "PREP", "dengan": "PREP",
    "saya": "PRO", "beliau": "PRO", "kami": "PRO", "kita": "PRO", "mereka": "PRO",
    "menuduh": "VACT", "membaca": "VACT", "menulis": "VACT",
    "dituduh": "VPAS", "dibaca": "VPAS", "ditulis": "VPAS",
    "pergi": "VERB", "tidur": "VERB", "makan": "VERB", "minum": "VERB", "datang": "VERB"
}


# =======================================
# 1. Contextual Features Assignment
def classify_contextual_features(tokens, features):
    classified_contextual_features = []
    for token in tokens:
        found = False
        for feature, values in features.items():
            if token[0] in values:
                classified_contextual_features.append(
                    (token[0], token[1], feature))
                found = True
        if not found:
            classified_contextual_features.append((token[0], token[1], ""))
    return classified_contextual_features

# 2. Morphological Featrues Assignment


def classify_morphological_features(tokens, features):
    classified_morphological_features = []
    # NumStr Dictionary
    num_str_words = ['satu', 'dua', 'tiga', 'empat', 'lima',
                     'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh']

    # Check each pattern
    for token in tokens:
        found = False
        for feature, pattern in features.items():
            if found:
                continue
            if pattern and re.fullmatch(pattern, token[0]):
                classified_morphological_features.append(
                    (token[0], token[1], token[2], feature))
                found = True

        # Check NumStr by list
        if token[0].lower() in num_str_words:
            classified_morphological_features.append(
                (token[0], token[1], token[2], "NumStr"))
            found = True

        # not found
        if not found:
            classified_morphological_features.append(
                (token[0], token[1], token[2], "Unknown"))

    return classified_morphological_features

# 3. Part Of Speech Feature Assignment


def classify_partOfSpeech_features(tokens, features):
    classified_partOfSpeech_features = []

    for token in tokens:
        classified_partOfSpeech_features.append(
            (token[0], token[1], token[2], token[3], partOfSpeech_features_dictionary.get(token[0], "OOV")))

    return classified_partOfSpeech_features

# Rules Dictionary
# =================


# 1. Rules Lokasi Bencana
  # 1.1( diawali dengan tipe token WORD, contextual features LOPP, morphological LOWERCASE, dan Part-Of-Speech PREPosition lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unknown], morphological TitleCase, dan Part-Of-Speech OOV) (di Jawa)

  # 1.2( diawali dengan tipe token WORD, contextual features LOPP, morphological LOWERCASE, dan Part-Of-Speech PREPosition lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unkown], morphological TitleCase, dan Part-Of-Speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features LSUF, morphological words TitleCase, dan part-of-speech OOV (di Jawa Tengah)

  # 1.3 diawaili dengan tipe token WORD, contextual features LOPP, morphological LOWERCASE, dan part-of-speech PREP lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features LPRE, morphological words TitleCase, dan part-of-speech OOV
  # lalu kata setelahnya diawali dengna tipe token WORD, contextual features [unknown], morphological words TitleCase, dan part-of-speech OOV (di Kabupaten Tapanuli)

  # 1.4 diawaili dengan tipe token WORD, contextual features LOPP, morphological LOWERCASE, dan part-of-speech PREP lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features LPRE, morphological words TitleCase, dan part-of-speech OOV
  # lalu kata setelahnya diawali dengna tipe token WORD, contextual features [unknown], morphological words TitleCase, dan part-of-speech OOV
  # lalu kata setelahnya diawali dengna tipe token WORD, contextual features LSUF -> Location Suffix, morphological words TitleCase, dan part-of-speech OOV (di Kabupaten Tapanuli Utara)

  # 1.5 diawaili dengan tipe token WORD, contextual features LPRE, morphological words TitleCase, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unknown], morphological words TitleCase, dan part-of-speech OOV (Provinsi Jawa)

  # 1.6( diawali dengan tipe token WORD, contextual features LOPP, morphological LOWERCASE, dan Part-Of-Speech PREPosition lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unknown], morphological lowercase, dan Part-Of-Speech OOV) (di jawa)

  # 1.7 diawali dengan tipe token WORD, contextual features [unknown], morphological Titlecase, dan Part-Of-Speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features LSUF, morphological titlecase, dan Part-Of-Speech OOV) (Papua Nugini)


# TODO: masih bingung disini
# 2. Rules Dampak Bencana
  # 2.1 diawali dengan tipe token word, contextual features [unknown], morphological features Titlecase, dan part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unknown], morphological features Lowercase, dan part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unknown], morphological features Digit, dan part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unknown], morphological features Lowercase, dan part-of-speech NOUN lalu
  # kata setelahnya tipe token word, contextual features [unknown], morphological features Lowercase, dan part-of-speech OOV lalu (Berdampak pada 2 rumah rusak)

  # 2.2 diawali dengan tipe token word, contextual features [unknown], morphological features lowercase, dan part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unknown], morphological features lowercase, dan part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unknown], morphological features digit, dan part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unknown], morphological features lowercase, dan part-of-speech NOUN lalu
  # kata setelahnya tipe token word, contextual features [unknown], morphological features lowercase, dan part-of-speech OOV (berdampak pada 2 rumah rusak)

  # 2.3 diawali dengan tipe token word, contextual features [unknown], morphological features lowercase, dan part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unkown], morphological features digit, an part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unkown], morphological features lowercae, dan part-of-speech NOUN lalu
  # kata setelahnya tipe token word, contextual features [unkwon], morphological features lowercase, dan part-of-speech OOV (berakibat 5 rumah rusak)

  # 2.4 diawali dengan tipe token word, contextual features [unknown], morphological features Titlecase, dan part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unkown], morphological features digit, an part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unkown], morphological features lowercae, dan part-of-speech NOUN lalu
  # kata setelahnya tipe token word, contextual features [unkwon], morphological features lowercase, dan part-of-speech OOV (Berakibat 5 rumah rusak)

  # 2.5 diawali dengan tipe token word, contextual features [unknown], morphological features lowercase, dan part-of-speech NOUN lalu
  # kata setelahnya tipe token word, contextual features [unkown], morphological features lowercase, an part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unkown], morphological features lowercae, dan part-of-speech OOV lalu
  # kata setelahnya tipe token word, contextual features [unkwon], morphological features lowercase, dan part-of-speech OOV
  # kata setelahnya tipe token word, contextual features [unkwon], morphological features lowercase, dan part-of-speech OOV (rumah warga alami rusak ringan)


# 3. Rules Jenis Bencana
  # 3.1 diawali dengan tipe token word, contextual features [unknown], morphological words LOWERCASE, dan Part-Of-Speech OOV (out of vocabulary) lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features DISASTER, morphological words TitleCase, dan Part-Of-Speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features DISASTER, morphological words TitleCase, dan Part-Of-Speech OOV. (terjadi Gempa Bumi)

  # 3.2 diawali dengan tipe token WORD, contextual features [unknown], morphological words TitleCase, dan Part-Of-Speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features DISASTER, morphological words TitleCase, dan Part-Of-Speech OOV lalu
  # kata setelahnya diawali dengan tipe token word, contextual features DISASTER, morphological words Titlecase, dan Part-Of-Speech OOV (out of vocabulary). (Terjadi Gempa Bumi)

  # 3.3 diawali dengan tipe token word, contextual features [unkown], morphological words LOWERCASE, dan Part-Of-Speech OOV (out of vocabulary) lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features DISASTER, morphological words TitleCase, dan Part-Of-Speech OOV (terjadi Banjir)

  # 3.4 diawali dengan tipe tokn word, contextual features [unknwon], morphological words LOWERCASE, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token word, contextual features DISASTER, morphological words LOWERCASE, dan part-of-speech OOV (terjadi banjir)

  # 3.5 diawali dengan tipe token word, contextual features [unknown], morphological words LOWERCASE, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token word, contextual features DISASTER, morphological words LOWERCASE, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token word, contextual features DISASTER, morphological words LOWERCASE, dan part-of-soeech OOV (terjadi gempa bumi)

  # 3.6 diawaili dengan tipe token word, contextual features DISASTER, moprho LOWERCASE, dan part-of-soeech OOV lalu
  # tipe token word, contextual features DISASTER, moprho LOWERCASE, dan part-of-soeech OOV lalu (tanah longsor)

  # 3.7 tipe token word, contextual features disaster, morphological lowercase, dan part-of-speech oov

  # 3.8 tipe token word, contextual features disaster, morphological titlecase, dan part-of-speech oov


# 4. Rules Waktu Bencana
  # 4.1 diawali dengan token word, contextual features [unknown], morphological words Titlecase, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unkown],morphological words TimeForm, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unknown], morphological words UPPERCASE, dan part-of-speech OOV (Pukul 18:00 WIB)

  # 4.2 diawali dengan token word, contextual features [unknown], morphological words lowercase, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unkown],morphological words TimeForm, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unknown], morphological words UPPERCASE, dan part-of-speech OOV (pukul 18:00 WIB)

  # 4.3 diawali dengan token WORD, contextual features [unknown], morphological words titlecase, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unkown],morphological words titlecase, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features DAY, morphological words titlecase, dan part-of-speech OOV (Pada Hari Selasa)

  # 4.4 diawali dengan token WORD, contextual features [unknown], morphological words lowercase, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unkown],morphological words lowercase, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features DAY, morphological words lowercase, dan part-of-speech OOV (pada hari selasa)

  # 4.5 diawali dengan tipe token WORD, contextual features [unkown],morphological words lowercase, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features DAY, morphological words lowercase, dan part-of-speech OOV (hari selasa)

  # 4.6 diawali dengan tipe token WORD, contextual features [unkown],morphological words Titlecase, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features DAY, morphological words Titlecase, dan part-of-speech OOV (Hari Selasa)

  # 4.7 diawali dengan token word, contextual features [unknown], morphological words Titlecase, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unkown],morphological words TimeForm, dan part-of-speech OOV lalu
  # (Pukul 18:00)

  # 4.8 diawali dengan token word, contextual features [unknown], morphological words lowercase, dan part-of-speech OOV lalu
  # kata setelahnya diawali dengan tipe token WORD, contextual features [unkown],morphological words TimeForm, dan part-of-speech OOV lalu
  # (pukul 18:00 WIB)

# result_lokasi_bencana = []
def match_lokasi_bencana(tokens):
    result_lokasi_bencana = []
    pengurang = 0
    if len(tokens) <= 3:
        pengurang = 0
    else:
        pengurang = 3
    for i in range(len(tokens)):
        try:
            # Rules 1
            if ((tokens[i][1] == 'WORD' and tokens[i][2] == LOPP and tokens[i][3] == LOWER_CASE and tokens[i][4] == PREP) and
                    (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV)):
                # print(f"Lokasi Bencana: {' '.join([tokens[i+1][0]])}")
                result_lokasi_bencana.append(tokens[i+1][0])

            # Rules 2
            if ((tokens[i][1] == 'WORD' and tokens[i][2] == LOPP and tokens[i][3] == LOWER_CASE and tokens[i][4] == PREP) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
                    (tokens[i+2][1] == "WORD" and tokens[i+2][2] == LSUF and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
                # print(f"Lokasi Bencana: {' '.join([tokens[i+1][0], tokens[i+2][0]])}")
                result_lokasi_bencana.append(tokens[i+1][0])
                result_lokasi_bencana.append(tokens[i+2][0])

            # Rules 3
            if ((tokens[i][1] == 'WORD' and tokens[i][2] == LOPP and tokens[i][3] == LOWER_CASE and tokens[i][4] == PREP) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == LPRE and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
                    (tokens[i+2][1] == "WORD" and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
                # print(f"Lokasi Bencana: {' '.join([tokens[i+1][0], tokens[i+2][0]])}")
                result_lokasi_bencana.append(tokens[i+1][0])
                result_lokasi_bencana.append(tokens[i+2][0])

            # Rules 4
            if ((tokens[i][1] == 'WORD' and tokens[i][2] == LOPP and tokens[i][3] == LOWER_CASE and tokens[i][4] == PREP) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == LPRE and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
                (tokens[i+2][1] == "WORD" and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV) and
                    (tokens[i+3][1] == "WORD" and tokens[i+3][2] == LSUF and tokens[i+3][3] == TITLE_CASE and tokens[i+3][4] == OOV)):
                # print(f"Lokasi Bencana: {' '.join([tokens[i+2][0]], tokens[i+3][0])}")
                result_lokasi_bencana.append(tokens[i+2][0])
                result_lokasi_bencana.append(tokens[i+3][0])

            # Rules 5
            if ((tokens[i][1] == 'WORD' and tokens[i][2] == LPRE and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
                    (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV)):
                # print(f"Lokasi Bencana: {' '.join([tokens[i+1][0]])}")
                result_lokasi_bencana.append(tokens[i+1][0])

            # Rules 6
            if ((tokens[i][1] == 'WORD' and tokens[i][2] == LOPP and tokens[i][3] == LOWER_CASE and tokens[i][4] == PREP) and
                    (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV)):
                # print(f"Lokasi Bencana: {' '.join([tokens[i+1][0]])}")
                result_lokasi_bencana.append(tokens[i+1][0])

            # Rules 7
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
                    (tokens[i+1][1] == 'WORD' and tokens[i][2] == LSUF and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV)):
                # print(f"Lokasi Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
                result_lokasi_bencana.append(tokens[i][0])
                result_lokasi_bencana.append(tokens[i+1][0])
        except IndexError:
            index_error = 1
    return set(result_lokasi_bencana)


def match_dampak_bencana(tokens):
    result_dampak_bencana = []
    pengurang = 0
    if len(tokens) <= 4:
        pengurang = 0
    else:
        pengurang = 4
    for i in range(len(tokens)):
        try:
            # Rules 1
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV) and
                (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == DIGIT and tokens[i+2][4] == OOV) and
                (tokens[i+3][1] == 'WORD' and tokens[i+3][3] == LOWER_CASE and tokens[i+3][4] == NOUN) and
                    (tokens[i+4][1] == 'WORD' and tokens[i+4][3] == LOWER_CASE and tokens[i+4][4] == OOV)):
                # print(f"Dampak Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0], tokens[i+3][0], tokens[i+4][0]])}")
                result_dampak_bencana.append(tokens[i][0])
                result_dampak_bencana.append(tokens[i+1][0])
                result_dampak_bencana.append(tokens[i+2][0])
                result_dampak_bencana.append(tokens[i+3][0])
                result_dampak_bencana.append(tokens[i+4][0])

            # Rules 2
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV) and
                (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == DIGIT and tokens[i+2][4] == OOV) and
                (tokens[i+3][1] == 'WORD' and tokens[i+3][3] == LOWER_CASE and tokens[i+3][4] == NOUN) and
                    (tokens[i+4][1] == 'WORD' and tokens[i+4][3] == LOWER_CASE and tokens[i+4][4] == OOV)):
                # print(f"Dampak Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0], tokens[i+3][0], tokens[i+4][0]])}")
                result_dampak_bencana.append(tokens[i][0])
                result_dampak_bencana.append(tokens[i+1][0])
                result_dampak_bencana.append(tokens[i+2][0])
                result_dampak_bencana.append(tokens[i+3][0])
                result_dampak_bencana.append(tokens[i+4][0])

            # Rules 3
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == DIGIT and tokens[i+1][4] == OOV) and
                (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == NOUN) and
                    (tokens[i+3][1] == 'WORD' and tokens[i+3][3] == LOWER_CASE and tokens[i+3][4] == OOV)):
                # print(f"Dampak Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0], tokens[i+3][0], tokens[i+4][0]])}")
                result_dampak_bencana.append(tokens[i][0])
                result_dampak_bencana.append(tokens[i+1][0])
                result_dampak_bencana.append(tokens[i+2][0])
                result_dampak_bencana.append(tokens[i+3][0])
                result_dampak_bencana.append(tokens[i+4][0])

            # Rules 4
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == DIGIT and tokens[i+1][4] == OOV) and
                (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == NOUN) and
                    (tokens[i+3][1] == 'WORD' and tokens[i+3][3] == LOWER_CASE and tokens[i+3][4] == OOV)):
                # print(f"Dampak Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0], tokens[i+3][0], tokens[i+4][0]])}")
                result_dampak_bencana.append(tokens[i][0])
                result_dampak_bencana.append(tokens[i+1][0])
                result_dampak_bencana.append(tokens[i+2][0])
                result_dampak_bencana.append(tokens[i+3][0])
                result_dampak_bencana.append(tokens[i+4][0])

            # Rules 5
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == NOUN) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV) and
                (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == OOV) and
                (tokens[i+3][1] == 'WORD' and tokens[i+3][3] == LOWER_CASE and tokens[i+3][4] == OOV) and
                    (tokens[i+4][1] == 'WORD' and tokens[i+4][3] == LOWER_CASE and tokens[i+4][4] == OOV)):
                # print(f"Dampak Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0], tokens[i+3][0], tokens[i+4][0]])}")
                result_dampak_bencana.append(tokens[i][0])
                result_dampak_bencana.append(tokens[i+1][0])
                result_dampak_bencana.append(tokens[i+2][0])
                result_dampak_bencana.append(tokens[i+3][0])
                result_dampak_bencana.append(tokens[i+4][0])

        except IndexError:
            index_error = 1
    return set(result_dampak_bencana)


def match_jenis_bencana(tokens):
    result_jenis_bencana = []
    pengurang = 0
    if len(tokens) <= 2:
        pengurang = 0
    else:
        pengurang = 2
    for i in range(len(tokens)):
        try:
            # Rules 1
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
                    (tokens[i+2][1] == 'WORD' and tokens[i+2][2] == DISASTER and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
                # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
                result_jenis_bencana.append(tokens[i][0])
                result_jenis_bencana.append(tokens[i+1][0])
                result_jenis_bencana.append(tokens[i+2][0])

            # Rules 2
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
                    (tokens[i+2][1] == 'WORD' and tokens[i+2][2] == DISASTER and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
                # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
                result_jenis_bencana.append(tokens[i][0])
                result_jenis_bencana.append(tokens[i+1][0])
                result_jenis_bencana.append(tokens[i+2][0])

            # Rules 3
            if ((tokens[i][1] == 'WORD' and tokens[i][2] == DISASTER and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
                    (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV)):
                # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
                result_jenis_bencana.append(tokens[i][0])
                result_jenis_bencana.append(tokens[i+1][0])

            # Rules 4
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
                    (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV)):
                # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
                result_jenis_bencana.append(tokens[i][0])
                result_jenis_bencana.append(tokens[i+1][0])

            # Rules 5
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+2][3] == LOWER_CASE and tokens[i+1][4] == OOV) and
                    (tokens[i+2][1] == 'WORD' and tokens[i+2][2] == DISASTER and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == OOV)):
                # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
                result_jenis_bencana.append(tokens[i][0])
                result_jenis_bencana.append(tokens[i+1][0])
                result_jenis_bencana.append(tokens[i+2][0])

            # Rules 6
            if ((tokens[i][1] == 'WORD' and tokens[i][2] == DISASTER and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
                    (tokens[i+1][1] == 'WORD' and tokens[i+1][2] == DISASTER and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV)):
                # print(f"Jenis Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
                result_jenis_bencana.append(tokens[i][0])
                result_jenis_bencana.append(tokens[i+1][0])

            # Rules 7
            if ((tokens[i][1] == 'WORD' and tokens[i][2] == DISASTER and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV)):
                # print(f"Jenis Bencana: {' '.join([tokens[i][0]])}")
                result_jenis_bencana.append(tokens[i][0])

            # Rules 8
            if ((tokens[i][1] == 'WORD' and tokens[i][2] == DISASTER and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV)):
                # print(f"Jenis Bencana: {' '.join([tokens[i][0]])}")
                result_jenis_bencana.append(tokens[i][0])

        except IndexError:
            index_error = 1
    return set(result_jenis_bencana)


def match_waktu_bencana(tokens):
    result_waktu_bencana = []
    pengurang = 0
    if len(tokens) <= 2:
        pengurang = 0
    else:
        pengurang = 2
    for i in range(len(tokens)):
        try:
            # Rules 1
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TIME_FORM and tokens[i+1][4] == OOV) and
                    (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == UPPER_CASE and tokens[i+2][4] == OOV)):
                # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
                result_waktu_bencana.append(tokens[i][0])
                result_waktu_bencana.append(tokens[i+1][0])
                result_waktu_bencana.append(tokens[i+2][0])

            # Rules 2
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TIME_FORM and tokens[i+1][4] == OOV) and
                    (tokens[i+2][1] == 'WORD' and tokens[i+2][3] == UPPER_CASE and tokens[i+2][4] == OOV)):
                # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
                result_waktu_bencana.append(tokens[i][0])
                result_waktu_bencana.append(tokens[i+1][0])
                result_waktu_bencana.append(tokens[i+2][0])

            # Rules 3
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TITLE_CASE and tokens[i+1][4] == OOV) and
                    (tokens[i+2][1] == 'WORD' and tokens[i+2][2] == DAY and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
                # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
                result_waktu_bencana.append(tokens[i][0])
                result_waktu_bencana.append(tokens[i+1][0])
                result_waktu_bencana.append(tokens[i+2][0])

            # Rules 4
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
                (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == LOWER_CASE and tokens[i+1][4] == OOV) and
                    (tokens[i+2][1] == 'WORD' and tokens[i+2][2] == DAY and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == OOV)):
                # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
                result_waktu_bencana.append(tokens[i][0])
                result_waktu_bencana.append(tokens[i+1][0])
                result_waktu_bencana.append(tokens[i+2][0])

            # Rules 5
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
                    (tokens[i+2][1] == 'WORD' and tokens[i+2][2] == DAY and tokens[i+2][3] == LOWER_CASE and tokens[i+2][4] == OOV)):
                # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
                result_waktu_bencana.append(tokens[i][0])
                result_waktu_bencana.append(tokens[i+1][0])
                result_waktu_bencana.append(tokens[i+2][0])

            # Rules 6
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
                    (tokens[i+2][1] == 'WORD' and tokens[i+2][2] == DAY and tokens[i+2][3] == TITLE_CASE and tokens[i+2][4] == OOV)):
                # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0], tokens[i+2][0]])}")
                result_waktu_bencana.append(tokens[i][0])
                result_waktu_bencana.append(tokens[i+1][0])
                result_waktu_bencana.append(tokens[i+2][0])

            # Rules 7
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == TITLE_CASE and tokens[i][4] == OOV) and
                    (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TIME_FORM and tokens[i+1][4] == OOV)):
                # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
                result_waktu_bencana.append(tokens[i][0])
                result_waktu_bencana.append(tokens[i+1][0])

            # Rules 8
            if ((tokens[i][1] == 'WORD' and tokens[i][3] == LOWER_CASE and tokens[i][4] == OOV) and
                    (tokens[i+1][1] == 'WORD' and tokens[i+1][3] == TIME_FORM and tokens[i+1][4] == OOV)):
                # print(f"Waktu Bencana: {' '.join([tokens[i][0], tokens[i+1][0]])}")
                result_waktu_bencana.append(tokens[i][0])
                result_waktu_bencana.append(tokens[i+1][0])

        except IndexError:
            index_error = 1
    return set(result_waktu_bencana)


def tokenize(df):
    list_of_token = []
    for index, row in df.iterrows():
        # tokenization
        tokens = word_tokenize(row[0])

        classified_tokens = [classify_token(token) for token in tokens]
        list_of_token.append(classified_tokens)
    return list_of_token


def feature_assignment(classified_tokens):
    result = []
    for token in classified_tokens:
        classified_contextual_features = classify_contextual_features(
            token, contextual_features_dictionary)

        classified_morphological_features = classify_morphological_features(
            classified_contextual_features, morphological_features_dictionary)

        classified_partOfSpeech_features = classify_partOfSpeech_features(
            classified_morphological_features, partOfSpeech_features_dictionary)

        result.append(classified_partOfSpeech_features)

    return result


def rules_assignment(classified_partOfSpeech_features):
    result = []
    for feature in classified_partOfSpeech_features:
        lokasi = match_lokasi_bencana(feature)
        dampak = match_dampak_bencana(feature)
        jenis = match_jenis_bencana(feature)
        waktu = match_waktu_bencana(feature)

        for token in feature:
            if token[0] in list(lokasi):
                result.append(
                    (token[0], token[1], token[2], token[3], token[4], 'Lokasi'))
            elif token[0] in list(dampak):
                result.append(
                    (token[0], token[1], token[2], token[3], token[4], 'Dampak'))
            elif token[0] in list(jenis):
                result.append(
                    (token[0], token[1], token[2], token[3], token[4], 'Bencana'))
            elif token[0] in list(waktu):
                result.append(
                    (token[0], token[1], token[2], token[3], token[4], 'Waktu'))
            else:
                result.append(
                    (token[0], token[1], token[2], token[3], token[4], 'O'))

    return result


def model_testing(predictions, actuals):
    TP = {}
    TN = {}
    FP = {}
    FN = {}

    # total
    total_TP = []
    total_TN = []
    total_FP = []
    total_FN = []

    print("Testing ....")
    # Calculate metrics
    accuracy = accuracy_score(actuals, predictions)
    # or 'micro', 'weighted' based on your class distribution
    precision = precision_score(actuals, predictions, average='macro')
    recall = recall_score(actuals, predictions, average='macro')
    f1Score = f1_score(actuals, predictions, average='macro')
    conf_matrix = confusion_matrix(actuals, predictions)

    # print the actuals and predictions
    print("Actuals", actuals)
    print("Predictions", predictions)
    # print confusion matrix
    print("Confusion Matrix:\n")
    print(conf_matrix)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=["Bencana", "Lokasi", "Waktu", "Dampak", "O"], columns=[
        "Bencana", "Lokasi", "Waktu", "Dampak", "O"])

    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix_df, annot=True)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.show()
    num_classes = conf_matrix.shape[0]
    for i in range(num_classes):
        TP[i] = conf_matrix[i, i]
        FN[i] = np.sum(conf_matrix[i, :]) - TP[i]
        FP[i] = np.sum(conf_matrix[:, i]) - TP[i]
        TN[i] = np.sum(conf_matrix) - (TP[i] + FN[i] + FP[i])

    # Print the results
    for i in range(num_classes):
        print(f"Class {i}:")
        print(f"TP= {TP[i]}")
        total_TP.append(TP[i])
        print(f"FN= {FN[i]}")
        total_FN.append(FN[i])
        print(f"FP= {FP[i]}")
        total_FP.append(FP[i])
        print(f"TN= {TN[i]}\n")
        total_TN.append(TN[i])
    print("Total TP=", sum(total_TP))
    print("Total FN=", sum(total_FN))
    print("Total FP=", sum(total_FP))
    print("Total TN=", sum(total_TN))
    # Print the evaluation metrics
    print("Accuracy:", f"{accuracy:.0%}")
    print("Precision:", f"{precision:.0%}")
    print("Recall:", f"{recall:.0%}")
    print("F1-Score:", f"{f1Score:.0%}")

    result = {
        "Accuracy": f"{accuracy:.0%}",
        "Precision": f"{precision:.0%}",
        "Recall": f"{recall:.0%}",
        "F1-Score": f"{f1Score:.0%}"
    }
    return result
