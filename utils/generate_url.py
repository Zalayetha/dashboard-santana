def generateUrl(url):
    if url == "STOPWORDS":
        return "http://127.0.0.1:5000/stopwords"
    elif url == "NORMALIZATION":
        return "http://127.0.0.1:5000/normalization"
    elif url == "CLASS_STATISTIC":
        return "http://127.0.0.1:5000/class-statistic"
    elif url == "SAVE_BIO_LABELING_STATISTIC":
        return "http://127.0.0.1:5000/statistic/bio-labeling"
    else:
        return ""
