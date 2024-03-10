import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))


def preprocess(text):
    result = str(text)
    result = result.replace(r"[^a-zA-Z0-9]", " ")
    result = result.replace(r"\s+", " ")
    result = result.lower()
    result = result.split()
    result = filter(lambda x: x not in STOPWORDS, result)
    result = map(lambda x: lemmatizer.lemmatize(x), result)
    return " ".join(result)
