import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def encode_categories(categories):
    """
    Encode a list of categorical data using LabelEncoder and return a DataFrame.

    Parameters:
    categories (list): List of categorical data.

    Returns:
    pd.DataFrame: DataFrame with original categories and their encoded labels.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(categories)
    return pd.DataFrame({"Category": categories, "Encoded_Labels": encoded_labels})


def one_hot_encode_categories(categories):
    """
    One-hot encode a list of categorical data using OneHotEncoder and return a DataFrame.

    Parameters:
    categories (list): List of categorical data.

    Returns:
    pd.DataFrame: DataFrame with one-hot encoded categories.
    """
    encoder = OneHotEncoder(sparse_output=False, dtype=int)
    encoded_data = encoder.fit_transform(pd.DataFrame({"Category": categories}))
    return pd.DataFrame(encoded_data, columns=encoder.categories_[0])


def bag_of_words(documents):
    """
    Convert a list of text documents to a BOW (Bag-of-Words) DataFrame using CountVectorizer.

    Parameters:
    documents (list): List of text documents.

    Returns:
    pd.DataFrame: DataFrame with BOW representation of the documents.
    """
    vectorizer = CountVectorizer()
    bow_vectors = vectorizer.fit_transform(documents)
    return pd.DataFrame(
        bow_vectors.toarray(), columns=vectorizer.get_feature_names_out()
    )


def tfidf_vectorize(documents):
    """
    Convert a list of text documents to a TF-IDF DataFrame using TfidfVectorizer.

    Parameters:
    documents (list): List of text documents.

    Returns:
    pd.DataFrame: DataFrame with TF-IDF representation of the documents.
    """
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(documents)
    return pd.DataFrame(
        tfidf_vectors.toarray(), columns=vectorizer.get_feature_names_out()
    )


categories = ["teacher", "nurse", "police", "doctor"]

# Label Encoding
label_encoded_df = encode_categories(categories)
print("Label Encoded DataFrame:")
print(label_encoded_df.head())

# One-Hot Encoding
one_hot_encoded_df = one_hot_encode_categories(categories)
print("\nOne-Hot Encoded DataFrame:")
print(one_hot_encoded_df.head())


documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Bag of Words Encoding
bow_df = bag_of_words(documents)
print("\nBag-of-Words DataFrame:")
print(bow_df.head())

# TF-IDF Encoding
tfidf_df = tfidf_vectorize(documents)
print("\nTF-IDF DataFrame:")
print(tfidf_df.head())
