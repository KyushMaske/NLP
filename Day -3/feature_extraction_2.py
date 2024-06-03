import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.downloader import load
from gensim.models import FastText

# from transformers import BertTokenizer, BertModel
# import torch
import gensim.downloader as api


def hashing_vectorizer_example():
    """
    Example of using HashingVectorizer to convert text documents into feature vectors.
    """
    # Example text documents
    documents = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]

    # Initialize HashingVectorizer
    vectorizer = HashingVectorizer(n_features=12)
    hashed_vectors = vectorizer.fit_transform(documents)
    hashed_df = pd.DataFrame(hashed_vectors.toarray())

    print("Hashing Vectorizer DataFrame:")
    print(hashed_df.head())


def word2vec_example():
    """
    Example of training Word2Vec models using CBOW and Skip-Gram approaches and extracting word vectors.
    """
    # Example sentences
    sentences = [
        ["this", "is", "the", "first", "document"],
        ["this", "document", "is", "the", "second", "document"],
        ["and", "this", "is", "the", "third", "one"],
        ["is", "this", "the", "first", "document"],
    ]

    # Train Word2Vec model using CBOW
    cbow_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
    vector = cbow_model.wv["document"]
    print("CBOW Vector for 'document':\n", vector)

    # Uncomment to train and use the Skip-Gram model
    # skipgram_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
    # vector = skipgram_model.wv['document']
    # print("Skip-Gram Vector for 'document':\n", vector)


def glove_example():
    """
    Example of using pre-trained GloVe model to compute similarity between word pairs.
    """
    # Load pre-trained GloVe model
    glove_model = load("glove-wiki-gigaword-50")
    word_pairs = [("learn", "learning"), ("india", "indian"), ("fame", "famous")]

    # Compute similarity for each pair of words
    for pair in word_pairs:
        similarity = glove_model.similarity(pair[0], pair[1])
        print(
            f"Similarity between '{pair[0]}' and '{pair[1]}' using GloVe: {similarity:.3f}"
        )


def fasttext_example():
    """
    Example of training FastText model and extracting word vectors, including for out-of-vocabulary words.
    """
    # Example sentences
    sentences = [
        ["this", "is", "the", "first", "document"],
        ["this", "document", "is", "the", "second", "document"],
        ["and", "this", "is", "the", "third", "one"],
        ["is", "this", "the", "first", "document"],
    ]

    # Train FastText model
    fasttext_model = FastText(sentences, vector_size=100, window=5, min_count=1, sg=1)

    # Get the vector for a word
    vector = fasttext_model.wv["document"]
    print("FastText Vector for 'document':\n", vector)

    # Get the vector for an out-of-vocabulary word
    vector_oov = fasttext_model.wv["unseenword"]
    print("FastText Vector for 'unseenword':\n", vector_oov)


def fasttext_pretrained_example():
    """
    Example of using pre-trained FastText model to compute similarity between word pairs.
    """
    # Load the pre-trained FastText model
    fasttext_model = api.load("fasttext-wiki-news-subwords-300")
    word_pairs = [("learn", "learning"), ("india", "indian"), ("fame", "famous")]

    # Compute similarity for each pair of words
    for pair in word_pairs:
        similarity = fasttext_model.similarity(pair[0], pair[1])
        print(
            f"Similarity between '{pair[0]}' and '{pair[1]}' using FastText: {similarity:.3f}"
        )


def bert_example():
    """
    Example of using pre-trained BERT model to compute similarity between word pairs based on [CLS] token embeddings.
    """
    # Load pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    word_pairs = [("learn", "learning"), ("india", "indian"), ("fame", "famous")]

    # Compute similarity for each pair of words
    for pair in word_pairs:
        tokens = tokenizer(pair, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)

        # Extract embeddings for the [CLS] token
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        similarity = torch.nn.functional.cosine_similarity(
            cls_embedding[0], cls_embedding[1], dim=0
        )

        print(
            f"Similarity between '{pair[0]}' and '{pair[1]}' using BERT: {similarity:.3f}"
        )


# Example usage of the functions
hashing_vectorizer_example()
word2vec_example()
glove_example()
fasttext_example()
# fasttext_pretrained_example()
# bert_example()
