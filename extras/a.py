import streamlit as st
import pandas as pd
import string
import nltk
import re

# Download NLTK resources if not already downloaded
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

stopwords = nltk.corpus.stopwords.words('english')

def remove_punctuation(text):
    """
    Remove punctuation marks from the text.

    Args:
        text (str): Input text.

    Returns:
        str: Text with punctuation removed.
    """
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def tokenization(text):
    """
    Split the text into individual words or tokens.

    Args:
        text (str): Input text.

    Returns:
        list: List of tokens.
    """
    tokens = re.split(r'\W+', text)
    return tokens

def remove_stopwords(text):
    """
    Remove stopwords from the text.

    Args:
        text (str): Input text.

    Returns:
        list: List of words without stopwords.
    """
    output = [i for i in text if i.lower() not in stopwords]
    return output

def stemming(text):
    """
    Perform stemming on the text.

    Args:
        text (list): List of words.

    Returns:
        list: List of stemmed words.
    """
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

def lemmatizer(text):
    """
    Perform lemmatization on the text.

    Args:
        text (list): List of words.

    Returns:
        list: List of lemmatized words.
    """
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

def process_text(column_name, df):
    """
    Process text data in a specified column of the DataFrame.

    Args:
        column_name (str): Name of the column containing text data.
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        pd.DataFrame: DataFrame with processed text data.
    """
    if column_name in df.columns:
        df['clean_Message'] = df[column_name].apply(remove_punctuation)
        df['msg_lower'] = df['clean_Message'].apply(lambda x: x.lower())
        df['msg_tokenized'] = df['msg_lower'].apply(lambda x: tokenization(x))
        df['no_stopwords'] = df['msg_tokenized'].apply(lambda x: remove_stopwords(x))
        df['msg_stemmed'] = df['no_stopwords'].apply(lambda x: stemming(x))
        df['msg_lemmatized'] = df['msg_stemmed'].apply(lambda x: lemmatizer(x))
        return df
    else:
        st.error(f"Column '{column_name}' not found in the DataFrame.")
        return None

def main():
    """
    Main function to process the CSV file using Streamlit.
    """
    st.title("CSV Text Preprocessing")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Columns in the CSV file:")
        st.write(df.columns)

        column_name = st.selectbox("Select the column containing text data", df.columns)

        if st.button("Process Text"):
            processed_df = process_text(column_name, df)
            if processed_df is not None:
                st.write("Processed Data:")
                st.write(processed_df.head())

                # Download processed data
                csv = processed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Processed CSV",
                    data=csv,
                    file_name='processed_text.csv',
                    mime='text/csv',
                )

if __name__ == "__main__":
    main()
