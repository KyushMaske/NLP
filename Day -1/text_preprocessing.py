import tkinter as tk
from tkinter import filedialog
import pandas as pd
import string
import nltk
import re

# Download NLTK resources if not already downloaded
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt') 

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

stopwords = nltk.corpus.stopwords.words('english')

def select_file():
    """
    Opens a file dialog for the user to select a CSV file.

    Returns:
        str: Path of the selected CSV file.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return file_path

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path of the CSV file.

    Returns:
        pd.DataFrame or None: DataFrame containing the data from the CSV file, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

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
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(text):
    """
    Remove stopwords from the text.

    Args:
        text (list): List of tokens.

    Returns:
        list: List of tokens without stopwords.
    """
    output = [i for i in text if i.lower() not in stopwords]
    return output

def stemming(text):
    """
    Perform stemming on the text.

    Args:
        text (list): List of tokens.

    Returns:
        list: List of stemmed tokens.
    """
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

def lemmatizer(text):
    """
    Perform lemmatization on the text.

    Args:
        text (list): List of tokens.

    Returns:
        list: List of lemmatized tokens.
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
        print(f"Column '{column_name}' not found in the DataFrame.")
        return None

def main():
    """
    Main function to process the CSV file.

    This function prompts the user to select a CSV file, selects a column from the CSV file,
    performs text preprocessing based on the user's choice, and saves the processed DataFrame
    to a new CSV file.
    """
    file_path = select_file()
    if file_path:
        print(f"Selected file: {file_path}")
        df = load_csv(file_path)
        if df is not None:
            print("Columns in the CSV file:")
            print(df.columns)
            column_name = input("Enter the name of the column containing text data: ")
            processed_df = process_text(column_name, df)
            if processed_df is not None:
                print(processed_df.head())
                save_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
                if save_file_path:
                    processed_df.to_csv(save_file_path, index=False)
                    print(f"Processed data saved to: {save_file_path}")
                else:
                    print("No file path provided to save the processed data.")
        else:
            print("Failed to load the CSV file.")
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()
