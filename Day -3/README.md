Continue of Day -2 about Word Embeddings


**Hashing Vectorizer**
-  Designed as memory efficient as possible. The vectorizer applies the hashing trick to encode them as numerical indexes. The downside is once vectorized, the features names can no longer be retrieved.Herr no vocabulary is required and can choose an arbitary-long fixed length vector.

If  using a large dataset for machine learning tasks and  have no use for the resulting dictionary of tokens, then HashingVectorizer would be a good candidate.

[Scikit Link](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)

[Link](https://www.chatgptguide.ai/2024/02/26/what-is-hashing-vectorizer/)



**Difference between Hashing Vectorizer, Count Vectorizer and TF-IDF**

![difff](https://github.com/KyushMaske/NLP/assets/92746457/9288b6bf-a3cb-4d59-9117-5462ba43cc43)


Traditional methods like BOW and TF-IDF treats words a unique discrete units,but word embeddings capture semantic relationship between words by placing semantically similar words close together in the vector space.Word embeddings try to preserve syntactical and semantic information.


**Word2Vec**
- Feed Forward Neural network based techinque for learning word embeddings introduced by Google in 2013.It has two primary models

1. CBOW - Continuous Bag of Words
- Predicts a target word on the context of surrounding words in a sentence or text

![CBOW](https://github.com/KyushMaske/NLP/assets/92746457/0043ee84-a571-40df-af44-2067e534497c)


2. Skip- Gram 

-  Predicts surrounding words given a target word.

![skip](https://github.com/KyushMaske/NLP/assets/92746457/34ba3848-969c-44f6-9ddc-12b81d6c37af)


**Difference between CBOW and skip-Gram**

![d cbow  skip](https://github.com/KyushMaske/NLP/assets/92746457/4e187143-c07b-4e9d-a7a9-94651e21c204)


**GLoVE**
- Unsupervised learning algorithm for obtaining vector representations for words developed by researchers at Stanford University in 2014.

- Representation of words as vectors in a continuous vector space where angle and direction of vectors corresponf to semantic connections between appropriate words.

![glove](https://github.com/KyushMaske/NLP/assets/92746457/67abb098-8796-4ffe-829d-688d21414ba3)


[Stanford Link](https://nlp.stanford.edu/projects/glove/)

[Github](https://github.com/stanfordnlp/GloVe)


**fastText**
-  Extension of Word2Vec by representing words as a bag of character n-grams developed by Facebook's AI Research(FAIR) lab.Particulary useful for handling out-of-vocabulary words and capturing morphological variations.

[fastText](https://fasttext.cc/)

[Github](https://github.com/facebookresearch/fastText)


**Difference between fastText and GLoVE**


![glove vs fasttext](https://github.com/KyushMaske/NLP/assets/92746457/41759939-b0f0-4f15-8d37-4cc9de26ec3d)


**BERT**
- Bidirectional Encoder Representations from Transformers
- Generates context-aware embeddings which capture meaning of words in relation to their surrounding context

- Generated dynamic embeddings that can change depending on the context in which word appears.

[Paper](https://arxiv.org/pdf/1810.04805)
