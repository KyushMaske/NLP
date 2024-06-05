import spacy
from spacy.language import Language
from spacy import displacy
from spacy.tokenizer import Tokenizer
from collections import Counter

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
print(nlp)

# Process a sample text
introduction_doc = nlp("This is your boy Kyush... Hey Guys")
print(type(introduction_doc))

for token in introduction_doc:
    print(token.text)
    
print([token.text for token in introduction_doc])

sentence = (
    "Gus Proto is a Python developer currently"
    " working for a London-based Fintech"
    " company. He is interested in learning"
    " Natural Language Processing."
)

about_doc = nlp(sentence)
sentences = list(about_doc.sents)
print(len(sentences), "length")


def process_text_with_custom_boundaries(text):
    """Process text and print sentences using '...' as a delimiter for sentence detection"""

    @Language.component("set_custom_boundaries")
    def set_custom_boundaries(doc):
        """Add support to use `...` as a delimiter for sentence detection"""
        for token in doc[:-1]:
            if token.text == "...":
                doc[token.i + 1].is_sent_start = True
        return doc

    # Add the custom boundary component before the parser
    nlp.add_pipe("set_custom_boundaries", before="parser")
    
    # Process the text
    custom_ellipsis_doc = nlp(text)
    
    # Extract sentences
    custom_ellipsis_sentences = list(custom_ellipsis_doc.sents)
    
    # Print each sentence
    for sentence in custom_ellipsis_sentences:
        print(sentence)


# Example usage
ellipsis_text = (
    "Gus, can you, ... never mind, I forgot"
    " what I was saying. So, do you think"
    " we should ..."
)
process_text_with_custom_boundaries(ellipsis_text)


def print_tokens_with_indices(text):
    """Process text and print each token with its starting character index."""

    # Process the text
    doc = nlp(text)
    
    # Print each token and its starting character index
    for token in doc:
        print(token, token.idx)


# Example usage
about_text = (
    "Gus Proto is a Python developer currently"
    " working for a London-based Fintech"
    " company. He is interested in learning"
    " Natural Language Processing."
)
print_tokens_with_indices(about_text)


def print_token_properties(text):
    """Process text and print each token with various properties."""

    # Process the text
    doc = nlp(text)
    
    # Print the header
    print(f"{'Text with Whitespace':22}{'Is Alphanumeric?':15}{'Is Punctuation?':18}{'Is Stop Word?'}")
    
    # Print each token's properties
    for token in doc:
        print(
            f"{str(token.text_with_ws):22}"
            f"{str(token.is_alpha):15}"
            f"{str(token.is_punct):18}"
            f"{str(token.is_stop)}"
        )


# Example usage
about_text = (
    "Gus Proto is a Python developer currently"
    " working for a London-based Fintech"
    " company. He is interested in learning"
    " Natural Language Processing."
)
print_token_properties(about_text)


def custom_tokenize_and_print(text, start, end):
    """Tokenize text with a custom tokenizer and print tokens in the specified range."""

    # Compile the prefix, suffix, and infix regular expressions
    prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)
    custom_infixes = [r"@"]  # Custom infix to include "@" character
    infix_re = spacy.util.compile_infix_regex(list(nlp.Defaults.infixes) + custom_infixes)
    
    # Create a custom tokenizer with the updated infix pattern
    nlp.tokenizer = Tokenizer(
        nlp.vocab,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
        token_match=None,
    )
    
    # Process the text with the custom tokenizer
    doc = nlp(text)
    
    # Print tokens from the specified range
    tokens = [token.text for token in doc[start:end]]
    print(tokens)


# Example usage
custom_about_text = (
    "Gus Proto is a Python developer currently"
    " working for a London@based Fintech"
    " company. He is interested in learning"
    " Natural Language Processing."
)
custom_tokenize_and_print(custom_about_text, 8, 15)


def print_spacy_stopwords():
    """Print the first 10 stop words from spaCy's English stop words list."""

    # Get the stop words
    spacy_stopwords = nlp.Defaults.stop_words
    
    # Print the number of stop words
    print(f"Total number of stop words: {len(spacy_stopwords)}")
    
    # Print the first 10 stop words
    for stop_word in list(spacy_stopwords)[:10]:
        print(stop_word)


# Example usage
print_spacy_stopwords()


def print_tokens_and_lemmas(text):
    """Process text with spaCy and print each token along with its lemma if different."""

    # Process the text
    doc = nlp(text)
    
    # Iterate over each token in the document
    for token in doc:
        # Check if the token and its lemma are different
        if str(token) != str(token.lemma_):
            # Print the token and its lemma
            print(f"{str(token):>20} : {str(token.lemma_)}")


# Example usage
conference_help_text = (
    "Gus is helping organize a developer"
    " conference on Applications of Natural Language"
    " Processing. He keeps organizing local Python meetups"
    " and several internal talks at his workplace."
)
print_tokens_and_lemmas(conference_help_text)


def get_most_common_words(text, n=5):
    """Process text with spaCy and return the most common non-stopword, non-punctuation words."""

    # Process the text
    doc = nlp(text)
    
    # Filter out stop words and punctuation, and extract the remaining words
    words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    # Use Counter to count occurrences of each word and retrieve the most common ones
    most_common_words = Counter(words).most_common(n)
    
    return most_common_words


# Example usage
complete_text = (
    "Gus Proto is a Python developer currently"
    " working for a London-based Fintech company. He is"
    " interested in learning Natural Language Processing."
    " There is a developer conference happening on 21 July"
    ' 2019 in London. It is titled "Applications of Natural'
    ' Language Processing". There is a helpline number'
    " available at +44-1234567891. Gus is helping organize it."
    " He keeps organizing local Python meetups and several"
    " internal talks at his workplace. Gus is also presenting"
    ' a talk. The talk will introduce the reader about "Use'
    ' cases of Natural Language Processing in Fintech".'
    " Apart from his work, he is very passionate about music."
    " Gus is learning to play the Piano. He has enrolled"
    " himself in the weekend batch of Great Piano Academy."
    " Great Piano Academy is situated in Mayfair or the City"
    " of London and has world-class piano instructors."
)
most_common_words = get_most_common_words(complete_text)
print(most_common_words)


def print_token_details(text):
    """Process text with spaCy and print detailed information about each token."""

    # Process the text
    doc = nlp(text)
    
    # Iterate over each token in the document
    for token in doc:
        # Print detailed information about the token
        print(
            f"""
TOKEN: {str(token)}
=====
TAG: {str(token.tag_):10} POS: {token.pos_}
EXPLANATION: {spacy.explain(token.tag_)}"""
        )


# Example usage
about_text = (
    "Gus Proto is a Python developer currently"
    " working for a London-based Fintech"
    " company. He is interested in learning"
    " Natural Language Processing."
)
print_token_details(about_text)


def visualize_dependency_parse(text, style="dep"):
    """Process text with spaCy and visualize the dependency parse tree."""

    # Process the text
    doc = nlp(text)
    
    # Visualize the dependency parse tree
    displacy.serve(doc, style=style)


# Example usage
about_interest_text = (
    "He is interested in learning Natural Language Processing."
)
visualize_dependency_parse(about_interest_text)
