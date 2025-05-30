import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words, all_sentences=None):
    """
    Return TF-IDF vector for the sentence:
    - TF: Term frequency in the sentence (normalized by sentence length)
    - IDF: Inverse document frequency based on all_sentences
    If all_sentences is None, fall back to binary bag-of-words
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    tfidf = [  0 ,  0.3 ,  0 , 0.2 ,   0 ,    0 ,    0] (example values)
    """
    # Stem each word in the sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    if all_sentences is None:
        # Fall back to binary bag-of-words
        bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in sentence_words:
                bag[idx] = 1
        return bag
    
    # Calculate TF (term frequency)
    tf = np.zeros(len(words), dtype=np.float32)
    for w in sentence_words:
        for idx, word in enumerate(words):
            if word == w:
                tf[idx] += 1
    if len(sentence_words) > 0:  # Avoid division by zero
        tf = tf / len(sentence_words)  # Normalize TF
    
    # Calculate IDF (inverse document frequency)
    idf = np.zeros(len(words), dtype=np.float32)
    for idx, word in enumerate(words):
        df = sum(1 for s in all_sentences if word in [stem(w) for w in s])  # Count sentences containing the word
        idf[idx] = np.log(len(all_sentences) / (df + 1))  # Add 1 to avoid division by zero
    
    # Calculate TF-IDF
    tfidf = tf * idf
    return tfidf