import nltk
#nltk.download('punkt',force=True)
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import numpy as np


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def  bag_of_words(tokenized_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

# a = "How long does shipping take?"
# print (a)

# a = tokenize(a)
# print (a)

# words = ["Organize","Organizing","organizes"]
# stemmed_words = [stem(w) for w in words]
# print (stemmed_words)

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# print(bag_of_words(sentence, words))

