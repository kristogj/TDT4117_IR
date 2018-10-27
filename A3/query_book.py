import random
import codecs
import string
import re
from nltk.stem.porter import PorterStemmer
import gensim
from pprint import pprint
import os
random.seed(123)


def text_to_paragraphs(filename):
    """

    :param filename: String
    :return: List[String]
    """
    f = codecs.open(filename, "r", "utf-8")
    paragraphs = []
    paragraph = []
    for line in f:
        if len(line) <= 2:
            paragraphs.append("".join(paragraph))
            paragraph = []
        else:
            paragraph.append(line.lower())
    f.close()
    return paragraphs


def filter_paragraphs(par):
    """

    :param par: List[String]
    :return: List[String]
    """
    return list(filter(lambda p: 'gutenberg' not in p and len(p) > 3, par))


def remove_punctuation(par):
    """

    :param par: List[String]
    :return: List[String]
    """
    regex = re.compile('[%s]' % re.escape(string.punctuation + "\n\r\t"))
    return list(map(lambda p: regex.sub("", p), par))


def tokenize_paragraphs(par):
    """

    :param par: List[String]
    :return: List[List[String]]
    """
    return list(map(lambda p: p.split(" "), par))


def stem_words(tok_par):
    """

    :param tok_par: List[List[String]]
    :return: List[List[String]]
    """
    stemmer = PorterStemmer()
    res = []
    for par in range(0, len(tok_par)):
        res.append(list(map(lambda word: stemmer.stem(word), tok_par[par])))
    return res


def get_stopwords():
    """

    :return: List[String]
    """
    f = open("stopwords.txt", "r")
    res = f.read().split(",")
    f.close()
    return res


def remove_stopwords(dictionary):
    stopwords = get_stopwords()
    stopwords = list(filter(lambda word: word in dictionary.token2id.keys(), stopwords))
    stop_ids = list(map(lambda word: dictionary.token2id[word], stopwords))
    dictionary.filter_tokens(stop_ids)


def paragraph_to_bow(dictionary, tok_par):
    return list(map(lambda par: dictionary.doc2bow(par), tok_par))


def main():
    """
    Data loading and preprocessing
    """
    # Read the book convert it to list of paragraphs
    original = text_to_paragraphs("book.txt")

    # Filter headers and footers
    filtered = filter_paragraphs(original)

    # Remove punctuation
    removed_punctuation = remove_punctuation(filtered)

    # Tokenize paragraphs
    tokenize = tokenize_paragraphs(removed_punctuation)

    # Stem the words
    stemmed = stem_words(tokenize)

    """
    Dictionary Building
    """

    # Build a dictionary
    dictionary = gensim.corpora.Dictionary(stemmed)

    # Remove stopwords
    remove_stopwords(dictionary)

    # Map paragraphs into Bags-of-Words - each paragraph is a list of pairs (word-index, word-count)
    bow = paragraph_to_bow(dictionary, stemmed)

    """
    Retrival Models
    """














main()