import random
import codecs
import string
import re
from nltk.stem.porter import PorterStemmer

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
    for par in range(0,len(tok_par)):
        res.append(list(map(lambda word: stemmer.stem(word), tok_par[par])))
    return res



def main():
    original = text_to_paragraphs("book.txt")

    # Filter headers and footers
    filtered = filter_paragraphs(original)

    # Remove punctuation
    removed_punctuation = remove_punctuation(filtered)

    # Tokenize paragraphs
    tokenize = tokenize_paragraphs(removed_punctuation)

    # Stem the words
    stemmed = stem_words(tokenize)


    for par in stemmed:
        for word in par:
            print(word)








main()