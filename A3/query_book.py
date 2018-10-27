import random
import codecs
import string
import re
from nltk.stem.porter import PorterStemmer
import gensim
from pprint import pprint
import os
import logging

random.seed(123)
# For logging events
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
            paragraph.append(line)
    f.close()
    return paragraphs


def filter_paragraphs(par):
    """

    :param par: List[String]
    :return: List[String]
    """
    return list(filter(lambda p: 'Gutenberg' not in p, par))


def remove_punctuation(par):
    """

    :param par: List[String]
    :return: List[String]
    """
    regex = re.compile('[%s]' % re.escape(string.punctuation + "\n\r\t"))
    return list(map(lambda p: regex.sub("", p).lower(), par))


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


def preprocessing(dictionary, query):
    low = list(map(lambda q: q.lower(),query))
    stripped = remove_punctuation(low)
    tokenize = tokenize_paragraphs(stripped)
    stemmed = stem_words(tokenize)
    bow = paragraph_to_bow(dictionary, stemmed)
    return bow


def print_result(original, res):
    for tup in res:
        index, score = tup
        # Get result paragraph
        par = original[index]
        # Truncate to 5 lines
        par_5lines = par.split("\n")[0:5]
        # Print result
        print(f"[paragraph {index}]")
        print("\n".join(par_5lines) + "\n")


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
    # Build TF_IDF model
    tfidf_model = gensim.models.TfidfModel(bow)

    # Map Bags-of-Words into TF_IDF weights
    tfidf_corpus = tfidf_model[bow]

    # Construct MatrixSimilarity object thath let us calculate similarities between paragraphs and queries
    tfidf_index = gensim.similarities.MatrixSimilarity(tfidf_corpus)

    # Repeat for LSI model
    lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
    lsi_corpus = lsi_model[tfidf_corpus]
    lsi_index = gensim.similarities.MatrixSimilarity(lsi_corpus)


    # Report and try to interpret first 3 LSI topics
    lsi_model.print_topics(3)

    """
    Querying
    """
    query = ["How taxes influence Economics?"]

    # Preprocess to bow
    query = preprocessing(dictionary, query)

    # Convert bow to TF-IDF rep
    tfidf_query = tfidf_model[query][0]

    # Do query and report top 3 most relevant paragraphs
    print("#TF-IDF MODEL")
    doc2similarity = enumerate(tfidf_index[tfidf_query])
    res_tfidf = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
    print("TF-IDF SIM", res_tfidf)
    print_result(original, res_tfidf)

    # Do the same with the LSI model
    lsi_query = lsi_model[tfidf_query]
    print("#LSIMODEL")
    lsi_query_sorted = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]
    print("LSI_MODEL TOPICS",lsi_model.show_topics())
    doc2similarity = enumerate(lsi_index[lsi_query_sorted])
    res_lsi = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
    print_result(original, res_lsi)


















main()