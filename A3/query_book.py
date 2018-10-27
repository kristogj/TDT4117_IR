import random
import codecs
import string
import re
from nltk.stem.porter import PorterStemmer
import gensim
import logging

random.seed(123)
# For logging events
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def text_to_paragraphs(filename):
    """
    Takes the path to a file, reads it and split on each paragraph
    :param filename: String
    :return: List[String]
    """
    f = codecs.open(filename, "r", "utf-8")
    paragraphs = []
    paragraph = []
    for line in f:
        if len(line) <= 2:
            if len(paragraph) == 0:
                continue
            paragraphs.append(" ".join(paragraph))
            paragraph = []
        else:
            paragraph.append(line)
    f.close()
    return paragraphs


def filter_paragraphs(par):
    """
    Removes paragraphs with Gutenberg in it
    :param par: List[String]
    :return: List[String]
    """
    return list(filter(lambda p: 'Gutenberg' not in p, par))


def remove_punctuation(par):
    """
    Removes !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\n\r\t and replace it with empty space
    :param par: List[String]
    :return: List[String]
    """
    regex = re.compile('[%s]' % re.escape(string.punctuation + "\n\r\t"))
    return list(map(lambda p: regex.sub(" ", p).lower(), par))


def tokenize_paragraphs(par):
    """
    Takes a list of strings and splits every string to a list of words
    :param par: List[String]
    :return: List[List[String]]
    """
    return list(map(lambda p: p.split(" "), par))


def stem_words(tok_par):
    """
    Stemming is the process of reducing inflected words to their word stem (base / root from)
    Example: fishing, fished and fisher has the stem fish
    :param tok_par: List[List[String]]
    :return: List[List[String]]
    """
    stemmer = PorterStemmer()
    return [list(map(lambda word: stemmer.stem(word), par)) for par in tok_par]



def get_stopwords():
    """
    Returns a list of stopwords that is located in stopwords.txt
    :return: List[String]
    """
    f = open("stopwords.txt", "r")
    res = f.read().split(",")
    f.close()
    return res


def remove_stopwords(dictionary):
    """
    Takes the dictionary made with gensim and remove all the stopwords
    :param dictionary: gensim.corpa Dictionary
    :return: null
    """
    stopwords = get_stopwords()
    # Remove stopwords that are not in the keys set of the dictionary
    stopwords = list(filter(lambda word: word in dictionary.token2id.keys(), stopwords))
    # Map the stopwords to the ids
    stop_ids = list(map(lambda word: dictionary.token2id[word], stopwords))
    # Remove stopwords from dictionary
    dictionary.filter_tokens(stop_ids)


def paragraph_to_bow(dictionary, tok_par):
    """
    Convert document into the bag-of-words (BoW) format = list of (token_id, token_count) tuples.
    :param dictionary: gensim.corpa Dictionary
    :param tok_par: List[List[String]]
    :return: List[List[(int,int)]]
    """
    return list(map(lambda par: dictionary.doc2bow(par), tok_par))


def preprocessing(dictionary, query):
    """
    First lower, strip, tokinize and stem the query then convert query into the
    bag-of-words (BoW) format = list of (token_id, token_count) tuples.
    :param dictionary: gensim.corpa Dictionary
    :param query: List[String]
    :return: List[List[(int,int)]]
    """
    low = list(map(lambda q: q.lower(), query))
    stripped = remove_punctuation(low)
    tokenize = tokenize_paragraphs(stripped)
    stemmed = stem_words(tokenize)
    bow = paragraph_to_bow(dictionary, stemmed)
    return bow


def print_result(original, res):
    """
    Truncate and print the results paragraphs
    :param original: List[String]
    :param res: List[(int,int)]
    :return: null
    """
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
    original = filter_paragraphs(original)

    # Remove punctuation
    removed_punctuation = remove_punctuation(original)

    # Tokenize paragraphs
    tokenize = tokenize_paragraphs(removed_punctuation)

    # Stem the words
    stemmed = stem_words(tokenize)

    """
    Dictionary Building
    """

    # Build a dictionary
    dictionary = gensim.corpora.Dictionary(stemmed)
    dictionary_size = len(dictionary)

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
    tfidf_index = gensim.similarities.MatrixSimilarity(tfidf_corpus, num_features=dictionary_size)

    # Repeat for LSI model
    lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
    lsi_corpus = lsi_model[tfidf_corpus]
    lsi_index = gensim.similarities.MatrixSimilarity(lsi_corpus, num_features=dictionary_size)

    # Report and try to interpret first 3 LSI topics
    lsi_model.print_topics(3)

    """
    Querying
    """
    query = ["How taxes influence Economics?"]

    # Prepare the query
    print("#TF-IDF MODEL")
    vec_bow = preprocessing(dictionary, query)

    # Convert bow to TF-IDF rep
    vec_tfidf = tfidf_model[vec_bow][0]

    # Report the 3 most relevant paragraphs according to TF-IDF model
    doc2similarity = enumerate(tfidf_index[vec_tfidf])
    res_tfidf = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
    print("TF-IDF SIM")
    print_result(original, res_tfidf)

    # Prepare the query
    print("#LSIMODEL")
    vec_lsi = lsi_model[vec_tfidf]
    lsi_query_sorted = sorted(vec_lsi, key=lambda kv: -abs(kv[1]))[:3]

    # Report the 3 topics with most significant weight
    topic_ids = list(map(lambda tup: tup[0], lsi_query_sorted))
    for ids in topic_ids:
        print(f"[topic {ids}]")
        print(lsi_model.print_topic(ids) + "\n")

    # Report the 3 most relevant paragraphs according to LSI model
    doc2similarity = enumerate(lsi_index[lsi_query_sorted])
    res_lsi = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
    print_result(original, res_lsi)


main()
