# -*- coding: utf-8 -*-

from gensim import utils
from gensim.parsing.preprocessing import STOPWORDS
import itertools
import logging
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import tarfile
from text2topics.phrases import ENTITIES
from textblob import TextBlob

## File processing - adapted from gensim api and tutorials. 

def process_page(page):
    """
    Preprocess a single periodical page, returning the result as
    a unicode string.

    Removes all non-alpha characters from the text.

    Args:
        page (str): Passes in the page object

    Returns:
        str: Content of the file, but without punctuation and non-alpha characters.
    """
    content = utils.any2unicode(page, 'utf8').strip()
    content = re.sub(r"[^a-zA-Z]", " ", content)
    
    return content


def iter_Periodicals(fname, log_every=500):
    """
    Yield plain text of each periodical page, as a unicode string. Extracts from a zip of the entire corpus.

    Args:
        fname (str): Name of the archive file.
        log_every (int): Logging frequency to report on the files.

    Yields:
        str: Yields the content of the file after passing it through the :func:`process_page` function.
    """
    doc_id = 0
    with tarfile.open(fname, 'r:gz') as tf:
        for file_number, file_info in enumerate(tf):
            if file_info.isfile():
                if log_every and doc_id % log_every == 0:
                    logging.info("extracting file #%i: %s" % (doc_id, file_info.name))
                title = file_info.name[2:]
                content = tf.extractfile(file_info).read()
                yield title, doc_id, process_page(content)
                doc_id += 1

## Additional steps

def connect_phrases(content, entities=ENTITIES):
    """Convert named entities into a single token.

    Args:
        content ():
        entities (str): List of frequent phrases, calculated separately and loaded from file for convenience.
    Yields:
        str: text of the file converted to lower case.
    """
    phrases = []
        
    # Use TextBlob to identify candidate phrases in incomming pages.
    for np in TextBlob(content).noun_phrases:
        if ' ' in np and np.lower() in entities:            
            phrases.append(np.lower())

    # Convert content of files to lower
    content = content.lower()
    
    # Work through the identified phrases and connect with an underscore.
    for phrase in phrases:
        replacement_phrase = re.sub('\s', '_', phrase)
        content = re.sub(phrase, replacement_phrase, content)

    return content

def lemmatize_tokens(tokens):
    """Convert tokens to lemmas.

    References:
        :func:`nltk.stem.wordnet.WordNetLemmatizer`
    """
    lemmatizer = WordNetLemmatizer()
    lemma_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return lemma_tokens

def filter_tokens(tokens):
    """Filter out short and stopword tokens for clustering.

    References:
        :func:`gensim.parsing.preprocessing.STOPWORDS`
    """
    token_list = []
    for token in tokens:
        if len(token) > 2 and token not in STOPWORDS:
            token_list.append(token)
        else:
            continue
                
    return token_list

class Lemma_Corpus(object):
    """Adds lemmatization step to the standard corpus creation workflow.

    References:
        :func:`nltk.word_tokenize`
        :func:`connect_phrases`
        :func:`lemmatize_tokens`
        :func:`filter_tokens`
    """
    def __init__(self, fname):
        self.fname = fname

    def process_corpus(self, content):
        content = connect_phrases(content)
        tokens = word_tokenize(content)
        lemmas = lemmatize_tokens(tokens)

        return filter_tokens(lemmas)

    def __iter__(self, log_every=1000):
        for title, doc_id, content in iter_Periodicals(self.fname):
            if log_every and doc_id % log_every == 0:
                logging.info("{}".format(self.process_corpus(content)))
            yield title, doc_id, self.process_corpus(content)