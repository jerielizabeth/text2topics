# -*- coding: utf-8 -*-

from collections import defaultdict
import matplotlib.pyplot as plt
from nltk import FreqDist
import numpy as np
import operator
import os
import pandas as pd
import re
import seaborn as sns
from text2topics import utilities

## Collection of functions for evaluating words in text files against a list of known words. 

def identify_errors(tokens, dictionary):
    """Compare words in documents to words in dictionary. 

    Args:
        tokens (list): List of all tokens in the document.
        dictionary (set): The set of approved words.
    Returns:
        set : Returns the set of tokens in the documents that are not 
            also dictionary words.
    """
    return set(tokens).difference(dictionary)

def get_error_stats(errors, tokens):
    """ Returns a dictionary recording each error and its 
    frequency in the document.

    Uses the FreqDist function from NLTK.

    `errors` generated with the `identify_errors` function.
    References:
        :func:`nltk.FreqDist`
        :func:`identify_errors`

    Args:
        errors (set): Set of errors identified in `identify_errors`.
        tokens (list): Tokenized content of the file being evaluated.
    Returns:
        dict: dictionary with error and frequency.
    """
    freq_distribution = FreqDist(tokens) 
    
    error_report = {}
    for error in list(errors):
        error_count = freq_distribution[error]
        error_report.update({error:error_count})
        
    return error_report    


def total_errors(error_report):
    """ Calculates the total errors recorded in the document.

    Use reports.get_error_stats() to generate `error_report`.

    References:
        :func:`get_error_stats`

    Args:
        error_report (dict): Dictionary of errors and counts generated using `get_error_stats` function.
    Returns:
        int: sum of the errors reported in document.
    """
    return(sum(error_report.values()))


def error_rate(error_total, tokens):
    """ Calculates the error rate of the document to 3 decimal places.

    Use reports.total_errors() to calculate `error_total`

    References:
        :func:`total_errors`

    Args:
        error_total (int): Calculated using the `total_errors` function from the dictionary of errors and their counts.
        tokens (list): List of tokens that compose the text.
    Returns:
        int: returns the total errors divided by the total number of tokens, to 3 decimal places.
    """
    if len(tokens) > 0:
        return(float("{0:.3f}".format(error_total/len(tokens))))
    else:
        return(np.nan)


def generate_doc_report(text, spelling_dictionary):
    """ 
    Creates a report (dictionary) on each document that includes:
        - number of tokens (num_tokens)
        - number of unique tokens (num_unique_tokens)
        - number of errors (num_errors)
        - error rate for the document (error_rate)
        - dictionary of the errors and their counts (errors)

    Uses a number of functions, including
    References:
        :func:`utilities.strip_punct`
        :func:`utilities.tokenize_text`
        :func:`utilities.to_lower`
        :func:`identify_errors`
        :func:`get_error_stats`
        :func:`total_errors`
        :func:`error_rate`

    Args:
        text (str): the content of the file being evaluated
        spelling_dictionary (set): a set containing the collection of verified words.
    Returns:
        dict: dictionary with keys num_tokens (int), num_unique_tokens (int), num_errors (int), error_rate (int), and errors (dict)
    """
    text = utilities.strip_punct(text)
    tokens = utilities.tokenize_text(text)
    tokens = utilities.to_lower(tokens)
    errors = identify_errors(tokens, spelling_dictionary)
    error_report = get_error_stats(errors, tokens)
    error_total = total_errors(error_report)
    rate = error_rate(error_total, tokens)
    return {'num_tokens': len(tokens),
             'num_unique_tokens': len(set(tokens)),
             'num_errors': error_total,
             'error_rate': rate,
             'errors': error_report}


def process_directory(directory, spelling_dictionary):
    """ 
    Composit function for processing an entire directory of files.
    Returns the statistics on the whole directory as a list of dictionaries.

    Uses the following functions:
    References:
        :func:`utilities.readfile`
        :func:`generate_doc_report`

    Args:
        directory (str): the location of the directory of files to evaluate.
        spelling_dictionary (set): the set containing all verified words against which the document is evaluated.
    Returns:
        list: list of dictionaries with the doc reports for each document in a directory.
    """
    corpus = (f for f in os.listdir(directory) if not f.startswith('.') and os.path.isfile(os.path.join(directory, f)))
        
    statistics = []
    for document in corpus:
        content = utilities.readfile(directory, document)
        stats = generate_doc_report(content, spelling_dictionary)
        stats.update({"doc_id": document})
        statistics.append(stats)
 
    return(statistics) 


## Collection of functions for parsing the error statistics from documents and directory of documents. 

def get_errors_summary(statistics):
    """
    Get statistics on the errors for the whole directory.
    
    Creates a dictionary (errors_summary) from all the reported errors/frequencies
    that records the error (as key) and the total count for that error (as value).
    
    Developed using: http://stackoverflow.com/questions/11011756, 
    http://stackoverflow.com/questions/27801945/

    References:
        :func:`process_directory`

    Args:
        statistics (list): list of the reports from all the files in a directory. 

    Returns:
        dict: token identified as not a word and the number of times it occurs across whole directory.
    """
    all_errors = (report['errors'] for report in statistics)       
    
    errors_summary = defaultdict(int)
    for doc in all_errors:
        for key, value in doc.items():
            errors_summary[key] += value

    return errors_summary

def top_errors(errors_summary, min_count):
    """ 
    Use the errors_summary to report the top errors.

    References:
        :func:`get_errors_summary`

    Args:
        errors_summary (dict): dictionary of error keys and total count in directory.
        min_count (int): include errors if frequency is greater than this value.
    Returns:
        dict: dictionary of all the errors and counts greater than the min value.

    """
    # Subset errors_summary using the min_count
    frequent_errors = {key: value for key, value in errors_summary.items() if value > min_count}

    # return sorted list of all errors with a count higher than the min_count
    return sorted(frequent_errors.items(), key=operator.itemgetter(1), reverse=True)

def long_errors(errors_summary, min_length=10):
    """
    Use the error_summary to isolate tokens that are longer thatn the min_length. 

    Used to identify strings of words that have been run together due to the failure of the OCR engine to recognize whitespace.

    References:
        :func:`get_errors_summary`

    Args:
        errors_summary (dict): dictionary of errors and counts per document in directory.
        min_length (int): number of characters that tokens must be longer than to be included. 
    Returns:
        tuple: list of errors longer than min length and minimum length value.
    """
    errors = list(errors_summary.keys())

    return ([x for x in errors if len(x) > min_length], min_length)

def tokens_with_special_characters(errors_summary):
    """
    References:
        :func:`get_errors_summary`

    Args:
        errors_summary (dict):
    Returns:
        dict: sorted dictionary
    """
    errors = list(errors_summary.keys())

    special_characters = []
    for error in errors:
        if re.search("[^a-z0-9-']", error):
            special_characters.append(error)
        else:
            pass

    sc_dict = dict(map(lambda key: (key, errors_summary.get(key, None)), special_characters))

    return sorted(sc_dict.items(), key=operator.itemgetter(1), reverse=True)


def docs_with_high_error_rate(corpus_statistics, min_error_rate=.2):
    """
    References:
        :func:`process_directory` or :func:`overview_report`
    
    Args:
        corpus_statistics (dict):
        min_error_rate (int): default of .2. Values must be greater than `min_error_rate` for doc to be included in results.

    Returns:
        dict: doc_id and error_count for error_rates higher than min_error_rate.
    """

    # Gather list of doc_id and num_errors
    docs_2_errors = {}
    for report in corpus_statistics:
        docs_2_errors.update({report['doc_id']: report['error_rate']})

    # Subset dictionary to get only records with error_rate above minimum
    problem_docs = {key: value for key, value in docs_2_errors.items() if value > min_error_rate}

    # return dictionary with doc_id and error_count if error rate higher than min_error_rate
    return sorted(problem_docs.items(), key=operator.itemgetter(1), reverse=True)

def docs_with_low_token_count(corpus_statistics, max_token_count=350):
    """
    """
    # Gather list of doc_ids and total token count
    docs_2_tokens = {}
    for report in corpus_statistics:
        docs_2_tokens.update({report['doc_id']: report['num_tokens']})

    # Subset dictionary to get only records wth value below the max
    short_docs = {key: value for key, value in docs_2_tokens.items() if value < max_token_count}

    # return dictionary with doc_id and token_count if count is lower than max_token_count
    return (short_docs, max_token_count)


## Collection of functions that use the statistics in dataframe format, including a distribution chart

def stats_to_df( corpus_statistics ):
	"""Convert stats to dictionary.
    Convert dictionary of corpus statistics to a dataframe for computations.
    
    Uses :mod:`pandas`.

    References:
        Uses the data generated by :func:`process_directory`.

    Note:
        Use :func:`GoH.reports.process_directory` before running `stats_to_df`

    Args:
        corpus_statistics (dict): List of dictionaries with the information about all of the files in a directory.
        The corpus data should be formatted as follows::

            {
                'doc_id': document,
                'num_tokens': len(tokens),
                'num_unique_tokens': len(set(tokens)),
                'num_errors': error_total,
                'error_rate': rate,
                'errors': error_report
            }

    Returns:
        dataframe: Returns a dataframe with the following columns::

            `doc_id`
            `error_rate`
            `num_tokens`
            `num_errors`

	"""
	df = pd.DataFrame(corpus_statistics, columns=["doc_id", "error_rate", "num_tokens", "num_errors"])

	return df

def token_count(df):
    """
    Dependencies:
        `stats_to_df()`
    Args:
        df (dataframe): dataframe with doc_id, error_rate, num_tokens, and num_errors
    Returns:
        int: sum of all tokens in a directory
    """
    return df['num_tokens'].sum()


def average_verified_rate(df):
    """ To compute average error rate, add up the total number of tokens
    and the total number of errors """
    total_tokens = token_count(df)
    total_errors = df['num_errors'].sum()

    if total_tokens > 0:
        return (total_tokens - total_errors)/total_tokens
    else:
        return np.nan

def average_error_rate(df):
    """
    Args:
        df (dataframe):
    Returns:
        int: result of total number of errors divided by the total number of docs.
    """
    error_sum = df['error_rate'].sum()
    total_docs = len(df.index)

    return error_sum/total_docs


def chart_error_rate_distribution( df, title ):
    """
    References:
        :func:`sns.distplot`
    
    Args:
        df (dataframe):
        title (str): Abbreviation for periodical
    Returns:
        plot object
    """
    
    df = df[pd.notnull(df['error_rate'])]
    
    # graph the distribution of the error rates

    x = pd.Series(df['error_rate'], name="Error Rate per Periodical Page")
    ax = sns.distplot(x)
    
    return(ax)

def overview_report(directory, spelling_dictionary, title):
    """
    Processes the tokens for all documents in a directory and returns a list of the error reports from each document (from `process_directory` function) and generates a distribution chart of the error rates.

    References:
        :func:`process_directory`
        :func:`stats_to_df`
        :func:`average_verified_rate`
        :func:`average_error_rate`
        :func:`token_count`
        :func:`chart_error_rate_distribution`

    Args:
        directory (str): directory of text files to be evaluated.
        spelling_directory (str): directory containing word lists to use to verify tokens.
        title (): title of the periodical being evaluated.
    
    Returns:
        list: list of dictionaries with the doc reports for each document in a directory.

    """
    corpus_statistics = process_directory(directory, spelling_dictionary)

    df = stats_to_df(corpus_statistics)

    print("Directory: {}\n".format(directory))
    print("Average verified rate: {}\n".format(average_verified_rate(df)))
    print("Average of error rates: {}\n".format(average_error_rate(df)))
    print("Total token count: {}\n".format(token_count(df)))

    chart_error_rate_distribution(df, title)

    return corpus_statistics
