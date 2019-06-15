# -*- coding: utf-8 -*-

"""
The utilities package provides a collection of functions used to prepare data for different
analysis tasks. These are largely helper functions, but some can operate on their own.

Examples:
    >>> text = GoH.utilities.readfile( input_dir, filename)
    >>> spelling_dictionary = GoH.utilities.create_spelling_dictionary( wordlists, directory )

"""

from nltk.tokenize import WhitespaceTokenizer
from nltk import word_tokenize
import os
import pandas as pd
import re
import tarfile

def readfile( input_dir, filename ):
	"""Reads in file from directory and file name.
	Returns the content of the file.

    Usage::

        >>> text = readfile(input_dir, filename)

    Args:
        input_dir (str): Directory with the input file.
        filename (str): Name of file to be read.

    Returns:
        str: Returns the content of the file as a string.

	"""
	with open(os.path.join(input_dir, filename)) as f:
		return f.read()

def strip_punct( text ):
    """Remove punctuation and numbers.
    Remove select punctuation marks and numbers from a text and replaces them with a space.
    
    Non-permanent changes for evaluation purposes only.
    
    Uses the :mod:`re` library.

    Args:
        text (str): Content to be evaluated.

    Returns:
        str: Returns the content string without the following characters: 0-9,.!?$:;&".
	"""
    text_cleaned = re.sub(r"[0-9,.!?$:;&\"]", " ", text)
    
    return text_cleaned

def tokenize_text( text, tokenizer='whitespace' ):
    """Converts file content to a list of tokens. 

    Uses :meth:`nltk.tokenize.regexp.WhitespaceTokenizer`.

    Args:
        text (str): Content to be tokenized.
        tokenizer(str): option of tokenizer. Current options are 'whitespace' and
            'word'.

    Returns:
        list: Returns a list of the tokens in the text, separated by white space.
	"""
    if tokenizer == 'whitespace' or tokenizer == 'word':
        if tokenizer == 'whitespace':
            return WhitespaceTokenizer().tokenize(text)
        elif tokenizer == 'word':
            return word_tokenize(text)
    else:
        raise ValueError('Tokenizer value {} is invalid. Must be "whitespace" or "word"'.format(tokenizer))


def to_lower( tokens ):
    """Convert all tokens to lower case.

    Args:
        tokens (list): List of tokens generated using a tokenizer.

    Returns:
        list: List of all tokens converted to lowercase.
	"""
    return [w.lower() for w in tokens]


def create_spelling_dictionary( directory, wordlists ):
    """Compile a spelling dictionary.
    Compiles a spelling dictionary from one or multiple
    wordlist files. Returns results as a set.

    References:
        :func:`GoH.utilities.readfile`

    Args:
        directory (str): Location of the wordlist files.
        wordlists (list): List of filenames for the wordlist files.


    Returns:
        set: List of unique words in all the compiled lists.
    """
    spelling_dictionary = []
    for wordlist in wordlists:
        words = readfile(directory, wordlist).splitlines()
        word_list = [w.lower() for w in words]
        for each in word_list:
            spelling_dictionary.append(each)

    return set(spelling_dictionary)


def get_year( page_id ):
    """Extract year information from a page ids.

    Note:
        File names must be structured as follows::

            TITLEYYYYMMDD-V00-00-page0.txt
        
        or::

            TITLEYYYYMMDD-V00-00.pdf

    Args:
        page_id (str): Filename to parse, format according to the note.
    
    Returns:
        str: Returns the first four digits, which corresponds to the year of publication.
    """
    split_id = page_id.split('-')
    dates = re.search(r'[0-9]+', split_id[0])
    
    return dates.group()[:4]

def get_title( page_id ):
    """Extract year information from a page ids.
    
    Note:
        File names must be structured as follows::

            TITLEYYYYMMDD-V00-00-page0.txt
        
        or::

            TITLEYYYYMMDD-V00-00.pdf
    
    Args:
        page_id (str): Filename to parse, formatted according to the note.
    
    Returns:
        str: Returns the title information from the page id.
    """
    split_id = page_id.split('-')
    title = re.match("[A-Za-z]+", split_id[0])
    
    return title.group()

def open_original_docs(filenames,
    pdf_dir='/Users/jeriwieringa/Dissertation/text/corpus-pdf/periodicals/', 
    text_dir='/Users/jeriwieringa/Dissertation/text/text/2017-04-Final-Corpus/'):
    """Opens the PDF and TXT files for a list of page ids.
    Used to verify the content of files that report high error rates or unusual error information.

    Args:
        pdf_dir (str): Path to directory with PDF files.
        text_dir (str): Path to directory with TXT files.
        filenames (list): List of filenames to open.
    """
    print("Opened files: \n")

    for filename in filenames:
        base_filename = filename.split('-')[:-1]
        pdf_filename = "{}.pdf".format('-'.join(base_filename))

        os.system("open {}".format(os.path.join(text_dir, filename)))
        os.system("open {}".format(os.path.join(pdf_dir, pdf_filename)))

        print("{}\n".format(filename))

def define_directories( prev, cycle, base_dir ):
    """Helper function for iterating through document cleaning.
    This function redefines the directory information for each round of cleaning.

    Args:
        prev (str): Name of cycle that was just completed.
        cycle (str): Name of current cycle.
        base_dir (str): Root directory

    Returns:
        dict: Dictionary with keys `prev` and `cycle` and values of the corresponding directory paths.
    """
    return {'prev': os.path.join(base_dir, prev), 'cycle': os.path.join(base_dir, cycle)}

def extract_words_from_dictionary(filepath):
    """Helper function for extracting a list of tokens from the output of Gensim's id2word function.
    Uses Pandas to load dictionary data as a dataframe. Returns tokens as a list.

    Args:
        filepath (str): Path to Gensim dictionary file (txt)
    Returns:
        list: List of unique tokens.
    """
    with open(filepath) as f:
        df = pd.read_table(f, header=None, names=['word_id', 'word', 'count'])
    
    return df.word.unique().tolist()

def create_tar_files(corpusDir, samplePrefix, tarFullCorpusObject, selectList):
    """Creates two corpus tar files from a selection list, a sample file and a holdout file.

    Note:
        Numbers in filenames denote [min tokens, max error rate, percent included]
        
    Args:
        corpusDir (str): Output path for tar files
        samplePrefix (str): Unique identifier for tar files
        tarFullCorpusObject (): Tar object from the full corpus
        selectList (list): List of filenames (basenames) to include in sample.
    Returns:
        No return     
    """
    SampleTar = tarfile.open(os.path.join(corpusDir, '{}Sample.tar.gz'.format(samplePrefix)), 'w:gz')
    HoldoutTar = tarfile.open(os.path.join(corpusDir, '{}Holdout.tar.gz'.format(samplePrefix)), 'w:gz')

    #Skip first member of tar file, as it is the directory
    for member in tarFullCorpusObject.getmembers()[1:]:
        if os.path.basename(member.name) in selectList:
            SampleTar.addfile(member, tarFullCorpusObject.extractfile(member))
        else:
            HoldoutTar.addfile(member, tarFullCorpusObject.extractfile(member))

    SampleTar.close()
    HoldoutTar.close()