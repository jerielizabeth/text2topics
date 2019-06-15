# -*- coding: utf-8 -*-

import operator
import re
from text2topics import utilities

## Shared Function

def replace_pair(pair, content):
    """
    Uses regex to locate a pair.
    First element of the tuple is the original error token.
    Second element of the tuple is the replacement token.
    References:
        :func:`re.sub`

    Args:
        pair (tuple): Error token, replacement token
        content (str): Content of the file
    Returns
        str: Corrected file content.
    """
    return re.sub(pair[0], ' {} '.format(pair[1]), content)


## Collection of functions for dealing with "burst" words

def find_split_words(pattern, content):
    """
    Args:
        pattern (): regex pattern
        content (str): document text
    Returns:

    """
    return pattern.findall(content)

def check_splits(pattern, spelling_dictionary, content, replacements):
    """
    Use regex to check if error is the result of a word that has been split in two.
    
    References:
        :func:`find_split_words`

    Args:
        pattern (str):
        spelling_dictionary ():
        content (str): text of the document
        replacements (list): empty list object that results are appended to.
    
    Returns:
        none

    (this is an error in the function - it appends to an existing list, rather than returning the list at the end of the function.)
    """
    # Use regex pattern to identify "split words"
    split_words = find_split_words(pattern, content)

    # Regex pattern finds last character as a separate match. Take the first match.
    for split_word in split_words:
        test_word = split_word[0]

        restored_word = re.sub(r'\s', r'', test_word)

        if restored_word.lower() in spelling_dictionary:
            replacements.append((split_word[0], restored_word))

        # Check if the restored word failed because it is two capitalized words combined
        # into one. Check for capital letter.
        elif re.search(r'[A-Z]', test_word):
            # Find the words by looking for Aaaa pattern
            words = re.findall('([A-Z][a-z]+)', test_word)
            for word in words:
                combo = re.sub(r'\s', r'', word)
                if combo.lower() in spelling_dictionary:
                    replacements.append((word, combo))
                else:
                    pass
        else:
            pass


## Collection of functions for correcting split words

def period_at_end(token):
    """
    Args:
        token (str): word being evaluated
    Returns:
        binary: True if last character is a period, false if not.
    """
    if list(token).pop() is ".":
        return True
    else:
        return False

def create_substitution(tokens, stem, get_prior, spelling_dictionary):
    """
    Args:
        tokens ():
        stem ():
        get_prior ():
        spelling_dictionary ():
    Returns:

    """
    locations = [i for i, j in enumerate(tokens) if j == stem]
    for location in locations:
        # Option 1
        if get_prior:
            prior_word = tokens[location - 1]
            sub_word = ''.join([prior_word, stem])

            if sub_word.lower() in spelling_dictionary:
                return (prior_word, stem)
            else:
                pass
        # Option 2
        else:
            try:
                next_word = tokens[location + 1]
                sub_word = ''.join([stem, next_word])

                if sub_word.lower() in spelling_dictionary:
                    return (stem, next_word)
                else:
                    if period_at_end(sub_word):
                        sub_stripped = "".join(list(sub_word)[:-1])
                        if sub_stripped.lower() in spelling_dictionary:
                            return (stem, "".join(list(next_word)[:-1]))
                        else:
                            pass
                    else:
                        pass
            except IndexError:
                pass

def check_if_stem(stems, spelling_dictionary, tokens, get_prior=True):
    """
    References:
        :func:`period_at_end`
        :func:`create_substitution`
    Args:
        stems ():
        spelling_dictionary ():
        tokens ():
        get_prior ():
    Returns:
        list: List of 

    """
    replacements = []
    for stem in stems:
        if len(stem) > 1:
            if period_at_end(stem):
                stem_stripped = "".join(list(stem)[:-1])
                if not stem_stripped.lower() in spelling_dictionary:
                    result = create_substitution(tokens, stem, get_prior, spelling_dictionary)
                    if result is None:
                        pass
                    else:
                        replacements.append(result)

            else:
                if not stem.lower() in spelling_dictionary:
                    result = create_substitution(tokens, stem, get_prior, spelling_dictionary)
                    if result is None:
                        pass
                    else:
                        replacements.append(result)

    return replacements

def replace_split_words(pair, content):
    """
    """
    return re.sub('{}\s+{}'.format(pair[0], pair[1]), '{}{}'.format(pair[0], pair[1]), content)


## Functions for dealing with long tokens due to OCR recognition errors

def check_for_repeating_characters(tokens, character):
    """
    References:
        :func:`re.findall`

    Args:
        tokens ():
        character ():
    Returns:

    """
    replacements = []
    pattern = "([" + character + "{2,}]{2,4})"

    for token in tokens:
        if len(token) > 12:
            if not re.findall(r'{}'.format(pattern), token):
                pass
            else:
                m_strings = re.findall(r'{}'.format(pattern), token)
                if len(m_strings) > 2:
                    replacements.append((token, ' '))
                else:
                    pass
        else:
            pass

    return replacements

## Functions for Dealing with Squashed Words

def get_approved_tokens(content, spelling_dictionary, verified_tokens):
    """
    References:
        :func:`utilities.strip_punct`
        :func:`utilities.tokenize_text`
    
    Note:
        Error again in function development - rather than returning a completed list, the list exists outside of the function.

    Args:
        content (str): document text
        spelling_dictionary (list): 
        verified_tokens (list): empty list

    Returns:
        None
    """
    text = utilities.strip_punct(content)
    tokens = utilities.tokenize_text(text)

    for token in tokens:
        if token.lower() in spelling_dictionary:
            verified_tokens.append(token)


def verify_split_string(list_split_string, spelling_dictionary):
    """
    References:
        :func:`
    
    Args:
        list_split_string ():
        spelling_dictionary ():
    
    Returns:
        binary: 
    """
    total_len = len(list_split_string)
    verified_splits = 0
    short_splits = 0
    
    for split in list_split_string:
        if split.lower() in spelling_dictionary:
            verified_splits = verified_splits + 1
            if len(split) < 3:
                short_splits = short_splits +1
        else:
            pass

    if verified_splits / total_len > .6:
        if short_splits / total_len < .5:
            return True
    else:
        return False


def infer_spaces(s, wordcost, maxword):
    """Uses dynamic programming to infer the location of spaces in a string without spaces.

    Solution from http://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words

    Args:
        s (str): long token made up of multiple words
        wordcost (dict): (magic from SO- `dict((k, log((i+1)*log(len(sorted_list_of_words)))) for i,k in enumerate(sorted_list_of_words))`)
        maxword (int): maximum length of tokens in document
    
    Returns:
        str: string of words from original token.    
    """

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))