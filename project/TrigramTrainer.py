#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.

Modified by Almir Aljic & Alexander Jakobsen in December 2019. Original version, named "BigramTrainer", has been extended and thus
re-named to "TrigramTrainer".
"""


class TrigramTrainer(object):
    """
    This class constructs a trigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file @code{f}.
        """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            #text = reader = str(text_file.read()).lower() # lower() means no capitalization.
            text = reader = str(text_file.read()) # Maintaining capitalization.
        try :
            self.tokens = nltk.word_tokenize(text) # Important that it is named self.tokens for the --check flag to work
        except LookupError :
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)


    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram,
        bigram and trigram counts.

        :param token: The current word to be processed.
        """

        n = len(self.index) # Number of unique words thus far.
        self.total_words += 1

        if token in self.index:
            self.unigram_count[token] += 1
        else:
            self.index[token] = n # Given a word, get its index from this dict.
            self.word[n] = token # Get the word with a given index.
            self.unigram_count[token] = 1

        # Set bigram count.
        if self.last_index != -1:
            prev_word = self.word[self.last_index] # Get the previous word.
            if prev_word in self.bigram_count:
                if token in self.bigram_count[prev_word]:
                    self.bigram_count[prev_word][token] += 1
                else:
                    self.bigram_count[prev_word][token] = 1
            else:
                self.bigram_count[prev_word][token] = 1

        # Set trigram count.
        if self.sub_two_index != -1:
            sub_two_word = self.word[self.sub_two_index]
            prev_word = self.word[self.last_index]

            if sub_two_word in self.trigram_count:
                if prev_word in self.trigram_count[sub_two_word]:
                    if token in self.trigram_count[sub_two_word][prev_word]:
                        self.trigram_count[sub_two_word][prev_word][token] += 1
                    else:
                        self.trigram_count[sub_two_word][prev_word][token] = 1
                else:
                    self.trigram_count[sub_two_word][prev_word][token] = 1
            else:
                self.trigram_count[sub_two_word][prev_word][token] = 1

        self.sub_two_index = int(self.last_index)
        self.last_index = self.index[token] # Index of most recently used (current iteration) token.
        self.unique_words = len(self.index)

    def stats(self):
        """
        Creates a list of rows to print of the language model.
        """
        rows_to_print = []
        bigram_rows = []
        trigram_rows = []

        first_row = str(self.unique_words) + ' ' + str(self.total_words)
        rows_to_print.append(first_row)

        # Frequency of occurrence of all unique words
        for i in range(len(self.word)):
            word = self.word[i]
            frequency_of_word = self.unigram_count[word]
            rows_to_print.append(str(i) + ' ' + word + ' ' + str(frequency_of_word))
            # Calculate bigram probabilities
            for second_word in self.bigram_count[word]:
                bigram_occurrences = self.bigram_count[word][second_word]
                p = str("%.15f" % math.log(bigram_occurrences/frequency_of_word))
                bigram_rows.append(str(self.index[word]) + ' ' + str(self.index[second_word]) + ' ' + p)

                # Calculate trigram probabilities.
                for third_word in self.trigram_count[word][second_word]:
                    trigram_occurrences = self.trigram_count[word][second_word][third_word]
                    p = str("%.15f" % math.log(trigram_occurrences/bigram_occurrences))
                    trigram_rows.append(str(self.index[word]) + ' ' + str(self.index[second_word]) + ' ' + str(self.index[third_word]) + ' ' + p)

        for row in bigram_rows:
            rows_to_print.append(row)
        EOB = "-2" # Signifies end of bigrams.
        rows_to_print.append(EOB)

        for row in trigram_rows:
            rows_to_print.append(row)

        EOF = "-1" # Signifies end of file.
        rows_to_print.append(EOF)

        return rows_to_print

    def __init__(self):
        """
        <p>Constructor. Processes the file <code>f</code> and builds a language model
        from it.</p>

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # The unigram counts.
        self.unigram_count = defaultdict(int)

        # The bigram counts.
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # The trigram counts.
        nested_dict = lambda: defaultdict(nested_dict)
        self.trigram_count = nested_dict()

        # The identifier of the previous word processed.
        self.last_index = -1

        # The identifier of the word processed 2 iterations ago.
        self.sub_two_index = -1

        # Number of unique words in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0


def main():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='TrigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()

    trigram_trainer = TrigramTrainer()

    trigram_trainer.process_files(arguments.file)

    stats = trigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)


if __name__ == "__main__":
    main()
