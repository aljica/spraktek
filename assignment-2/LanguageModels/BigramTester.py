#  -*- coding: utf-8 -*-
import math
import argparse
import nltk
import codecs
from collections import defaultdict
import json
import requests

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter.
        """
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        # Important that it is named self.logProb for the --check flag to work
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                # YOUR CODE HERE
                for i in range(self.unique_words):
                    _, word, frequency = map(str, f.readline().strip().split(' '))
                    self.index[i], self.word[word], self.unigram_count[word] = word, i, int(frequency)
                for line in f:
                    if line.strip() == "-1":
                        break
                    # Get index of first word and second word respectively, and their bigram prob
                    i, j, prob = map(str, line.strip().split(' '))
                    first_word, second_word = self.index[int(i)], self.index[int(j)]
                    self.bigram_prob[int(i)][int(j)], self.bigram_prob[first_word][second_word] = float(prob), float(prob)
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False


    def compute_entropy_cumulatively(self, word):
        # YOUR CODE HERE
        self.test_words_processed += 1

        if word in self.word:
            # If the current word (from the test corpus) we're examining was also in our training corpus
            index_of_current_word = self.word[word] # Get index of word being examined
            if self.last_index == -1:
                # The previous word in the test corpus does not appear in our language model, so we use linear interpolation
                # Note that P(word|previous_word) = 0, but P(word) = self.unigram_count[word] / self.total_words
                # i.e. P(word) is the frequency of occurrence of the given word in our training corpus
                prob_word = self.unigram_count[word] / self.total_words
                total_prob = self.lambda2 * prob_word + self.lambda3
                self.logProb += math.log(total_prob)
                self.last_index = index_of_current_word # New last_index is the index of the word currently being examined
                return
            else:
                previous_word = self.index[self.last_index] # Previous word

            # The two words previous_word and word [corresponding to w1 and w2 respectively] then both appear in our
            # language model. Now we have to determine whether or not the unison of those words, in the given order, i.e. the bigram,
            # appears in our language model. If so, we can get that bigram's corresponding log probability.
            if word in self.bigram_prob[previous_word]:
                # If the word we're examining ever appeared in the order [previous_word + ' ' + word] in our training corpus, then
                # do linear interpolation, using log probability from our language model
                prob_bigram = math.exp(self.bigram_prob[previous_word][word]) # We need to take the exponent here, we'll take the log later
                prob_word = self.unigram_count[word] / self.total_words
                total_prob = self.lambda1 * prob_bigram + self.lambda2 * prob_word + self.lambda3
                self.logProb += math.log(total_prob)
            else:
                # However, if the bigram did NOT appear, we use linear interpolation
                # Note that the circumstances are as follows: previous_word and word both exist in our language model.
                # However, the bigram of previous_word and word does not exist in our language model. As a result,
                # the probability P(word|previous_word) = 0 according to our language model. Simultaneously,
                # P(word) = self.unigram_count[word] / self.total_words. Same situation as previously:
                prob_word = self.unigram_count[word] / self.total_words
                total_prob = self.lambda2 * prob_word + self.lambda3
                self.logProb += math.log(total_prob)

            # Regardless of whether or not the bigram also exists in our language model, we have to set the new self.last_index:
            self.last_index = index_of_current_word
        else:
            # Otherwise, if word found in test corpus is not in our language model, then it means
            # that the last index ought to be -1. As for probability estimation, we use linear interpolation, but
            # note that P(word|previous_word) = 0 and P(word) = 0, so:
            self.logProb += math.log(self.lambda3)
            self.last_index = -1

    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower()) # Important that it is named self.tokens for the --check flag to work
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
                ### Added line below
                self.logProb = self.logProb / -self.test_words_processed # Average logProb
            return True
        except IOError:
            print('Error reading testfile')
            return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')
    parser.add_argument('--check', action='store_true', help='check if your alignment is correct')

    arguments = parser.parse_args()

    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)
    if arguments.check:
        results  = bigram_tester.logProb

        payload = json.dumps({
            'model': open(arguments.file, 'r').read(),
            'tokens': bigram_tester.tokens,
            'result': results
        })
        response = requests.post(
            'https://language-engineering.herokuapp.com/lab2_tester',
            data=payload,
            headers={'content-type': 'application/json'}
        )
        response_data = response.json()
        if response_data['correct']:
            print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))
            print('Success! Your results are correct')
        else:
            print('Your results:')
            print('Estimated entropy: {0:.2f}'.format(bigram_tester.logProb))
            print("The server's results:\n Entropy: {0:.2f}".format(response_data['result']))

    else:
        print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))

if __name__ == "__main__":
    main()
