import math
import argparse
import codecs
from collections import defaultdict
import random
import numpy # For selecting words based on their probabilities

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):

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


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                # REUSE YOUR CODE FROM BigramTester.py here
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

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and following the distribution
        of the language model.
        """
        # YOUR CODE HERE
        words_to_print = []
        words_to_print.append(w) # First word (user-inputted one) added

        for i in range(n):
            words = []
            probabilities = []
            most_recent_word = words_to_print[len(words_to_print) - 1] # Index of most recently used word (starts with starting word)
            subsequent_words = self.bigram_prob[most_recent_word] # All words that ever appear after the first word, and the probabilities
            for next_word in subsequent_words:
                # For each word that follows
                words.append(next_word) # The word
                log_prob = subsequent_words[next_word] # Get that bigram's log probability
                #regular_prob = math.exp(log_prob) # The bigram's (most_recent_word + next_word) probability (NOT expressed as a log)
                #regular_prob = float("{0:.15f}".format(regular_prob))
                probabilities.append(math.exp(log_prob))
            if words:
                # If the bigram was also in our language model
                ## Received error "probabilities do not sum to one" from numpy. Fixed it by following link below:
                ## https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
                p = numpy.array(probabilities)
                p /= p.sum() # Normalize
                chosen_word = numpy.random.choice(words, 1, p=p) # Choose a word
                words_to_print.append(chosen_word[0])
            else:
                # If there were no bigrams, then
                # we must choose another 'starting' word
                random_index = random.randrange(0, self.unique_words, 1)
                new_word = self.index[random_index]
                words_to_print.append(new_word)

        for word in words_to_print:
            print(word + ' ')

def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
