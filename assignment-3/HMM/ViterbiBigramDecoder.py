from Key import Key
import math
import sys
import numpy as np
import codecs
import argparse
import json
import requests

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class ViterbiBigramDecoder(object):
    """
    This class implements Viterbi decoding using bigram probabilities in order
    to correct keystroke errors.
    """
    def init_a(self, filename):
        """
        Reads the bigram probabilities (the 'A' matrix) from a file.
        """
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                i, j, d = [func(x) for func, x in zip([int, int, float], line.strip().split(' '))]
                self.a[i][j] = d


    # ------------------------------------------------------


    def init_b(self):
        """
        Initializes the observation probabilities (the 'B' matrix).
        """
        for i in range(Key.NUMBER_OF_CHARS):
            cs = Key.neighbour[i]

            # Initialize all log-probabilities to some small value.
            for j in range(Key.NUMBER_OF_CHARS):
                self.b[i][j] = -float("inf")

            # All neighbouring keys are assigned the probability 0.1
            for j in range(len(cs)):
                self.b[i][Key.char_to_index(cs[j])] = math.log( 0.1 )

            # The remainder of the probability mass is given to the correct key.
            self.b[i][i] = math.log((10 - len(cs))/10.0)


    # ------------------------------------------------------



    def viterbi(self, s):
        """
        Performs the Viterbi decoding and returns the most likely
        string.
        """
        # First turn chars to integers, so that 'a' is represented by 0,
        # 'b' by 1, and so on.
        index = [Key.char_to_index(x) for x in s]

        # The Viterbi matrices
        self.v = np.zeros((len(s), Key.NUMBER_OF_CHARS))
        self.v[:,:] = -float("inf")
        self.backptr = np.zeros((len(s) + 1, Key.NUMBER_OF_CHARS), dtype='int')

        # Initialization
        self.backptr[0,:] = Key.START_END
        self.v[0,:] = self.a[Key.START_END,:] + self.b[index[0],:]

        # Induction step

        for t in range(1, len(index)):

            o_t = index[t] # Emission/observation (from mistyped_*)
            q_k_possible = Key.neighbour[o_t] # All possible q_k, excluding o_t
            for k in range(len(q_k_possible) + 1):
                if k < len(q_k_possible):
                    # If keystroke was mistake
                    q_k = Key.char_to_index(q_k_possible[k])
                else:
                    # If keystroke was intended, i.e. q_k = o_t
                    q_k = o_t

                for i in range(len(self.v[t-1])):
                    # Looping through alphabet
                    p = self.v[t-1][i] # Probability
                    if p == -float("inf"):
                        continue
                    q_i = i # Alphabetic index of the q_i from time state (t-1)

                    transitional_p = self.a[q_i][q_k] # Transitional probability P(q_k|q_i)
                    observational_p = self.b[o_t][q_k] # Given the observation of o_t, P(q_k)?
                    p = p + transitional_p + observational_p # Do the calculation

                    p_current = self.v[t][q_k] # Currently stored probability for that q_k

                    if p > p_current:
                        # If the newly calculated path is better than the already stored one
                        self.v[t][q_k] = p
                        self.backptr[t][q_k] = q_i # Pointing back at the best path

        # Finally return the result
        string = ''

        next = self.backptr[len(self.v) - 1][26] # Always start with space
        for i in range(len(self.v) - 2, 0, -1):
            next = self.backptr[i][next]
            string = Key.index_to_char(next) + string

        # REPLACE THE LINE BELOW WITH YOUR CODE
        return string


    # ------------------------------------------------------



    def __init__(self, filename=None):
        """
        Constructor: Initializes the A and B matrices.
        """
        # The trellis used for Viterbi decoding. The first index is the time step.
        self.v = None

        # The bigram stats.
        self.a = np.zeros((Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS))

        # The observation matrix.
        self.b = np.zeros((Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS))

        # Pointers to retrieve the topmost hypothesis.
        backptr = None

        if filename: self.init_a(filename)
        self.init_b()



    # ------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(description='ViterbiBigramDecoder')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', type=str, help='decode the contents of a file')
    group.add_argument('--string', '-s', type=str, help='decode a string')
    parser.add_argument('--probs', '-p', type=str,  required=True, help='bigram probabilities file')
    parser.add_argument('--check', action='store_true', help='check if your answer is correct')

    arguments = parser.parse_args()

    if arguments.file:
        with codecs.open(arguments.file, 'r', 'utf-8') as f:
            s1 = f.read().replace('\n', '')
    elif arguments.string:
        s1 = arguments.string

    # Give the filename of the bigram probabilities as a command line argument
    d = ViterbiBigramDecoder(arguments.probs)

    # Append an extra "END" symbol to the input string, to indicate end of sentence.
    result = d.viterbi(s1 + Key.index_to_char(Key.START_END))

    if arguments.check:
        payload = json.dumps({
            'a': d.a.tolist(),
            'string': s1,
            'result': result
        })
        response = requests.post(
            'https://language-engineering.herokuapp.com/lab3_bigram',
            data=payload,
            headers={'content-type': 'application/json'}
        )
        response_data = response.json()
        if response_data['correct']:
            print(result)
            print('Success! Your results are correct')
        else:
            print('Your results:')
            print(result)
            print('Your answer is {0:.0f}% similar to that of the server'.format(response_data['result'] * 100))
    else:
        print(result)


if __name__ == "__main__":
    main()
