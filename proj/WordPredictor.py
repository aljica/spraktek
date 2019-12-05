import argparse
import codecs
from collections import defaultdict

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.

Modified December 2019 by Almir Aljic.
"""

class WordPredictor:
    """
    This class predicts words using a language model.
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

        # The trigram log-probabilities.
        nested_dict = lambda: defaultdict(nested_dict)
        self.trigram_prob = nested_dict()

        # Number of unique words in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # User-inputted words.
        self.words = []

        # Number of words to recommend to the user.
        self.num_words_to_recommend = 3


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                for i in range(self.unique_words):
                    _, word, frequency = map(str, f.readline().strip().split(' '))
                    self.word[i], self.index[word], self.unigram_count[word] = word, i, int(frequency)
                # Read all bigram probabilities.
                for line in f:
                    if line.strip() == "-2":
                        break
                    # Get index of first word and second word respectively, and their bigram prob
                    i, j, prob = map(str, line.strip().split(' '))
                    first_word, second_word = self.word[int(i)], self.word[int(j)]
                    self.bigram_prob[first_word][second_word] = float(prob)
                # Read all trigram probabilities.
                for line in f:
                    if line.strip() == "-1":
                        break
                    i, j, k, p = map(str, line.strip().split(' '))
                    first_word, second_word, third_word = self.word[int(i)], self.word[int(j)], self.word[int(k)]
                    self.trigram_prob[first_word][second_word][third_word] = float(p)

                x=list(self.trigram_prob["are"]["you"])
                y=list(self.trigram_prob["are"]["you"].values())

                print(x)
                print(y)

                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def print_console(self, words, new_word):
        """
        Prints the console.
        """
        print("\n")
        print(words)
        all_words = ""
        for word in words:
            all_words += word + " "
        all_words += new_word + "_"
        print(all_words)

    def get_subsequent_bigram_words(self):
        """
        Fetches all possible subsequent words that are found as part of
        a bigram from the language model.
        """

        if len(self.words) > 0:
            prev_word = self.words[len(self.words) - 1]
            w = self.bigram_prob.get(prev_word, "empty")
            if w != "empty":
                words = list(w)
                p = list(w.values())
                return words, p

        return [], []

    def get_subsequent_trigram_words(self):
        """
        Fetches all possible subsequent words as part of a trigram, given by
        the language model.
        """

        if len(self.words) > 1:
            sub_two_word = self.words[len(self.words) - 2]
            prev_word = self.words[len(self.words) - 1]
            w = self.trigram_prob.get(sub_two_word, "empty")
            if w != "empty":
                w = w.get(prev_word, "empty")
                if w != "empty":
                    words = list(w)
                    p = list(w.values())
                    return words, p

        return [], []

    def top_n_gram_words(self, user_input, n_gram):
        """
        Determines int(self.num_words_to_recommend) words that are most likely to appear after a given
        n-gram, depending on user input. All possible words that can follow
        the n-gram is given by subsequent_words, which are taken from
        self.trigram_prob or self.bigram_prob depending on the n-gram.

        The number n is based on self.num_words_to_recommend, and is given in
        the constructor of this class.
        """
        if n_gram == 3:
            words, p = self.get_subsequent_trigram_words()
        elif n_gram == 2:
            words, p = self.get_subsequent_bigram_words()
        else:
            return []

        if len(words) == 0 and len(p) == 0:
            return []

        if user_input != "":
            words_user_input = []
            p_user_input = []
            for i in range(len(words)):
                word = words[i]
                if len(user_input) <= len(word):
                    test_word = word[0:len(user_input)]
                    if user_input == test_word:
                        words_user_input.append(word)
                        p_user_input.append(p[i])

            words = words_user_input
            p = p_user_input
            if len(words) == 0 and len(p) == 0:
                return []

        recommended_words = []
        for i in range(self.num_words_to_recommend):
            m = max(p)
            indices_of_max_p = [i for i, j in enumerate(p) if j == m]

            if len(indices_of_max_p) > 1:
                chosen_word = words[indices_of_max_p[0]] # Choose first word
                chosen_word_count = self.unigram_count[chosen_word]
                for j in range(1, len(indices_of_max_p)):
                    word = words[indices_of_max_p[j]]
                    count = self.unigram_count[word]

                    if count > chosen_word_count:
                        chosen_word_count = count
                        chosen_word = word
                recommended_words.append(chosen_word)
            else:
                recommended_words.append(words[indices_of_max_p[0]])

            word_just_added = recommended_words[i]
            idx = words.index(word_just_added)
            words.pop(idx)
            p.pop(idx)
            if len(words) == 0 and len(p) == 0:
                break

        return recommended_words

    def top_unigram_words(self, user_input):
        words = []
        counts = []
        for w in self.index:
            if user_input in w:
                words.append(w)
                counts.append(self.unigram_count[w])
        if len(words) == 0 and len(counts) == 0:
            return []

        words_to_return = []
        for i in range(self.num_words_to_recommend):
            index_of_word_with_max_count = counts.index(max(counts))
            word = words[index_of_word_with_max_count]
            words_to_return.append(word)

            idx_to_remove = words.index(word)
            words.pop(idx_to_remove)
            counts.pop(idx_to_remove)

            if len(words) == 0 and len(counts) == 0:
                break

        return words_to_return

    def rec_words(self, user_input):
        """
        Checks for trigram and bigram probabilities. If none are found,
        a search of highest unigram count is returned.
        Returns a list of words to recommend (size of list is determined by
        self.num_words_to_recommend)
        """
        recommended_words = []

        trigrams = self.top_n_gram_words(user_input, 3) # Most probable words to appear in the context of the given trigram, given a user input.
        bigrams = self.top_n_gram_words(user_input, 2)
        unigrams = self.top_unigram_words(user_input)

        recommended_words += [trigrams[x] for x in range(len(trigrams))]
        recommended_words += [bigrams[x] for x in range(len(bigrams))]
        recommended_words += [unigrams[x] for x in range(len(unigrams))]

        words_to_recommend = []
        for word in recommended_words:
            if word in words_to_recommend:
                continue
            else:
                words_to_recommend.append(word)

        if len(words_to_recommend) >= self.num_words_to_recommend:
            return words_to_recommend[0:3]

        return words_to_recommend


    def type(self):
        letter = ""
        new_word = ""

        while letter != " ":

            self.print_console(self.words, new_word)
            recommended_words = self.rec_words(new_word)

            for i in range(len(recommended_words)):
                print(i+1, "-", recommended_words[i])

            possible_choices = [(str(x) + "-") for x in range(1, len(recommended_words) + 1)]

            letter = input("Enter your letter (or choice): ")

            # Here we allow the user to choose one of the words from the list of recommended words.
            if letter in possible_choices:
                number_of_word = possible_choices.index(letter)
                chosen_word = recommended_words[number_of_word]
                self.words.append(chosen_word)
                break

            # Handle cases for basic functionality.
            if letter == "quit-this":
                return True

            if letter == "re-":
                # Wipes the current letters user has typed for an uncompleted word.
                break

            if len(letter) > 1:
                print("\n###---Please enter one letter at a time!!---###\n")
                continue

            if letter == "":
                continue

            if letter == " ":
                # Here we have to determine when to add a new word.
                if new_word == "":
                    break
                self.words.append(new_word)

            new_word += letter

        return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Word Predictor')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')

    arguments = parser.parse_args()

    word_predictor = WordPredictor()
    word_predictor.read_model(arguments.file)

    while True:
        if(word_predictor.type()):
            print("\nEXITING!")
            break

if __name__ == "__main__":
    main()
