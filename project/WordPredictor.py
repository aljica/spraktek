import argparse
import codecs
from collections import defaultdict
from operator import itemgetter
import nltk
import sys

class WordPredictor:
    """
    This class predicts words using a language model.
    """
    def __init__(self, filename):

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

        # Number of words to recommend to the user. Keep this number reasonable, <10.
        self.num_words_to_recommend = 3

        if not self.read_model(filename):
            # If unable to read model (file missing?).
            print("Unable to read model, was the filepath correctly specified?")
            sys.exit()

        self.welcome()

    def test(self):
        w = self.trigram_prob.get(self.words[0], "empty")
        if w != "empty":
            w = w.get(self.words[1], "empty")
            if w != "empty":
                words_and_p = [(w, p) for w, p in w.items()]

        print(words_and_p)
        print("Sorting")
        words_and_p.sort(key=itemgetter(1), reverse=True)
        print(words_and_p)

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

                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def welcome(self):
        print("Welcome to the Word Prediction Program.")
        user_input = ""
        while user_input != "quit":
            print("\nPlease enter 'stats' to check how many keystrokes you would save if you were to type out the contents of a test file.")
            print("Please enter 'type' to type freely and see a list of recommended words with each keystroke you make.")
            print("Enter 'quit' to quit.")
            user_input = input("Your choice: ")
            if user_input == "stats":
                self.run_stats()
            elif user_input == "type":
                self.run_type()
            else:
                if user_input != "quit":
                    print("\nPlease input 'stats' or 'type' (without the quotation marks).")

    def run_type(self):
        while True:
            if (self.type_word()):
                print("\nExiting type.")
                break
        self.words = [] # Reset

    def run_stats(self):
        filepath = ""
        while filepath != "quit":
            filepath = input("\nInput a filepath to a text file to run stats (or type quit): ")
            self.stats(filepath)

    def print_console(self, words, new_word):
        """
        Prints the console.
        """
        print("\n")
        all_words = ""
        for word in words:
            if word in [".", ",", "!", "?"]:
                all_words = all_words.strip() # Remove last whitespace.
                all_words += word + " "
                continue
            all_words += word + " "
        all_words += new_word + "_"
        print(all_words)

    def edits1(self, word):
        """
        All edits that are one edit away from the given word.
        Source: https://norvig.com/spell-correct.html
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """
        All edits that are two edits away from the given word.
        Source: https://norvig.com/spell-correct.html
        """
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def known(self, words):
        """
        The permutations of one or two edits from the misspelled word that constitute real words, found in our vocabulary.
        Source: https://norvig.com/spell-correct.html
        """
        return set((w, self.unigram_count[w]) for w in words if w in self.index)

    def spell_check(self, word):
        """
        Finds possible corrections of misspelled words.
        """
        possible_words = self.known(self.edits2(word))
        if len(possible_words) == 0: return []
        most_frequently_used_words = []

        for i in range(self.num_words_to_recommend):
            word = max(possible_words, key=itemgetter(1))[0] # See https://stackoverflow.com/questions/13145368/find-the-maximum-value-in-a-list-of-tuples-in-python
            most_frequently_used_words.append(word)
            possible_words.remove((word, self.unigram_count[word])) # So that we do not choose it again in the next iteration.
            if len(possible_words) == 0:
                break

        return most_frequently_used_words



    def get_n_grams(self, prev_word = None, two_words_back = None):
        """
        Returns either bigram probabilities given historical word prev_word or
        trigram probabilities given historical words two_words_back & prev_word.
        """
        if two_words_back:
            w = self.trigram_prob.get(two_words_back, "empty")
            if w != "empty":
                w = w.get(prev_word, "empty")
                if w != "empty":
                    words_and_p = [(w, p) for w, p in w.items()]
                    words_and_p.sort(key=itemgetter(1), reverse=True) # Sorted from highest to lowest probability.
                    return words_and_p
        else:
            w = self.bigram_prob.get(prev_word, "empty")
            if w != "empty":
                words_and_p = [(w, p) for w, p in w.items()]
                words_and_p.sort(key=itemgetter(1), reverse=True) # Sorted from highest to lowest probability.
                return words_and_p

        return []

    def type_word(self):
        """
        Handles user inputs.
        """
        letter = ""
        new_word = ""

        if len(self.words) == 0:
            # If the user hasn't written any words yet.
            possible_trigrams = None
            possible_bigrams = self.get_n_grams(prev_word = ".") # Get start-of-sentence probabilities (bigrams).
        elif len(self.words) == 1:
            possible_trigrams = None
            possible_bigrams = self.get_n_grams(str(self.words[len(self.words) - 1])) # Get bigram probabilities.
        else:
            possible_trigrams = self.get_n_grams(str(self.words[len(self.words) - 1]), str(self.words[len(self.words) - 2])) # Get trigram probabilities.
            possible_bigrams = self.get_n_grams(str(self.words[len(self.words) - 1])) # Get bigram probabilities.

        while letter != " ":
            self.print_console(self.words, new_word)
            recommended_words = self.rec_words(new_word)

            if len(recommended_words) == 0:
                # If there are no recommended words, check if user has misspelled the word.
                # BUT, if there is a non-alphabetic character in new_word, then do NOT check spelling.
                # This is because if the user has written "$1" and it doesn't exist in the vocabulary,
                # then the system might end up recommending any two-letter word "to", "am", ... etc.
                check_spell = True
                for c in new_word:
                    if not c.isalpha():
                        check_spell = False
                        break
                if check_spell:
                    recommended_words = self.spell_check(new_word)

            for i in range(len(recommended_words)):
                print(i+1, "-", recommended_words[i])

            possible_choices = [(str(x) + "-") for x in range(1, len(recommended_words) + 1)]

            letter = input("Enter your letter (or choice): ")

            # Here we allow the user to choose one of the words from the list of recommended words.
            if letter in possible_choices:
                number_of_word = possible_choices.index(letter)
                chosen_word = recommended_words[number_of_word]
                self.words.append(chosen_word)
                self.unigram_count[chosen_word] += 1 # Update unigram count.
                break

            # Handle cases for basic functionality.
            if letter == "quit":
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

                if new_word not in self.index:
                    # If the word user typed does not exist in our vocabulary.
                    self.index[new_word] = len(self.index)
                    self.word[len(self.index)] = new_word
                    self.unigram_count[new_word] = 1

                self.words.append(new_word)
                self.unigram_count[new_word] += 1 # Update unigram count (keeps track of how frequently user uses this word).

            new_word += letter

        return False


    def stats(self, filepath):
        """
        Determines number of saved keystrokes given an input file.
        """
        self.total_keystrokes = 0 # Number of total keystrokes required for the entire file.
        self.user_keystrokes = 0 # Number of keystrokes user had to type.
        try:
            with open(filepath, 'r') as f:
                text = str(f.read())
                try:
                    self.tokens = nltk.word_tokenize(text)
                except LookupError:
                    nltk.download('punkt')
                    self.tokens = nltk.word_tokenize(text)
        except FileNotFoundError:
            print("File does not exist.")
            return

        print("Number of words/tokens in test file", len(self.tokens))
        n = 0 # Number of analyzed tokens from test file thus far.
        for token in self.tokens:
            if token == "" or token == " ":
                # If somehow a token is just blank, skip it.
                continue

            n += 1
            if n%1000 == 0:
                print("\nStats generated on", n, "words from the test file")
                print("Total keystrokes in test file thus far", self.total_keystrokes, "user had to type", self.user_keystrokes)
                print("User had to make", 100 * self.user_keystrokes / self.total_keystrokes, "percent of the keystrokes.")

            self.total_keystrokes += len(token) + 1 # Add the number of keystrokes required to type out the word. Plus 1 for the space before the next token.
            user_input = ""
            recommended_words = self.rec_words(user_input)
            if token in recommended_words:
                self.user_keystrokes += 1 # Add 1 keystroke as user selects the recommendation.
                self.words.append(token)
                continue

            # Now, for each letter that has to be typed:
            for letter in token:
                user_input += letter
                self.user_keystrokes += 1 # Increment number of keystrokes user has had to make.

                if user_input == token:
                    self.user_keystrokes += 1 # Add 1 for the space user would have to add.
                    self.words.append(token)
                    break

                recommended_words = self.rec_words(user_input)
                if token in recommended_words:
                    self.user_keystrokes += 1 # Add 1 keystroke as user selects the recommendation.
                    self.words.append(token)
                    break

        print("\nFinal information, based on entire test file:")
        print("Total words in test file", n, "- Total keystrokes in test file", self.total_keystrokes, "user had to type", self.user_keystrokes)
        print("User had to make", 100 * self.user_keystrokes / self.total_keystrokes, "percent of the keystrokes.")
        self.words = [] # Reset


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Word Predictor')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')

    arguments = parser.parse_args()

    word_predictor = WordPredictor(arguments.file)

if __name__ == "__main__":
    main()
