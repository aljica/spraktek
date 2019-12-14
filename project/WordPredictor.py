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
    def __init__(self, filename, stats_file = None):

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
        self.num_words_to_recommend = 3 # Also called the prediction window size.

        if not self.read_model(filename):
            # If unable to read model (file missing?).
            print("Unable to read model, was the filepath correctly specified?")
            sys.exit()

        if stats_file:
            self.stats(stats_file)
            sys.exit()

        self.welcome()

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
            print("Please enter 'type' to type freely and see a list of recommended words with each keystroke you make.")
            print("Enter 'quit' to quit.")
            user_input = input("Your choice: ")
            if user_input == "type":
                self.run_type()
            else:
                if user_input != "quit":
                    print("\nPlease input 'type' to type words or 'quit' (without the quotation marks).")

    def run_type(self):
        while True:
            if (self.type_word()):
                print("\nExiting type.")
                break
        self.words = [] # Reset

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



    def get_n_grams(self, prev_word = None, two_words_back = None, user_input = ""):
        """
        Returns either bigram probabilities given historical word prev_word or
        trigram probabilities given historical words two_words_back & prev_word.
        If no historical words are specified, then unigram counts are returned.

        Based on user_input for current word being inputted.
        """
        if prev_word and two_words_back:
            w = self.trigram_prob.get(two_words_back, "empty")
            if w != "empty":
                w = w.get(prev_word, "empty")
                if w != "empty":
                    words_and_p = [(w, p) for w, p in w.items()]
                    words_and_p.sort(key=itemgetter(1), reverse=True) # Sorted from highest to lowest probability.
                    possible_words = [w for (w, p) in words_and_p if w[:len(user_input)] == user_input] # Only the words that start with user_input.
                    return possible_words
        elif prev_word and not two_words_back:
            w = self.bigram_prob.get(prev_word, "empty")
            if w != "empty":
                words_and_p = [(w, p) for w, p in w.items()]
                words_and_p.sort(key=itemgetter(1), reverse=True) # Sorted from highest to lowest probability.
                possible_words = [w for (w, p) in words_and_p if w[:len(user_input)] == user_input] # Only the words that start with user_input.
                return possible_words
        else:
            words_and_p = [(w, p) for w, p in self.unigram_count.items()]
            words_and_p.sort(key=itemgetter(1), reverse=True) # Sorted from highest to lowest probability.
            possible_words = [w for (w, p) in words_and_p if w[:len(user_input)] == user_input] # Only the words that start with user_input.
            return possible_words

        return []

    def recommend_words(self, prev_word = None, two_words_back = None, user_input = "", possible_words = None):
        """
        Recommends possible words using self.get_n_grams().

        If user specifies possible_words as param, then the list is filtered to remove words that don't start with user_input.
        """
        if possible_words:
            return [w for w in possible_words if w[:len(user_input)] == user_input]
        return self.get_n_grams(prev_word, two_words_back, user_input)

    def type_letter(self, possible_choices):
        """
        Prompts user for letter inputs.

        :param possible_choices is a list of ["1-", "2-", ...] all recommended words user can choose.
        """
        while True:
            letter = input("Enter a character (or choose a recommended word): ")
            if len(letter) == 1 or letter in ["quit", "reset", " "] + possible_choices:
                return letter
            print("\nPlease input a character. You can also type 'quit' to quit the program, 'reset' to reset the word or input a blank space to finish typing your word.")
        pass

    def type_word(self):
        """
        Handles user inputs.
        """
        letter = ""
        new_word = ""

        check_unigram = True # Flag variable to check unigrams if possible_words is empty.

        if len(self.words) == 0:
            # If the user hasn't written any words yet.
            possible_words = self.recommend_words(prev_word = ".") # Get start-of-sentence probabilities (bigrams).
        elif len(self.words) == 1:
            possible_words = self.recommend_words(prev_word = str(self.words[len(self.words) - 1])) # Get bigram probabilities.
        else:
            possible_words = self.recommend_words(prev_word = str(self.words[len(self.words) - 1]), two_words_back = str(self.words[len(self.words) - 2])) # Get trigram probabilities.
            possible_words += self.recommend_words(prev_word = str(self.words[len(self.words) - 1]))

        while letter != " ":
            self.print_console(self.words, new_word)
            #words_to_recommend = possible_words[:self.num_words_to_recommend]
            words_to_recommend = []
            i = 0
            while len(words_to_recommend) < self.num_words_to_recommend and len(possible_words) != 0:
                # This is to ensure we always have 3 *distinct* words that are being recommended.
                try:
                    word = possible_words[i]
                    if word in words_to_recommend:
                        i += 1
                        continue
                    else:
                        words_to_recommend.append(word)
                except IndexError:
                    break

            if len(words_to_recommend) < self.num_words_to_recommend:
                if check_unigram:
                    # If there are no recommended bigrams, check unigrams.
                    possible_words += self.recommend_words(user_input = new_word)
                    check_unigram = False
                    continue
                if len(words_to_recommend) == 0:
                    # Then, we know user either misspelled the word or wishes to add a new one we haven't heard of before.
                    if new_word.isalpha():
                        # Only try to correct spelling if the word user is typing does not contain a non-alphabetic character.
                        words_to_recommend = self.spell_check(new_word)

            for i in range(len(words_to_recommend)):
                print(i+1, "-", words_to_recommend[i])

            possible_choices = [(str(i) + "-") for i in range(1, len(words_to_recommend) + 1)]
            letter = self.type_letter(possible_choices)

            if letter in possible_choices:
                number_of_word = possible_choices.index(letter)
                chosen_word = words_to_recommend[number_of_word]
                self.unigram_count[chosen_word] += 1
                self.words.append(chosen_word)
                break

            if letter == "quit":
                return True

            if letter == "reset":
                break

            if letter == " ":
                if new_word == "":
                    break

                # Add new words. Also update their unigram count.
                if new_word not in self.index:
                    self.index[new_word] = len(self.index)
                    self.word[len(self.index)] = new_word
                    self.unigram_count[new_word] = 1
                    self.words.append(new_word)
                    break

                self.words.append(new_word)
                self.unigram_count[new_word] += 1
                break

            new_word += letter

            possible_words = self.recommend_words(user_input = new_word, possible_words = possible_words)

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
            if n%100 == 0:
                print("\nStats generated on", n, "words from the test file")
                print("Total keystrokes in test file thus far", self.total_keystrokes, "user had to type", self.user_keystrokes)
                print("User had to make", 100 * self.user_keystrokes / self.total_keystrokes, "percent of the keystrokes.")

            self.total_keystrokes += len(token) + 1 # Add the number of keystrokes required to type out the word. Plus 1 for the space before the next token.
            user_input = ""

            if len(self.words) == 0:
                # If the user hasn't written any words yet.
                possible_words = self.recommend_words(prev_word = ".") # Get start-of-sentence probabilities (bigrams).
            elif len(self.words) == 1:
                possible_words = self.recommend_words(prev_word = str(self.words[len(self.words) - 1])) # Get bigram probabilities.
            else:
                possible_words = self.recommend_words(prev_word = str(self.words[len(self.words) - 1]), two_words_back = str(self.words[len(self.words) - 2])) # Get trigram probabilities.
                possible_words += self.recommend_words(prev_word = str(self.words[len(self.words) - 1]))

            if token in possible_words:
                token_recommendation_rank = possible_words.index(token) + 1 # If rank would be recommended first or second etc (depending on probability).
                if token_recommendation_rank <= self.num_words_to_recommend:
                    # Then we can choose the word right away.
                    self.user_keystrokes += 1 # User keystroke for choosing the recommendation.
                    self.unigram_count[token] += 1 # Update unigram count.
                    self.words.append(token)
                    continue
            else:
                possible_words += self.recommend_words() # Add all unigrams.
                if token in possible_words:
                    token_recommendation_rank = possible_words.index(token) + 1 # If rank would be recommended first or second etc (depending on probability).
                    if token_recommendation_rank <= self.num_words_to_recommend:
                        # Then we can choose the word right away.
                        self.user_keystrokes += 1 # User keystroke for choosing the recommendation.
                        self.unigram_count[token] += 1
                        self.words.append(token)
                        continue
                else:
                    # Then, the token being typed is a word not in our vocabulary, i.e. we have not seen it before. User has to type the whole thing out!
                    # Here, we should use the spell_check algorithm, but decided not to in order to make the code more effective!
                    self.user_keystrokes += len(token) + 1 # To type out the word and add a space (+1).
                    # Add the word to our vocabulary.
                    self.index[token] = len(self.index)
                    self.word[len(self.index)] = token
                    self.unigram_count[token] = 1
                    self.words.append(token)
                    continue

            for letter in token:
                user_input += letter
                self.user_keystrokes += 1

                possible_words = self.recommend_words(user_input = user_input, possible_words = possible_words) # Update possible words based on user_input.

                token_recommendation_rank = possible_words.index(token) + 1
                if token_recommendation_rank <= self.num_words_to_recommend:
                    self.user_keystrokes += 1 # For choosing the recommendation.
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
    parser.add_argument('--stats', '-s', type=str, required=False, help='input a test file to run statistics on (how many keystrokes you would have saved)')

    arguments = parser.parse_args()

    word_predictor = WordPredictor(arguments.file, arguments.stats)

if __name__ == "__main__":
    main()
