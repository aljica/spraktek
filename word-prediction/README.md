# As part of the project in DD1418.

First, install dependencies by running:

pip3 install -r requirements.txt

Then, generate the language model by running:

python3 TrigramTrainer.py -f guardian_training.txt -d model.txt

Then, run the main program by running:

python3 WordPredictor.py -f model.txt

-------------------

# How to use the program (demo of built-in commands)

At the welcome screen, input either 'stats' or 'type'. Type 'quit' to quit.

One of the options is 'stats'. Choosing the stats option prompts you to firstly
input the path to a test file. The test file should contain, for instance,
an article from a news paper. It might be that you just copy-pasted an article
from BBC into your test file. Make sure your test file is a .txt file!

Stats will use the Word Predictor to type each word in your test file and
check how many keystrokes you would have saved.

You can also choose the option 'type' at the welcome screen.

This lets you input a word letter by letter, and with each keystroke a list of
recommended completion words will be updated. You will see each letter you type
appear in the dialogue window above the list of recommended words.

To finish typing a word, input a space and press enter. You will now be
prompted to begin typing the next word.

To choose the first word from the list of recommended words, input
"1-" (without the quotation marks) and press enter. Input "2-" for the second
word, etc.

To quit typing and return to the welcome screen, input "quit" (without the quotation marks).
