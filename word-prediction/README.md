# As part of the project in DD1418.

First, install dependencies by running:

pip3 install -r requirements.txt

Then, open another terminal and type "python3" (run the python console). Type:
nltk.download('brown')
to download the brown corpus from nltk's webpage.

Then, generate the language model by running:

python3 TrigramTrainer.py -f guardian_training.txt -d model.txt

Then, run the main program by running:

python3 WordPredictor.py -f model.txt

You may also specify a test file to run stats on by typing:

python3 WordPredictor.py -f model.txt -tf your_test_file.txt

If you subsequently choose to run stats in the welcome window,
stats will automatically be run on your test file.
Once again, the "-tf" option is optional and we do not recommend using it because
regardless, you are prompted to specify the file path of your test file if you choose
'stats' in the welcome window.

Your test file should contain an article you copy-pasted from a news paper website
or similar.

-------------------

# How to use the program (demo of built-in commands)

At the welcome screen, input either 'stats' or 'type'. Type 'quit' to quit.

If you type 'stats', and if you specified a test file using the -tf option,
stats will automatically be run using your test file. Stats means the program
will use the Word Predictor to type each word in your test file and 
check how many keystrokes you would have saved.

Otherwise, if you didn't specify a test file, you will be prompted
to run stats on the Brown Corpus. Type "yes" to confirm (beware! This takes a *very* long time).

Typing "no" will let you specify a file path to your test file, on which stats will be run.

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
