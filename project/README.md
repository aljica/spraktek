# As part of the project in DD1418.

First, install dependencies by running:

pip3 install -r requirements.txt

Then, generate the language model by running:

python3 TrigramTrainer.py -f guardian_training.txt -d model.txt

Then, run the main program by running:

python3 WordPredictor.py -f model.txt

-------------------

# How to use the program (demo of built-in commands)

At the welcome screen, input 'type' to type words and receive a list of recommended
words with each keystroke you make. Or type 'quit' to quit.

To choose the first word from the list of recommended words, input
"1-" (without the quotation marks) and press enter. Input "2-" for the second
word, etc.

Type "reset" to reset the current letters you have inputted for a word (if you change your mind).

To finish typing a word, input a space and press enter. You will now be
prompted to begin typing the next word.

To quit typing and return to the welcome screen, input "quit" (without the quotation marks).
To quit the program, input "quit" again.

Instead of running the base implementation described above, you can also generate statistics.

Type:
python3 WordPredictor.py -f model.txt -s bbc_article.txt

By supplying a text file after the '-s' flag, the program will generate statistics on how
many keystrokes you would have saved if you were to type the contents of bbc_article.txt
while using the word predictor. Make sure the file you supply after the '-s' flag is a .txt file!

------------------------

# Retrieving data to run stats on

To retrieve the BBC politics data used to generate statistics on, visit the following webpage:
http://mlg.ucd.ie/datasets/bbc.html

Under "Dataset: BBC", choose to ">> Download raw text files"
Unzip this file and cd into bbc-fulltext/bbc/politics/
Now, type:
cat * > politics.txt

Now, you may run:
python3 WordPredictor.py -f model.txt -s politics.txt

And the program will generate statistics using the politics.txt file.

--------------------

To retrieve the SMS data used to generate statistics on, visit the following webpage:
http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

Scroll down slightly to the table, and click on "Link 1". smsspamcollection.zip will now be downloaded.

Unzip and cd into the new folder. Move SMSSpamCollection.txt to the project's working directory. Then, run:
python3 cleanse_sms.py
and a file "sms.txt" will be outputted in your current working directory.

Now, you may run:
python3 WordPredictor.py -f model.txt -s sms.txt

And the program will generate statistics using the sms.txt file.

--------------------

To generate stats on the spelling correction algorithm, do the following:

Visit https://www.dcs.bbk.ac.uk/~ROGER/corpora.html and download birkbeck.dat
(will be saved as missp.dat on your computer).

Then, run the following:
python3 spell_check_stats.py

Make sure that model.txt and missp.dat are in the same directory as spell_check_stats.py!

The program will now, for each misspelled word, try to correct it by running the algorithm and
checking if any of the words exist in the vocabulary. A prediction window size of 3 is used.
