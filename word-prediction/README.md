First, install dependencies by running:

pip3 install -r requirements.txt

Then, generate the language model by running:

python3 TrigramTrainer.py -f guardian_training.txt -d model.txt

Then, run the main program by running:

python3 WordPredictor.py -f model.txt

-------------------

Type your word by inputting one letter at a time and pressing enter.
You will see your letter appear in the dialogue window above the list of
recommended words.

To finish typing a word, input a space and press enter. You will now be
prompted to begin typing the next word.

To choose the first word from the list of recommended words, input
"1-" (without the quotation marks) and press enter. Input "2-" for the second
word, etc.

To quit the program, input "quit" (without the quotation marks).
The program will now announce that it is "EXITING".

nltk.download('brown') is required for running the stats.
