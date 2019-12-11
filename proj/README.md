First, install dependencies by running:

pip3 install -r requirements.txt

Then, generate the language model by running:

python3 TrigramTrainer.py -f guardian_training.txt -d model.txt

Then, run the main program by running:

python3 WordPredictor.py -f model.txt
