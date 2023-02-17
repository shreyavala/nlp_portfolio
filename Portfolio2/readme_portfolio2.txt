Author: Shreya Valaboju
Date: 2/15/2023

OVERVIEW OF PORTFOLIO 2

HOW TO EXECUTE:
    * If running on Pycharm IDE:
        - Edit Configurations and add relative path of input/data text file in the parameters
    * If running on terminal/command line:
        - Type in this format to execute python file: python main.py [insert relative path of input text file]
        - Ex: python main.py anat19.txt
    * Please ensure you have all the necessary libraries installed in your python environment, i.e. nltk.
    * Ensure you input file is in the same folder as main.py

DESCRIPTION:

    This program's purpose is to showcase some basic functionality of the popular NLP library, nltk.
    This program first preprocesses and outputs information about an input text file given. For instance,
    the lexical diversity is calculated, number of tokens, lemmatization, and pos tagging, and number of nouns are all
    outputted to the console given some text as input. Next, the guessing game begins. The user is allowed to
    guess a word by typing in a letter. The default/starting score is 5. If the user enters a '!' as input
    or their score is below 0, the game ends. Score increments by 1 when a correct letter is guessed and
    decrements by 1 if the guess is incorrect.


EXTRA NOTES/ASSUMPTIONS:
    - the nltk pos tagger might return a different set of tokens each run. this is due to the randomness associated with the tagger and
        the fact that we have preprocessed the tokens.
    - the program counts the number of nouns in the text, this includes all types of nouns, i.e. personal pronouns, possessive pronouns, etc.
    - program may take a few seconds to fully execute, this is normal.