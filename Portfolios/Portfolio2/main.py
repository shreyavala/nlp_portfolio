# Name: Shreya Valaboju
# Course/Section: CS 4395.001
# Notes: Refer to file named "readme_portfolio2.txt" on instructions on how to run, extra notes, and an overview/description
# Github Link: https://github.com/shreyavala/nlp_portfolio





# import necessary libraries
import sys
import pathlib
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from random import seed
from random import randint
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def guessingGame(text):
    """
       starts the guessing game for the user after preprocessing, creating a dictionary

        Parameters:
        text (string): raw text from input file
    """

    # begin by reading in the text file, preprocessing, and creating a dictionary of nouns to use in the guessing game
    sorted_nouns = createDictionary(text)
    score = 5 # start user score at 5
    guessed_letters=[]

    print("-------------------------------------------------------------------------------------------------------------")
    print("Let's play a word guessing game!")
    print("Rules: ")
    print("- guess a word by guessing a letter each try")
    print("- Default/starting score is 5")
    print("- if a letter is in the word, score increases by 1, else score decrements by 1")
    print("- if score is less than 0, game ends")
    print("- enter '!' to quit the game")

    # choose a random word from list of nouns
    word=getRandomWord(sorted_nouns)

    # get user input, print out blank
    blank = resetBlanks(word)
    user_input = input("Guess a Letter: ")

    while user_input != "!" and score >=0: #game ends with cummulative score < 0 or user enters ! to quit

        if user_input in word:    # if letter is in word, increase score by 1, guess is correct

            if user_input not in guessed_letters: # keep track of letters already guessed, increment score only if new letter is guessed
                score += 1
                guessed_letters.append(user_input)
            b=0
            for w in word:
                if w == user_input: # replace blank/empty underscore with the letter
                    blank[b] = w
                b+=1
            print("Right! Score is ",score)

        else: # guess is incorrect
            score-=1
            print("Sorry, guess again. Score is ",score)

        print("\n")
        print(''.join(blank))

        if "_" not in blank:    # user correctly guessed entire word
            print("You solved it!")
            print("Current Score: ", score)
            print("\nGuess another word")
            word = getRandomWord(sorted_nouns) # get a new word
            blank = resetBlanks(word) # reset blank for word
            user_input = input("Guess a Letter: ") # get user input again
        elif score >=0:
            user_input = input("Guess a Letter: ")  # get user input again

    # if the user enters a '!' or the score is < 0, end the game
    if user_input == "!":
        print("\nUser entered '!'")
    elif score < 0:
        print("\nYou Lose. Score < 0.")

    print("Final Score: ", score)
    print("Ending Game...")


# generates a random word from the list of nouns given as a parameter. returns the randomly generated word
def getRandomWord(nouns_list):
    rand_index = 0
    #seed(1234) #uncomment to get the same word over again
    for i in range(50):
        rand_index = randint(1, 50)
    word = nouns_list[rand_index]
    #print("Word: ", word)  # delete, for debugging purposes
    return word

# after a new word is generated, reset the 'underscore' or blanks to empty and to the size of that word
def resetBlanks(w):
    blank = []
    for s in range(len(w)):
        blank.append("_")
    print(''.join(blank))
    return blank


def preprocess(text):
    """
        preprocesses the raw text through tokenizing, lemmatization, and pos tagging

        Parameters:
        text (string): raw text from input file

        Returns:
        nouns (dict): a dictionary of nouns
        tokens (list): list of words/tokens as a result from tokenizing the input, raw text
    """

    # lowercase and tokenize the text
    text = text.lower()
    text_arr = word_tokenize(text)

    # calculate and print lexical diversity
    lex_diversity = calculateLexicalDiversity(text_arr)
    print("Lexical Diversity: %.2f" % lex_diversity)

    # not in nltk stopword list, removes punctuation (only alpha), not in nltk stopwords list
    tokens = [t for t in text_arr if t.isalpha() and t not in stopwords.words('english') and len(t) > 5]
    token_size = len(tokens)

    # lemmatize tokens, pull out unique tokens only using a set
    tokens_lemma = set(lemmatize(tokens))

    # pos tagging/printing
    nouns = posTag(tokens_lemma)
    print("\nNumber of Tokens: ", token_size) # print number of tokens before lemmatization
    print("\nNumber of Nouns: ", len(nouns))

    return nouns, tokens


def createDictionary(text):
    """
        creates a dictionary of nouns with the number of occurrences for each in the text

        Parameters:
        text (string): raw text from input file

        Returns:
        list of 50 sorted nouns based on number of occurrences
    """

    nouns, tokens = preprocess(text)
    nouns_dict={} # dictionary of nouns in the format: {nouns: count of nouns in tokens}
    nouns_dict_sorted={}

    # store count of each noun in tokens in the dictionary, nouns_dict
    for n in nouns:
        nouns_dict[n[0]]=tokens.count(n[0])

    # sort the dictionary by count
    for pos in sorted(nouns_dict, key=nouns_dict.get, reverse=True):
        nouns_dict_sorted[pos] = nouns_dict[pos]

    # print 50 most common nouns (highest count), from previously sorted dictionary
    print("\n50 Most Common Words: ")
    i=0
    nouns_dict_sorted_50={}
    for k in nouns_dict_sorted:
        if i == 50:
            break
        nouns_dict_sorted_50[k] = nouns_dict_sorted[k]
        i+=1
    print(nouns_dict_sorted_50)

    return list(nouns_dict_sorted_50.keys()) # saved and returned as a list



# tokens are POS-tagged using nltk's built-in library. returns a list of lemmas that are nouns only
def posTag(tokens):

    tokens_tagged = nltk.pos_tag(tokens)
    print("\nFirst 20 Tokens Tagged: ") # print the first 20 tagged
    print(tokens_tagged[:20])
    # make a list with only lemmas that are pos-tagged as nouns
    # NN, NNS, NNP, NNPS, PRP, PRPS
    nouns = [tag for tag in tokens_tagged if tag[1] == "NN" or tag[1] == "NNS" or  tag[1] == "NNP" or tag[1] == "NNPS" or tag[1] == "PRP" or tag[1] == "PRPS"]
    return nouns


# lemmatizes the tokens and returns a list of those lemmatized tokens
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer() # instantiate wordnet lemmatizer
    tokens_lemma=[]

    for t in tokens:    # lemmatize each token
        tokens_lemma.append(lemmatizer.lemmatize(t))
    return tokens_lemma


# calculates the lexical diversity (% of unique tokens) of the text
def calculateLexicalDiversity(tokens):
    return len(set(tokens))/len(tokens)



if __name__ == '__main__':
    if len(sys.argv) < 2:  # check if number of arguments is at least 1, if not terminate program
        print("ERROR: Please enter argument (sysarg) containing input/data file relative path. Re-run program.")
        quit()

    try:
        with open(pathlib.Path.cwd().joinpath(sys.argv[1]), 'r') as f:  # find data file and open
            text_in = f.read()
            guessingGame(text_in) # start game

    except FileNotFoundError:
        print("ERROR: Input/data file provided cannot be found. Please re-run program.")
        quit()

