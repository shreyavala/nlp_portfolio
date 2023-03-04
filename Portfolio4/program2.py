# Name: Shreya Valaboju
# Name: Soham Mukherjee
# Course/Section: CS 4395.001
# Portfolio 4: Ngrams, Program 2
# Notes:
#   - run program 1 before running program 2
#   - ensure all necessary libraries are downloaded/imported
#   - ensure all files are in the same directory as this python file
#   - the 2 execution parameters are test file (1) and the file with the correct language classifications (2)
#         SYSARGV[1] = 'LangId.test'
#         SYSARGV[2] = 'LangId.sol'




#  import necessary libraries
import os
import sys
import pathlib
import pickle
import nltk
from nltk import word_tokenize
from nltk.util import ngrams

nltk.download('punkt')



def compute_accuracy(f):
    """
        computes accuracy from predicted languages from the test file to actual languages
        Parameters:
            f (string): name of file with correct classifications
    """

    accuracy = 0
    total = 0  # total number of classifications made
    incorrect_lines = []  # holds the line numbers of the incorrectly classified lines

    results_file = open("result.txt", "r")  # file with the predicted languages for each line in the test file
    correct_file = open(f, "r") # file with correct classifications for each line in the test file

    results = results_file.readlines()
    correct = correct_file.readlines()

    for l in range(len(results)):
        if results[l] == correct[l]:  # if correctly classified language
            accuracy += 1
        else:  # if incorrectly classified, save line numbers into array
            incorrect_lines.append(correct[l].split(" ")[0])

        total += 1

    accuracy = accuracy / total  # compute accuracy as number of correctly classified lines/total classifications made
    print("Accuracy: ", str(accuracy))
    print("Incorrect Classification Line Numbers: ", incorrect_lines)


# b. For each line in the test file, calculate a probability for each language (see note below) and
#       write the language with the highest probability to a file.
#  Each bigram’s probability with Laplace smoothing is: (b + 1) / (u + v) where b is the bigram count,
#       u is the unigram count of the first word in the bigram, and v is the total vocabulary
#       size (add the lengths of the 3 unigram dictionaries).
def compute_prob(eb, eu, ib, iu, fb, fu, test_data):

    """
        computes probabilities for each language and writes the language with the highest probability to a file
        Parameters:
            eu (dictionary) : english unigram dictionary
            eb (dictionary) : english bigram dictionary
            iu (dictionary) : italian unigram dictionary
            ib (dictionary) : italian bigram dictionary
            fu (dictionary) : french unigram dictionary
            fb (dictionary) : french bigram dictionary
            test_data (string): test file contents
    """

    v = len(eu) + len(iu) + len(fu)  # v is the total vocabulary size (add the lengths of the 3 unigram dictionaries).
    lineNum = 1 # keeps track of the line numbers of the file

    # read test data line by line
    for test in test_data.splitlines():

        # 1. create bigrams, unigrams for test line
        unigrams_test = word_tokenize(test)
        bigrams_test = list(ngrams(unigrams_test, 2))

        # 2. calculate the probability using laplace smoothing for each language-  english, italian, french (refer to prof's notebook)
        # each bigram’s probability with Laplace smoothing is: (b + 1) / (u + v) where b is the bigram count,
        #       u is the unigram count of the first word in the bigram, and v is the total vocabulary
        p_laplace_english = 1
        for bigram in bigrams_test:
            b = eb[bigram] if bigram in eb else 0
            u = eu[bigram[0]] if bigram[0] in eu else 0
            p_laplace_english = p_laplace_english * ((b + 1) / (u + v))

        ###########################################################################
        p_laplace_italian = 1
        for bigram in bigrams_test:
            b = ib[bigram] if bigram in ib else 0
            u = iu[bigram[0]] if bigram[0] in iu else 0
            p_laplace_italian = p_laplace_italian * ((b + 1) / (u + v))

        ###########################################################################
        p_laplace_french = 1
        for bigram in bigrams_test:
            b = fb[bigram] if bigram in fb else 0
            u = fu[bigram[0]] if bigram[0] in fu else 0
            p_laplace_french = p_laplace_french * ((b + 1) / (u + v))

        # 3. write to file which language has the highest probability, and that is the classification that's made
        result_file = open("result.txt", "a")  # holds the classifications for each line, the 'results.'
        highest_probability = max(p_laplace_french, p_laplace_italian, p_laplace_english) # retreive the highest probability
        if highest_probability == p_laplace_english:
            result_file.write(str(lineNum) + " English\n")
        elif highest_probability == p_laplace_french:
            result_file.write(str(lineNum) + " French\n")
        else:
            result_file.write(str(lineNum) + " Italian\n")
        result_file.close()
        lineNum += 1


if __name__ == '__main__':
    if len(sys.argv) < 3:  # check if number of arguments is at least 1, if not terminate program
        print("ERROR: Please enter argument (sysarg) containing training and test files' relative paths. Re-run "
              "program.")
        quit()

    try:
        # Remove results file at the start
        if os.path.exists("result.txt"):
            os.remove("result.txt")

        # a. Read in pickled dictionaries.
        # open pickle file, save bigrams, unigrams for all languages to dictionaries (outputted from executing program1.py)
        eb = pickle.load(open('english_train_bigram.p', 'rb'))
        eu = pickle.load(open('english_train_unigram.p', 'rb'))
        ib = pickle.load(open('italian_train_bigram.p', 'rb'))
        iu = pickle.load(open('italian_train_unigram.p', 'rb'))
        fb = pickle.load(open('french_train_bigram.p', 'rb'))
        fu = pickle.load(open('french_train_unigram.p', 'rb'))

        # the executiion parameters should be the test file (1) and the file with the correct language classifications (2)
        # SYSARGV[1] = 'LangId.test'
        # SYSARGV[2] = 'LangId.sol'

        with open(pathlib.Path.cwd().joinpath(sys.argv[1]), 'r') as f:  # find test file and open
            compute_prob(eb, eu, ib, iu, fb, fu, f.read())  # compute the probabilities using the training data from the pickle files
            compute_accuracy(sys.argv[2])   # compute accuracy of predictions made

    except FileNotFoundError:
        print("ERROR: Input/data file provided cannot be found. Please re-run program.")
        quit()
