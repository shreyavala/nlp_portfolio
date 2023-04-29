# Name: Shreya Valaboju
# Name: Soham Mukherjee
# Course/Section: CS 4395.001
# Program 1, Portfolio 4: Ngrams
# Notes:
#   - run this file before running program2.py
#   - ensure all necessary libraries are downloaded/imported
#   - all files should be in the same directory as this python file
#   - sysargv[1] should be 1 of the 3 language's training data (i.e. 'LangId.train.English')
#   - sysargv[2] should be another of the 3 language's training data (i.e. 'LangId.train.Italian')
#   - sysargv[3] should be the last of the 3 language's training data (i.e. 'LangId.train.French')
#   - each language should have 1 unigrams dictionary pickle file and another for bigrams
#   - this program may take a couple of seconds to completely execute, this is normal


# import necessary libraries
import sys
import pathlib
import pickle
import nltk
from nltk import word_tokenize
from nltk.util import ngrams

nltk.download('punkt')


# use the bigram list to create a bigram dictionary of bigrams and counts, [‘token1 token2’] -> count
def createBigramDictionary(bigrams):
    bigrams_dict = {}
    print("Creating Bigrams Dictionary...")
    for b in set(bigrams):
        bigrams_dict[b] = bigrams.count(b) # count occurrences of how many times the bigram occurs
    return bigrams_dict


# generate unigrams using nltk's ngrams(), generator
def generateUnigrams(unigrams):
    print("Creating Unigram Dictionary...")
    unigram_dict = {t: unigrams.count(t) for t in set(unigrams)} # count occurrences of how many times the unigram occurs
    return unigram_dict


# generate bigrams using nltk's ngrams()
def generateBigrams(tokens):
    bigrams = list(ngrams(tokens, 2))
    return bigrams


# tokenize and perform basic preprocessing (check if this is necessary)
def tokenize(text):
    text = text.lower()
    tokens_arr = word_tokenize(text)
    tokens = [t for t in tokens_arr if t.isalpha()]  # only count alphabetical tokens, no numbers or punctuation
    return tokens


def start(text_in):
    """
        computes accuracy from predicted languages from the test file to actual languages
        Parameters:
            text_in (string): file contents of a language's training data
        Returns:
            unigram_dict (dictionary): dictionary of unigrams and counts, [‘token1] -> count
            bigrams_dict (dictionary): dictionary of bigrams and counts, [‘token1 token2’] -> count
    """
    tokens = tokenize(text_in)  # start program

    # create a bigram dictionary of counts
    bigrams_list = generateBigrams(tokens)
    bigrams_dict = createBigramDictionary(bigrams_list)
    print(bigrams_dict)

    # create a unigram dictionary of counts
    unigram_dict = generateUnigrams(tokens)
    print(unigram_dict)

    return unigram_dict, bigrams_dict


# pickle the unigram and bigram dictionaries
def pickleDictionaries(uni_dict, bi_dict, lang):
    pickle.dump(uni_dict, open(lang + '_train_unigram.p', 'wb'))
    pickle.dump(bi_dict, open(lang + '_train_bigram.p', 'wb'))


if __name__ == '__main__':
    if len(sys.argv) < 4:  # check if number of arguments is at least 1, if not terminate program
        print("ERROR: Please enter argument (sysarg) containing input/data file relative path. Re-run program.")
        quit()

    try:
        # get paths for all 3 language training data files
        file_path1 = pathlib.Path.cwd().joinpath(sys.argv[1])
        file_path2 = pathlib.Path.cwd().joinpath(sys.argv[2])
        file_path3 = pathlib.Path.cwd().joinpath(sys.argv[3])

        lang1 = str(file_path1).split("train.")[1].lower()
        lang2 = str(file_path2).split("train.")[1].lower()
        lang3 = str(file_path3).split("train.")[1].lower()

        # for each language, create unigram and bigram dictionaries
        with open(file_path1, 'r', encoding="utf8") as f:  # find data file and open
            text_in = f.read()
            print("Language 1: ")
            unigram_dict, bigram_dict = start(text_in)
            # pickle unigram and bigram dictionaries into files
            pickleDictionaries(unigram_dict, bigram_dict, lang1)

        with open(file_path2, 'r', encoding="utf8") as f:  # find data file and open
            text_in = f.read()
            print("Language 2: ")
            unigram_dict, bigram_dict = start(text_in)
            # pickle unigram and bigram dictionaries into files
            pickleDictionaries(unigram_dict, bigram_dict, lang2)

        with open(file_path3, 'r', encoding="utf8") as f:  # find data file and open
            text_in = f.read()
            print("Language 3: ")
            unigram_dict, bigram_dict = start(text_in)
            # pickle unigram and bigram dictionaries into files
            pickleDictionaries(unigram_dict, bigram_dict, lang3)


    except FileNotFoundError:
        print("ERROR: Input/data file provided cannot be found. Please re-run program.")
        quit()