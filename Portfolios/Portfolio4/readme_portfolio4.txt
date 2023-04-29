Author: Shreya Valaboju, Soham Mukherjee
Date: 3/4/2023


Description:

This program will have 3 languages, Italian, English, and French to train on. Program 1 will 
create unigrams and bigrams out of the training data and store them into dictionaries. After the
unigrams and bigrams are created, Program 2 will test by calculating probabilities for a given test file using
laplace smoothing. For each line in the test file, the language which has the highest probability will be written to the file 'results.txt.'
Program 2 will also compute the accuracy as the percentage of correctly classified instances in the
test set. A narrative is also written which gives an overview of ngrams. 


Training Files: data/LangId.train.English, data/LangId.train.Italian, data/LangId.train.French
Test File: data/LandId.test
Correct Classifications File: data/LandId.sol

How to Execute:
    - Please run program 1 BEFORE program 2
    - ensure all necessary modules are installed before execution

    Program 1 Example:
        each execution paramater aside from the program path should be one of the 3 language training data, below is an example
        $ python program1.py data/LangId.train.English data/LangId.train.French data/LangId.train.Italian

    Program 2 Example:
        each execution paramater aside from the program path should be the  test file (1) and the file with the correct language classifications (2)
        $ python program2.py data/LangId.test data/LangId.sol


