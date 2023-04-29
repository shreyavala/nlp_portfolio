# Names: Shreya Valaboju, Soham Mukherjee
# Course/Section: CS 4395.001
# Portfolio 6: Web Crawler
# Description: This program scrapes information from the Dallas Mavericks Wikipedia page. It extracts sentences and important terms
#   using tf-idf. We manually determined 10 terms to use for a chat (to be developed later). We developed a primlimary knowledge base
#   to be used for the chatbot as well, which is printed to the console.

# How to Run:
#   - there are no additional sysargs, run the program normally.
#   - $python main.py
#   - ensure all the necessary libraries are downloaded prior to execution

# import libraries
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import os
import re
import pickle
import math
import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from pprint import pprint

stopwords = stopwords.words('english')
nltk.download('punkt')


# this function gets the players name from the pickled dictionary
def get_players():
    # Send a request to the URL and create a BeautifulSoup object
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Find the table containing the roster information
    roster_table = soup.find("table", {"class": "toccolours"})

    # Find all the rows in the table
    rows = roster_table.find_all("tr")

    players = []

    # Loop over the rows and print out their contents
    for row in rows[1:]:
        # Check if the row contains the table header, skip if it does
        if row.find("th"):
            continue
        # Get the third column, which contains player name
        raw_name = row.find_all("td")[2].get_text().strip()
        # Replace non-breaking space character with regular space
        raw_name = raw_name.replace('\xa0', '_')
        # remove (TW) flag
        raw_name = raw_name.replace("(TW)", "")
        # Modify the name
        player = "_".join(raw_name.split(",")[::-1]).strip()
        if player == "McKinley IV__Wright":
            player = "Luka_Doncic"
        if player == "Jr._ Tim_Hardaway":
            player = "Davis_Bertans"
        players.append(player)

    return players


# this function scrapes and displays details about the players from the url to use in the knowledge base
def display_info(player_url):
    # Send a GET request to the URL and store the response
    response = requests.get(player_url)

    # Parse the HTML content of the response using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the infobox on the page
    infobox = soup.find('table', {'class': 'infobox vcard'})

    # Create an empty dictionary to store the statistics
    stats = {}

    player_name = player_url.split("/wiki/")[1]
    stats['name'] = player_name.replace("_", " ")

    # Loop through all rows in the infobox
    for row in infobox.find_all('tr'):
        # Find the header cell and data cell for each row
        header = row.find('th')
        data = row.find('td')

        # If both the header and data cells exist, add them to the dictionary
        if header and data:
            # Use the header text as the key and the data text as the value
            stats[header.get_text().strip()] = data.get_text().strip()

    # remove any special characters in stats
    for key, value in stats.items():
        stats[key] = value.replace('\xa0', ' ').replace('\u200b', '').replace('\ufeff', '').replace('–', '-').replace(
            '\n', '~').replace('\u2192', '').replace('\u00fc', 'u')
    new_stats = dict()
    for k in stats.keys():
        old_key = k
        new_key = old_key.replace('\u2013', '-').replace('\u00a0', '')
        new_stats[new_key] = stats[old_key]
    # Print the dictionary of statistics
    print(new_stats)
    return new_stats


# Unpickles the players' info and prints it accordingly
def print_players(fp_name):
    players_pickle = open(fp_name, "rb")
    players_info = pickle.load(players_pickle)
    pprint(players_info)


# This function builds a searchable knowledge base of facts that a chatbot (to be developed later) can
# share related to the 10 terms manually picked. For our program, the knowledge base is a python dictionary
def knowledge_base(terms):
    players_dict = {}
    players = get_players()
    url_starter = "https://en.wikipedia.org/wiki/"

    print("Knowledge Base: ")

    for term in terms:
        for player in players:
            if term.lower() in player.lower():
                # given an existing player, get the url and begin parsing info
                player_url = url_starter + player
                player_dict = display_info(player_url)
                players_dict[player] = player_dict
    with open('players.p', 'wb') as f:
        # Dump the binary dictionary into the file.
        pickle.dump(players_dict, f)

    fp_name = 'players.p'
    print_players(fp_name)


# Manually determined the top 10 terms (players) from step 4, based on the domain knowledge
def get_top_ten(top_ten_terms):
    # sort to get top ten, replace temporary players with popular ones
    # top_ten_terms = sorted(top_ten_terms, key=lambda x: x[1], reverse=True)[:10]
    # top_ten_terms = top_ten_terms[:10]

    top_ten_terms_list = ['doncic', 'irving', 'wood', 'bertans', 'powell', 'bullock', 'kleber', 'ntilikina', 'mcgee',
                          'pinson']

    # for i in range(len(top_ten_terms)):
    #     if top_ten_terms[i][0] == 'lawson':
    #        top_ten_terms[i] = ('doncic', 0.9)
    #    if top_ten_terms[i][0] == 'wright':
    #        top_ten_terms[i] = ('irving', 0.89)
    #    if top_ten_terms[i][0] == 'bertāns':
    #        top_ten_terms[i] = ('bertans', 0.88)

    print("\nTop 10 terms:")
    top_ten = []
    for i in range(len(top_ten_terms_list)):
        # print(str(i + 1) + ": " + top_ten_terms[i][0])
        # top_ten.append(top_ten_terms[i][0])
        print(str(i + 1) + ": " + top_ten_terms_list[i])
        top_ten.append(top_ten_terms_list[i])

    print("\n")
    knowledge_base(top_ten)


# this function calculates term frequencies of all documents
def tf(text):
    tf_dict = {}
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in stopwords]  # extract alpha and non-stopwords only

    # get term frequencies in a more Pythonic way
    token_set = set(tokens)
    tf_dict = {t: tokens.count(t) for t in token_set}

    # normalize tf by number of tokens
    for t in tf_dict.keys():
        tf_dict[t] = tf_dict[t] / len(tokens)

    return tf_dict


# tf-idf dictionaries are created for each file
def create_tfidf(tf, idf):
    tf_idf = {}
    for t in tf.keys():
        tf_idf[t] = tf[t] * idf[t]
    return tf_idf


# the function extracts at least 25 important terms from the pages using an importance measure such as term frequency, or tf-idf.
def tfidf(sentence_files):
    vocab_per_url = []
    tf_dicts_all = []
    vocab = set()

    for sf in sentence_files:
        current_file = open(sf, "r", encoding="utf-8")  # open file to read

        # lowercase, remove punctuation
        text = current_file.read().lower()
        text = re.sub(r'[^\w\s]', '', text)

        # get tf (term frequencies) dictionaries for each file
        tf_dict = tf(text)
        vocab_per_url.append(tf_dict.keys())
        tf_dicts_all.append(tf_dict)

        # add to vocab
        vocab = vocab.union(set(tf_dict.keys()))

    # get idf
    idf_dict = {}
    for term in vocab:
        temp = ['x' for voc in vocab_per_url if term in voc]
        idf_dict[term] = math.log((1 + len(sentence_files)) / (1 + len(temp)))

    # get tf-idf dictionary for each document and print top 25 terms from each
    url_num = 0
    top_ten_terms = list()

    for termf in tf_dicts_all:
        tfd = create_tfidf(termf, idf_dict)
        doc_term_weights = sorted(tfd.items(), key=lambda x: x[1], reverse=True)
        # print("\nPlayer " + str(url_num) + " top 25 terms: ", doc_term_weights[:25])
        print(sentence_files[url_num] + " top 25 terms: ", doc_term_weights[:25])
        top_ten_terms.extend(doc_term_weights)
        url_num += 1

    get_top_ten(top_ten_terms)  # call to get top 10 terms


# the function to cleans up the text from each file. we deleted newlines and tabs.
#   and extracted sentences with NLTK’s sentence tokenizer and wrote those sentences to a new file
def clean(files_arr):
    sentences_files = []

    # iterate through each file, 'url_.txt'
    for f in files_arr:
        current_file = open(f, "r", encoding="utf-8")  # open file to read
        text_in = current_file.read()

        # remove newlines, tabs
        text_in = text_in.strip()
        text_in = re.sub('\s+', ' ', text_in)

        # lowercase, everything between brackets, parenthesis
        text_in = re.sub("\(.*?\)", "", text_in)
        text_in = re.sub("\[.*?\]", "", text_in)

        # extract sentences using nltk's tokenizer, write sentences to a new file
        sent_arr = sent_tokenize(text_in)
        fname = "cleaned_" + f
        cleaned_file = open(fname, 'a', encoding="utf-8")
        for sentence in sent_arr:
            cleaned_file.write(sentence + "\n")
        sentences_files.append(fname)
        cleaned_file.close()

    # call to tf-idf function
    tfidf(sentences_files)


# this function 'crawls' through the Dallas Mavericks Wiki page and uses beautifulsoup to scrape text and urls within
# the domain.
def webcrawl(base_url):
    urls = []  # holds a list of urls found within and outside the domain
    files = []  # stores the files created for each link

    # base page and soup objects created for the base domain, the dallas mavs wiki page
    base_page = requests.get(base_url)
    base_soup = BeautifulSoup(base_page.content, 'html.parser')

    # get players links
    name_column_number = 0
    table = base_soup.find('table', class_='sortable')  # the table with information about each player
    # table = base_soup.find_all('table')[0] - for link 7
    for row in table.tbody.find_all('tr'):  # iterate through each row of the table
        name_column_number = 0
        columns = row.find_all('td')  # get the columns of the table
        for td in columns:
            if td.a and name_column_number == 2:  # the second column holds the name of each player hyperlinked to
                # their own wiki pages
                urls.append(urljoin(base_url, td.a.get('href')))
            name_column_number += 1

    # get external links (5) from reference list of the domain
    num_external_links = 0
    reference_data = base_soup.findAll('div', attrs={'class': 'reflist'})
    for div in reference_data:
        links = div.findAll('a')
        for a in links:
            if num_external_links == 5:
                break
            link_str = urljoin(base_url, a.get('href'))  # link string
            # exclude links that are within the same domain and are not scrapable, such as pdfs, jpgs, etc.
            if 'wiki' not in link_str and 'pdf' not in link_str and 'jpg' not in link_str and 'mavs.com' not in link_str and 'web.archive' in link_str:
                urls.append(urljoin(base_url, link_str))
                # print(urljoin(base_url, link_str))
                num_external_links += 1

    # loop through the URLs and scrape all text off each page. Store each page’s text in its own file
    for i in range(len(urls)):
        # create new soup objects and request for each link we scrape from
        page = requests.get(urls[i])
        while page.status_code == 404:  # skip over urls that return a 404 error
            print("404: ", urls[i], i)
            i += 1
            page = requests.get(urls[i])
        soup = BeautifulSoup(page.content, 'html.parser')

        file_name = 'url' + str(i) + '.txt'  # name of the file to be created
        has_text = False
        for p in soup.select('p'):
            url_file = open(file_name, "a", encoding="utf-8")  # append the scraped text to the newly created file
            url_file.write(p.get_text() + '\n')
            has_text = True
            # print(p.get_text())
        if has_text:
            files.append(file_name)

    clean(files)


if __name__ == '__main__':

    url = 'https://en.wikipedia.org/wiki/Dallas_Mavericks'  # starting/base domain url

    # Remove all url text files at the start
    for i in range(50):
        file_name = 'url' + str(i) + '.txt'
        file_name2 = 'cleaned_url' + str(i) + '.txt'
        if os.path.exists(file_name):
            os.remove(file_name)
        if os.path.exists(file_name2):
            os.remove(file_name2)

    webcrawl(url)  # begin web crawl function
