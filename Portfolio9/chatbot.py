# Portfolio 7: Chatbot
# Names: Shreya Valaboju (sxv180047), Soham Mukherjee (sxm180113)
# CS4395.001


# Description: Chatbot, "Champ," that answers questions about players on the Dallas Mavericks 2022/2023 team. Refer to the README for
#   instructions on how run this chatbot. For sample dialog, analysis, and an in-depth description, refer to the Chatbot Report.
#   The report, code, data/knowledge base, and readme files can be located in the github repos:
#   Shreya's Repo: https://github.com/shreyavala/nlp_portfolio
#   Soham's Repo: https://github.com/Zakenmaru/CS4395_Portfolio


# import libraries
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import random
from nltk.corpus import wordnet
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
import re
from pprint import pprint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

players_info = {}  # knowledge base with player info
intents_dict = {}  # basic intents not about players specifically


# print and store the players knowledge base
def printKnowledgeBase(fp_name):
    players_pickle = open(fp_name, "rb")
    players_info = pickle.load(players_pickle)
    #pprint(players_info)
    return players_info


# gets all the synonyms for a specfic topic
def getTopicSynonyms(topics_list):
    synonyms = []
    syn_dict = {}

    # iterate through each topic from the knowledge base
    for topic in topics_list:
        for syn in wordnet.synsets(topic):  # get all synonyms for each topic and put them into a list
            for l in syn.lemmas():
                synonyms.append(l.name())
        syn_dict[topic] = synonyms

    return syn_dict

# checks if user is talking about something unrelated to players
def checkIntents(user_query):
    intent_not_about_player = False

    if user_query in intents_dict['greet']['patterns']:  # check if the user is just greeting
        print("Champ: " + random.choice(intents_dict['greet']['responses']))
        intent_not_about_player = True
    elif user_query in intents_dict['thanks']['patterns']:  # check if the user is thanking the bot
        print("Champ: " + random.choice(intents_dict['thanks']['responses']))
        intent_not_about_player = True
    elif user_query in intents_dict['goodbye']['patterns']:  # check if the user is saying bye to the bot
        print("Champ: " + random.choice(intents_dict['goodbye']['responses']))
        intent_not_about_player = True
    elif user_query in intents_dict['funny'][
        'patterns']:  # check if the user is asking about the pheonix suns (our rival)
        print("Champ: " + random.choice(intents_dict['funny']['responses']))
        intent_not_about_player = True
    elif user_query in intents_dict['goat'][
        'patterns']:  # check if the user is asking who is the best player, who is the GOAT
        print("Champ: " + random.choice(intents_dict['goat']['responses']))
        intent_not_about_player = True
    elif user_query in intents_dict['mark_cuban']['patterns']:  # check if the user is asking about mark cuban
        print("Champ: " + random.choice(intents_dict['mark_cuban']['responses']))
        intent_not_about_player = True
    elif user_query in intents_dict['general_mavs']['patterns']:  # check if the user is asking general mavs/nba question
        print("Champ: " + random.choice(intents_dict['general_mavs']['responses']))
        intent_not_about_player = True

    return intent_not_about_player


# grabs general player info, if specific topic not found
def getGeneralPlayerInfo(player_name):
    name = player_name.replace("_", " ")
    print("\t" + name + " is a player on the 2022-2023 Dallas Mavericks team. "
                        "\n\t He has been playing basketball since " + players_info[player_name]['Playing career'] +
          "\n\t He plays " + players_info[player_name]['Position'])


# use cosine similarity and tf-idf vectorization to match a player to what the user input is or who the user is
# asking about
def train(words):
    player_names = list(players_info.keys())  # list of player names

    topic_keywords = ["Born", "College", "NBA draft", "High school", "weight", "height", "Position",
                      "Playing career", "Men's basketball", "Nationality", "League", "Past Teams"]  # topics that we have knowledge on
    topic_dict = getTopicSynonyms(
        topic_keywords)  # topics can be mentioned in a different way, like "born" is also "birth"
    similarity_scores_names = []
    user_query = ' '.join(words)

    if checkIntents(user_query):  # check if the user is talking about a player or not
        return

    # check if the user is talking about any of the players we have info on
    new_player_names = []  # holds names in a more readable way, like "Luka_Doncic" -> "Luka Doncic"
    for n in player_names:
        n = n.split("_")
        if n[0].lower() == 'tim jr.':  # special case for player, Tim Hardaway jr.
            n[0] = 'tim'
        new_player_names.append(n[0].lower())
        new_player_names.append(n[1].lower())

    intersection_players = len(
        [i for i in new_player_names if i in words])  # check if the user mentions any of the players we have info on

    if intersection_players == 0:  # no players in user query, can't understand -> output default
        scores = SentimentIntensityAnalyzer().polarity_scores(user_query)
        if scores['neg'] > 0:  # if the sentiment is overly negative, apologize to the user
            print(
                "Champ: Sorry you feel this way. Try asking me something about a player (general/specific). I will try my best to help you.")
        else:
            print("Champ: I'm happy to help you, but try asking me something about a player.")
        return

    # find the closest player name to what the user is asking about, using cosine similarity and tf-idf vectorizer
    for name in player_names:
        name = name.replace("_", " ")
        vectorizer = TfidfVectorizer().fit_transform([user_query, name])
        similarity_scores_names.append(cosine_similarity(vectorizer[0], vectorizer[1])[0][0])

    most_similar_player_idx = similarity_scores_names.index(
        max(similarity_scores_names))  # index of most similar player name in list
    player_name = player_names[most_similar_player_idx]  # get player name from dictionary

    # find similarity between user query and topic_keywords, try to understand what topic they're talking about
    similarity_scores = []
    topic = ""
    topic_synonym_exists = False
    for topic in topic_keywords:  # using cosine similarity and tf-idf vectorizer
        vectorizer = TfidfVectorizer().fit_transform([user_query, topic])
        similarity_scores.append(cosine_similarity(vectorizer[0], vectorizer[1])[0][0])

    most_similar_topic_idx = similarity_scores.index(max(similarity_scores))  # index of most similar topic name in list

    # if they're talking about a player, but can't match to a topic
    if all(sim_score == 0 for sim_score in similarity_scores):
        # try to see if a synonym is being said and match to topic
        for t in topic_dict.keys():
            if topic_synonym_exists:  # don't go through all topics
                break
            for syn in topic_dict[t]:  # for all synonyms
                if syn != t and syn in words:  # found a synonym the user mentions
                    topic_synonym_exists = True
                    topic = t
                    break
        if not topic_synonym_exists:
            print("Champ: Could not find that specific information about " + player_name.replace("_", " "))
            print("Champ: Here are some facts about " + player_name.replace("_", " ") + ": ")
            # pprint(players_info[player_name])
            getGeneralPlayerInfo(player_name)
            return

    if not topic_synonym_exists:  # no synonym, topic said verbatim by the user
        topic = topic_keywords[most_similar_topic_idx]

    # find the player and the related topic and print info found in players dictionary
    if topic in players_info[player_name]:
        print("Champ: Here's some information about", player_name.replace("_", " "), "relating to", topic)
        if topic != "Past Teams":
            pprint(players_info[player_name][topic])
        else:
            # raw_teams = players_info[player_name][topic][2:-2]
            past_teams_list = []
            for team in players_info[player_name][topic]:
                past_teams_list.append(set(team.items()))

            # Sort in proper order
            teams_list = sorted(list(past_teams_list[0]), key=lambda x: x[0])

            for idx, team in enumerate(teams_list, 1):
                print(f"{idx}. {team[0]}: {team[1]}")
    else:
        print("Champ: Could not find information about " + player_name.replace("_",
                                                                               " ") + " and their " + topic + "; it may not exist. Try again?")


# lemmatize, remove stop words, numbers, lowercase
def preprocess(text_in):
    lemma = nltk.stem.WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    word_tokens = nltk.word_tokenize(text_in.lower())  # tokenize into words and lowercase
    word_tokens_l = []

    for word in word_tokens:  # remove punctuation and try lemmatize player names
        word = re.sub(r'[^\w\s]', '', word)  # remove punctuation
        if word != '':  # make sure word is not empty
            word_tokens_l.append(word)

    word_tokens_l = [word for word in word_tokens_l if word not in stop_words]  # remove stop words
    word_tokens_l = [lemma.lemmatize(word) for word in word_tokens_l]  # lemmatize

    # get player info by using cosine-similarity to find most similar player in user-input
    train(word_tokens_l)


# asks user about likes/dislikes and updates the user model
def getUserModel(users, user_name):
    if user_name in users:
        print(f"Welcome back, {user_name}!")
        user_info = users[user_name]
    else:
        user_info = {"name": user_name, "personal_info": {}, "likes": [], "dislikes": []}
        users[user_name] = user_info

    # Get personalized remarks from the user model
    if user_info["likes"]:
        print(f"Champ: By the way {user_name}, I remember you said you like {', '.join(user_info['likes'])}.")
    if user_info["dislikes"]:
        print(f"Champ: Also, I remember you said you don't like {', '.join(user_info['dislikes'])}.")

    # Update user model based on user's response
    print(f"Champ: {user_name}, is there anything else you'd like me to know about you?")
    new_info = input("Likes or dislikes, specifically? ")

    likes = []
    dislikes = []

    analyzer = SentimentIntensityAnalyzer()
    phrases = re.split(r'[^\w\s]', new_info)  # split on all punctuation
    for phrase in phrases:
        sentiment = analyzer.polarity_scores(phrase)

        for i, word in enumerate(phrase.lower().split()):
            if sentiment['compound'] > 0 and word in ["love", "like", "enjoy"]:
                likes.append(phrase.split()[i + 1])
            elif sentiment['compound'] < 0 and word in ["hate", "dislike", "don't like"]:
                dislikes.append(phrase.split()[i + 1])

    if likes:
        user_info["likes"].extend(likes)

    if dislikes:
        user_info["dislikes"].extend(dislikes)

# load existing user data from file, or initialize new file if it doesn't exist
def loadUsers():
    try:
        with open("data/users.json", "r") as f:
            users = json.load(f)
    except FileNotFoundError:
        users = {}
    return users


# update the user model
def updateUser(users):
    with open("data/users.json", "w") as f:
        json.dump(users, f)


# the intents file handles basic things like greetings, goodbye, random questions about mark cuban, etc.
def loadIntents(intents_file):
    with open(intents_file, 'r') as f:
        intents_file = json.load(f)

    for i in intents_file['intents']:
        intents_dict[i['tag']] = {
            'patterns': i['patterns'],
            'responses': i['responses']
        }


# begins chat session and conversation with user
def chat():
    users = loadUsers()

    # print intro message and how to use chatbot
    print("Champ: Howdy! My name is Champ, and I'm a bot who knows all about the players on the Dallas Mavericks!")
    print("Ask me anything or something specific about a player from the ")
    print("2022/2023 Roster:")
    print(
        "\t\n" + "0. Justin Holiday" + "\t\n" + "1. Theo Pinson""\t\n" + "2. Kyrie Irving""\t\n" + "3. Jaden Hardy""\t\n" + "7. Dwight Powell" +
        "\t\n" + "8. Josh Green" + "\t\n" + "11. Tim Hardaway Jr." + "\t\n" + "13. Markieff Morris" + "\t\n" + "21. Frank Ntilikina" +
        "\t\n" + "25. Reggie Bullock" + "\t\n" + "35. Christian Wood" + "\t\n" + "42. Maxi Kleber" + "\t\n" + "44. Davis Bertans" +
        "\t\n" + "77. Luka Doncic" + "\t\n" + "00. Javale McGee" + "\n")
    print("You can ask me things like: "
          "\t\n 'What is Kyrie Irving's height?' \t\n 'tell me about Luka Doncic's NBA draft' \t\n 'Where did Javale go to college?'\n")
    print("I have information about a player's college, high school, height, weight, playing career, draft, teams, "
          "and more!")
    print("Type 'quit' to end session and to stop chatting.")
    user_name = input("\nEnter your name: ")

    getUserModel(users, user_name)  # retrieve and update user model

    print("Champ: Howdy " + user_name + "! Go ahead, ask me question")
    while True:
        user_input = input(user_name + ": ")
        if 'quit' in user_input.lower():
            break
        else:
            preprocess(user_input)  # preprocess user questions

    updateUser(users)
    print(
        "Champ: Thanks for chatting, " + user_name + "! I hope I answered all your questions, and always - Go Mavs! :)")


if __name__ == '__main__':
    players_info = printKnowledgeBase("data/players.p")
    loadIntents('data/intents.json')
    chat()