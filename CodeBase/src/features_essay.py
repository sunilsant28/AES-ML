# -*- coding: utf-8 -*-
import sys

sys.path.append("/lib/python2.7/site-packages")
import pandas as pd
from collections import Counter
import numpy as np
import nltk
import math
import enchant as pychant
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

dictionary = pychant.Dict("en_US")

essayKeywords = [
['computers', 'society', 'online', 'technology', 'learn', 'ideas', 'concern', 'time', 'family', 'friends', 'exercise', 'improve', 'help', 'effect', 'distract'],
['censorship', 'censor', 'books', 'magazines', 'children', 'shelf', 'moral', 'police', 'boycott', 'young', 'content', 'learning', 'minds', 'adult', 'media', 'vulgar', 'material'],
['cyclist', 'rough', 'terrain', 'fear', 'scared', 'worry', 'hope', 'hill', 'confidence'],
['persevere', 'feeling', 'spring', 'Saeng'],
['home', 'lives', 'family', 'sense', 'great', 'love', 'parents', 'respect', 'community', 'neighbour'],
[ 'builder', 'building', 'empire', 'state', 'obstacles', 'dirigible' ,'face', 'foundation', 'design', 'building', 'mast', 'airships', 'fly'],
['patience', 'patient', 'understand', 'tolerant', 'time', 'proud', 'wait', 'long'],
['people', 'laughter', 'important', 'relationship', 'laughing', 'worth', 'love', 'moment', 'bit', 'little', 'happiness', 'happy', 'benefits']
]

import re

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def getWordCount(text):
    # split the text into words
    wordList = re.findall(r'\w+', text)

    # print wordList
    # print len(wordList)
    return len(wordList)


def getAvgSentenceLength(text):
    # split the essay into sentences
    sentList = nltk.sent_tokenize(text)

    sumSentLength = 0
    for sent in sentList:
        sumSentLength = sumSentLength + getWordCount(sent)

    # print float(sumSentLength)/len(sentList)
    return float(sumSentLength) / len(sentList)


def getAvgWordLength(text):
    # split the essay into sentences
    wordList = re.findall(r'\w+', text)

    sumWordLength = 0
    for word in wordList:
        sumWordLength = sumWordLength + len(word)

        return float(sumWordLength) / len(wordList)


def getStdDevSentenceLength(text):
    # split the essay into sentences
    sentList = nltk.sent_tokenize(text)

    # mean sentence length
    mean = getAvgSentenceLength(text)

    nr = 0.0
    for sent in sentList:
        nr = nr + (getWordCount(sent) - mean) ** 2
    return math.sqrt(nr / len(sentList))


def getStdDevWordLength(text):
    # split the essay into sentences
    wordList = re.findall(r'\w+', text)

    # mean sentence length
    mean = getAvgWordLength(text)

    nr = 0.0
    for sent in wordList:
        nr = nr + (getWordCount(sent) - mean) ** 2
    # print math.sqrt(nr/len(sentList))
    return math.sqrt(nr / len(wordList))


def getSentenceCount(text):
    # split the essay into sentences
    sentList = nltk.sent_tokenize(text)
    # print sentList
    return len(sentList)


# Read the CSV into a pandas data frame (df)
#   With a df you can do many things
with open('feature.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    dictionary = pychant.Dict("en_US")
    columnName = ["noOfCharacters", "noOfWords", "noOfUniqueWords", "sentCount", "noOfCommas", "noOfSemicolons",
                  "noOfQuestions", "noOfExclamation", "noOfSpellErrors", "meanWorldLength", "avgWordLength",
                  "avgSentenceLength", "stdDeviationSentence", "noOfNounCount", "noOfVerbCount", "noOfAdjCount",
                  "noOfAdvCount", "score"]
    spamwriter.writerow(columnName)
    columnValues = []

    # Read the CSV into a pandas data frame (df)
    #   With a df you can do many things
    df = pd.read_csv('../../Data/training_set_rel3.csv', delimiter=',')

    for i in range(0, len(df)):
        essaySetId = df.at[i, 'essay_set']
        print df.at[i, 'essay_set']
        # essayOneKeywords = ['computers', 'society', 'online', 'technology', 'learn', 'ideas', 'concern', 'time','family', 'friends', 'exercise', 'improve', 'help', 'effect', 'distract']
        essayOneKeywords = essayKeywords[essaySetId]
        text = df.at[i, 'essay']
        score = df.at[i, 'domain1_score']
        essay = "".join(i for i in text if ord(i) < 128)
        essay = unicode(essay, errors='ignore')

        essay = " ".join(filter(lambda x: x[0] != '@',
                                essay.split()))  # To remove proper nouns tagged in the data-set which may result into false positives during POS tagging.

        stop_words = set(stopwords.words('english'))
        print stop_words
        # stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
        word_tokens = word_tokenize(essay)

        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)

        ##print(word_tokens)
        ##print(filtered_sentence)

        essarCounter = Counter(essay)
        ##ƒprint essay

        noOfCharacters = len(essay)
        words = essay.split()

        noOfSpellErrors = 0
        keywordRelevance = 0;

        for word in words:
            if not dictionary.check(word):
                ##print word
                noOfSpellErrors = noOfSpellErrors + 1
            for keyword in essayOneKeywords:
                if (re.search(keyword, word)):
                    print keyword
                    keywordRelevance = keywordRelevance + 1
            if word in stop_words:
                words.remove(word)

        wordList = re.findall(r'\w+', essay)

        noOfWords = len(words)
        counter = Counter(words)
        # print Counter(words)

        noOfUniqueWords = len(counter)

        wordLengthSum = sum(counter.itervalues())

        avgSentenceLength = getAvgSentenceLength(essay)

        stdDeviationSentence = getStdDevSentenceLength(essay)

        sentCount = getSentenceCount(essay)

        noOfCommas = essarCounter[',']
        noOfSemicolons = essarCounter[',']
        noOfQuestions = essarCounter['?']
        noOfExclamation = essarCounter['!']

        meanWorldLength = wordLengthSum / getWordCount(essay)
        # print counter.itervalues()

        # medianWorldLength = np.median(counter.itervalues())
        # stdDeviationWordLength = np.std(counter.itervalues())
        avgWordLength = getAvgWordLength(essay)
        stdDeviationWord = getStdDevWordLength(essay)

        print "no of characters", noOfCharacters
        print "no of words", noOfWords
        print "no of unique words", noOfUniqueWords
        print "No of sentences: ", sentCount
        print "Avg Sentence length", avgSentenceLength
        print "Avg Word length", avgWordLength
        print "Standard Deviation Sentence", stdDeviationSentence
        print "Standard Deviation word", stdDeviationWord
        print "no of commas", noOfCommas
        print "no of semi colons", noOfSemicolons
        print "no of questions", noOfQuestions
        print "no of exclamation", noOfExclamation
        print "no of spelling errors", noOfSpellErrors
        print "mean value of word length", meanWorldLength

        ### structural done now information
        ## code provided by aanjan

        ## syntactical analysis

        # POS TAGS
        count = Counter([j for i, j in nltk.pos_tag(words)])

        noOfNounCount = count['NN'] + count['NNS'] + count['NNPS'] + count['NNP']
        noOfVerbCount = count['VB'] + count['VBG'] + count['VBP'] + count['VBN'] + count['VBZ']
        noOfAdjCount = count['JJ'] + count['JJR']
        noOfAdvCount = count['RB'] + count['RBR'] + count['RBS']

        print "Syntactical features "
        print "no of Nouns", noOfNounCount
        print "no of Verbs", noOfVerbCount
        print "no of Adjectives", noOfAdjCount
        print "no of Adverbs", noOfAdvCount

        ### information extraction
        ## for keywords relevance we can write a util but currently it is manual
        print "relevant keywords ", keywordRelevance
        columnValues = [noOfCharacters, noOfWords, noOfUniqueWords, sentCount, noOfCommas, noOfSemicolons,
                        noOfQuestions, noOfExclamation, noOfSpellErrors, meanWorldLength, avgWordLength,
                        avgSentenceLength, stdDeviationSentence, noOfNounCount, noOfVerbCount, noOfAdjCount,
                        noOfAdvCount, score]
        print columnName
        print columnValues
        spamwriter.writerow(columnValues)