import nltk
import sys
import os
import math

import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file = dict()

    for i in os.listdir(directory):
        with open(os.path.join(directory, i), "r") as f:
            file[i] = f.read()

    return file


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokenized = nltk.tokenize.word_tokenize(document.lower())
    l = []
    for i in tokenized:
        if i not in string.punctuation:
            if i not in nltk.corpus.stopwords.words("english"):
                l.append(i)
    return l


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf = dict()

    words = set(word for i in documents.values() for word in i)

    for word in words:
        count = 0
        for i in documents.values():
            if word in i:
                count += 1
        idf[word] = math.log(len(documents) / count)

    return idf

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    score = dict()

    for file, word in files.items():
        score[file] = 0
        for i in query:
            score[file] += word.count(i) * idfs[i]

    sorted_score = sorted(score.items(), key = lambda x: x[1], reverse = True)
    sorted_score = [x[0] for x in sorted_score]

    return sorted_score[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    score = dict()
    for sentence, swords in sentences.items():
        s = 0
        for word in query:
            if word in swords:
                s += idfs[word]

        if s != 0:
            density = sum([swords.count(x) for x in query]) / len(swords)
            score[sentence] = (s, density)

    sorted_score = [k for k, v in sorted(score.items(), key = lambda x:(x[1][0], x[1][1]), reverse = True)]

    return sorted_score[:n]




if __name__ == "__main__":
    main()
