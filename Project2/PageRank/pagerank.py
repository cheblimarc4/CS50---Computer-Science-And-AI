import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = dict()

    num_pages = len(corpus)

    num_links = len(corpus[page])


    if corpus[page]:
        for i in corpus:
            probability_distribution[i] = (1 - damping_factor) / num_pages

        for i in corpus[page]:
            probability_distribution[i] += damping_factor / num_links

    else:
        for i in corpus:
            probability_distribution[i] = 1 / num_pages

    return probability_distribution

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    probability_distribution = dict()

    corpus_page = []

    for i in corpus:
        probability_distribution[i] = 0
        corpus_page.append(i)

    page = random.choice(corpus_page)

    for i in range(1, n):
        latest_distribution = transition_model(corpus, page, damping_factor)
        for j in probability_distribution:
            probability_distribution[j] = ((i - 1) * probability_distribution[j] + latest_distribution[j]) / i

        page = random.choices(list(probability_distribution.keys()), list(probability_distribution.values()), k = 1)[0]
        #essayer d'enlever le 0 et de mettre choice sans s
    return probability_distribution

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    probability_distribution = dict()

    N = len(corpus)

    for i in corpus:
        probability_distribution[i] = 1 / N

    changes = True

    while changes:
        changes = False
        pd_copy = copy.deepcopy(probability_distribution)
        for i in corpus:
            probability_distribution[i] = ((1 - damping_factor) / N) + (damping_factor * sigma(corpus, probability_distribution, i))
            changes = changes or abs(pd_copy[i] - probability_distribution[i]) > 0.001

    return probability_distribution

def sigma(corpus, distribution, page):
    s = 0
    for i in corpus:
        if page in corpus[i]:
            s += distribution[i] / len(corpus[i])
    return s


if __name__ == "__main__":
    main()
