import fileinput
import argparse
from collections import defaultdict

def make_tuples(tuple_size, sentences):
    """
    Iterates through the words in the sentences, yielding them as tuples of
    size tuple_size.
    
    >>> list(make_tuples(3, ['hurray for information extraction', 'text mining is super fun']))
    [('hurray', 'for', 'information'),
     ('for', 'information', 'extraction'),
     ('text', 'mining', 'is'),
     ('mining', 'is', 'super'),
     ('is', 'super', 'fun')]
    """
    for sentence in sentences:
        words = sentence.split()
        for i, w in enumerate(words[:-(tuple_size - 1)]):
            yield tuple(words[i:i+tuple_size])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given a bunch of sentences, outputs feature vectors of the words')
    parser.add_argument('-n', default=100, type=int, 
            help='Length of feature vectors (default is 100)')
    parser.add_argument('-f', default='-', type=str, metavar='filename',
            help='File containing sentences to process (defaults to stdin)')

    args = parser.parse_args()
    vecs = defaultdict(lambda : [0]*args.n)
    for w1, w2 in make_tuples(2, fileinput.input(args.f)):
        vecs[w1][hash(w2) % args.n] += 1
        vecs[w2][hash(w1) % args.n] += 1

    for w, v in vecs.iteritems():
        print(w + ' ' + ' '.join(map(str, v)))

